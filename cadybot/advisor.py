"""What to ask the model, and how to render what comes back.

Two entry points: `ask` (a verdict on one question) and `brief` (ranked
recommendations). Both take the deterministic snapshot as input — the model is
only ever asked to interpret numbers, never to produce them.

Output is schema-constrained on both backends, so a malformed response is
rejected rather than merely unlikely.

Two properties of the schemas below are load-bearing and easy to undo by
accident:

- Field order is the generation order. Both backends decode in declared order
  (Ollama compiles the schema to a grammar; the Anthropic output_format follows
  the declaration), so a schema that names its conclusion first gets a
  conclusion written before any reasoning exists to support it. Tam et al.
  (EMNLP 2024) measured Claude-3-Haiku on GSM8K falling from 86.51% to 23.44%
  under JSON mode and traced it to exactly this: 100% of responses put the
  answer before the reason. Reasoning fields therefore come first here, and the
  renderers put the conclusion back on top for the reader.
- Nothing in `Brief` expresses an opinion about a past recommendation. Grading
  belongs to scorecard.py, which no model touches; verdicts arrive in the
  prompt as facts, and the model's only job around them is writing the sentence.
"""

import json
import re
from typing import Any, Dict, List, Literal, Optional, Set

from pydantic import (
    BaseModel, Field, PrivateAttr, create_model, field_validator, model_validator,
)

from . import agenda, config, db, llm, plays, prompts, scorecard, snapshot

# Re-exported so callers catch one name regardless of backend.
Refused = llm.Refused
BackendError = llm.BackendError

# The exact metric names a recommendation may commit to. Rendered into the
# schema as an enum, so an invented metric is rejected by the decoder rather
# than discovered at grading time, when the only remaining options are to guess
# or to drop the row.
METRIC_CHOICES: List[str] = list(snapshot.SCOREABLE_METRICS) + ["none"]

# The narrower set a reflection may name. Event counts only: everything in
# NON_COUNT_METRICS is a share, an age, a stock or a bounded index, and the
# member that matters is activity.days_since_owner_posted, which rises by 1.0
# every day precisely because nobody posts. A thought that says "watch this"
# about a clock schedules itself forever and learns nothing each time.
COUNT_METRIC_CHOICES: List[str] = [
    m for m in snapshot.SCOREABLE_METRICS if m not in scorecard.NON_COUNT_METRICS
] + ["none"]

# Quantities spelled out in words, for Reflection.note_to_self. "one" and "a"
# are absent on purpose: "one member asked a question" and "a third of the
# server" differ, and refusing the first would reject most honest sentences.
_SPELLED_NUMBER = (
    r"\b(?:two|three|four|five|six|seven|eight|nine|ten|eleven|twelve|"
    r"twenty|thirty|forty|fifty|sixty|seventy|eighty|ninety|hundred|thousand|"
    r"dozen|couple|half|third|quarter|percent|per cent)\b"
)


class Verdict(BaseModel):
    reasoning: str = Field(
        description="Your working. Start by restating the founder's message as a "
        "neutral yes/no question, then answer that."
    )
    evidence: str = Field(
        description="At most three sentences, citing a number, name, or message "
        "from the snapshot. This is what the founder reads."
    )
    would_change_my_mind: str = Field(
        description="The specific fact or number that would flip this verdict."
    )
    # Literal becomes a JSON Schema enum, so an invalid verdict is rejected by
    # the decoder rather than merely discouraged by the description.
    verdict: Literal["yes", "no", "not_yet"]
    instead: Optional[str] = Field(
        default=None, description="Concrete alternative action. Required unless verdict is yes."
    )
    confidence: Literal["low", "medium", "high"]
    confidence_pct: int = Field(
        ge=0, le=100, description="The same call as a percentage, 0-100."
    )

    # Set after generation, never by the model, and excluded from the schema.
    _unverified: List[str] = PrivateAttr(default_factory=list)


class Recommendation(BaseModel):
    evidence: str = Field(
        description="The number, name, or message from the snapshot that justifies this."
    )
    reasoning: str = Field(description="Why this is the highest-value move right now.")
    would_change_my_mind: str = Field(
        description="What you would have to see in the snapshot to withdraw this."
    )
    play_fails_when: str = Field(
        description="The condition under which this play backfires."
    )
    play: str = Field(
        default="none",
        description="Which play from the catalogue this is. The list you are "
        "given is computed from the snapshot: a play whose precondition does "
        "not hold, or which the founder has already done, is not in it. Use "
        "'none' when nothing fits.",
    )
    headline: str = Field(description="The imperative, one line.")
    action: str = Field(
        description="Exactly what to do, specific enough to do today without thinking."
    )
    metric: str = Field(
        description="The one snapshot metric that should move if this works. Use "
        "'none' if no snapshot metric would move — that is an honest answer and "
        "is graded as unmeasurable, not as a failure.",
        json_schema_extra={"enum": METRIC_CHOICES},
    )
    direction: Literal["up", "down", "unchanged"]
    horizon_days: int = Field(
        ge=7, le=90, description="How long before this can fairly be judged."
    )
    guardrail_metric: str = Field(
        description="The metric that must not get worse while this is running. "
        "'none' if nothing is at risk.",
        json_schema_extra={"enum": METRIC_CHOICES},
    )

    @field_validator("metric", "guardrail_metric")
    @classmethod
    def _known_metric(cls, value: str) -> str:
        # The enum above is enforced by Ollama's grammar. The Anthropic path
        # needs this, and both paths need it if a schema ever loses the enum.
        if value not in METRIC_CHOICES:
            raise ValueError(
                "%r is not a snapshot metric. Choose one of: %s"
                % (value, ", ".join(METRIC_CHOICES))
            )
        return value

    @model_validator(mode="after")
    def _bet_must_be_won_by_improving(self) -> "Recommendation":
        """Refuse a metric and direction whose success is the server getting worse.

        scorecard.POLARITY is which way each metric is better, and until now only
        the guardrail consulted it. Nothing stopped a recommendation naming
        members.gone_quiet.count with direction "up": it reads like a confident,
        measurable, pre-registered bet, and the scorer graded it `worked` when
        eighteen more members went silent. Rejected here, at the point the schema
        is validated, so the bet cannot be stored at all — scorecard.py refuses
        it again at grading time, for rows written before this existed.
        """
        polarity = scorecard.POLARITY.get(self.metric)
        if polarity is None or self.direction == "unchanged":
            return self
        if (self.direction == "up") != (polarity > 0):
            raise ValueError(
                "%s going %s is the direction that makes the server worse, so it "
                "cannot be what this recommendation is for. Name the metric that "
                "should improve, or use 'none'." % (self.metric, self.direction)
            )
        return self


class Brief(BaseModel):
    # Field order is generation order, so the headline is last: it is the
    # conclusion, and a conclusion written before the recommendations exist is
    # written before there is any reasoning to support it. render_brief puts it
    # back on top for the reader.
    #
    # Zero recommendations is allowed. playbooks/seed.md calls doing nothing to
    # the server "a legitimate and frequently correct recommendation", and a
    # schema with a floor of one manufactures something to do on every run.
    recommendations: List[Recommendation] = Field(default_factory=list, max_length=3)
    no_action_reason: Optional[str] = Field(
        default=None,
        description="Required when there are no recommendations. Must cite a "
        "number from the snapshot.",
    )
    dont: Optional[str] = None
    headline: str

    # Set after generation, never by the model, and excluded from the schema.
    _unverified: List[str] = PrivateAttr(default_factory=list)
    _verdicts: List[Dict[str, Any]] = PrivateAttr(default_factory=list)
    _open: Optional[Dict[str, Any]] = PrivateAttr(default=None)
    # Which backend actually wrote this, which is not always the default one.
    _backend: Optional[str] = PrivateAttr(default=None)

    @model_validator(mode="after")
    def _abstention_must_be_argued(self) -> "Brief":
        """Doing nothing is allowed; refusing to say why is not.

        The opposite failure to padding a brief is abstention as cowardice, so
        an empty list has to be paid for with a reason that cites a number.
        """
        if self.recommendations:
            return self
        reason = (self.no_action_reason or "").strip()
        if not reason:
            raise ValueError(
                "no_action_reason is required when there are no recommendations"
            )
        if not re.search(r"\d", reason):
            raise ValueError(
                "no_action_reason must cite a number from the snapshot"
            )
        return self


class Reflection(BaseModel):
    """One thought, prompted by something that happened rather than by a person.

    Same field-order law as everything else here: working first, conclusions
    last. `to_founder` is the very last thing generated, after the model has
    already committed to whether there is anything worth saying.
    """

    restated: str = Field(
        description="The thing that happened, restated as a neutral question in "
        "your own words. Nobody asked you this; say what you are actually asking."
    )
    reasoning: str = Field(description="Your working. Nobody reads this but you.")
    evidence: str = Field(
        description="The number from the snapshot, or from the facts above, that "
        "this rests on. If there isn't one, say that instead of finding one."
    )
    note_to_self: str = Field(
        description="At most two sentences, carried into a later brief and read "
        "cold, by you, with no memory of today. No numbers."
    )
    watch_metric: str = Field(
        description="The event count whose movement would tell you whether you "
        "were right, or 'none'.",
        json_schema_extra={"enum": COUNT_METRIC_CHOICES},
    )
    worth_telling_founder: bool = Field(
        description="A veto, not a send button. False is the normal answer."
    )
    to_founder: Optional[str] = Field(
        default=None,
        description="At most three sentences, plain. Only if worth_telling_founder.",
    )

    # Set after generation, never by the model, and excluded from the schema.
    _unverified: List[str] = PrivateAttr(default_factory=list)

    @field_validator("watch_metric")
    def _known_metric(cls, value: str) -> str:
        """The enum above is enforced by Ollama's grammar. Anthropic needs this."""
        if value not in COUNT_METRIC_CHOICES:
            raise ValueError(
                "%r is not an event count. Choose one of: %s"
                % (value, ", ".join(COUNT_METRIC_CHOICES))
            )
        return value

    @field_validator("note_to_self")
    def _no_quantities(cls, value: str) -> str:
        """A note is prompt input for the next two months. Quantities rot.

        verify_evidence would catch a number that was invented today, but the
        likelier failure is one that was *true* today and false in October —
        "activity fell to 4 a week" passes every other check on the day it is
        written and is a lie by the time it is read back. Nothing quantitative
        needs to be carried, because the snapshot is right there when the note
        is read.

        Digits are the easy half. A digit filter alone lets through "activity
        fell to four a week and joins to two", which is the identical failure
        spelled out, so the cardinals and the common fraction words are refused
        too. This is a word list and word lists are never complete — "a handful"
        gets through — but it moves the leak from *the obvious phrasing* to an
        unusual one, and what does get through is bounded by two sentences and
        sixty days.
        """
        # Dropped rather than refused. Raising here fails the whole Reflection,
        # and on CPU inference that is a four-minute generation thrown away
        # because the least important field — a private note nobody reads —
        # contained "0 messages". The guarantee is about what gets *stored*, and
        # an empty note stores nothing: agenda.live_notes already filters
        # `note_to_self <> ''`, so a discarded note is simply not carried.
        if re.search(r"\d", value) or re.search(_SPELLED_NUMBER, value, re.IGNORECASE):
            return ""
        return value


# --- prompt assembly -------------------------------------------------------

# One-step reframing. UK AISI (arXiv:2602.23971) measured sycophancy beta at
# 1.13 unmitigated, 0.51 for a system-prompt instruction not to be sycophantic,
# and 0.16 for making the model restate the question neutrally first — and the
# effect grew with how certain the founder sounded, which is exactly how a
# founder types. prompts.py already has the weak arm; this is the strong one.
# The two-step variant scores better still and costs a second model call, which
# is 40-180s on ollama for one verdict, so it is not used.
REFRAME = (
    "Restate the founder's message as a neutral yes/no question in your "
    "`reasoning` field before answering it. Answer the restated question, not "
    "the original phrasing."
)

# The binding detail behind the four commitment fields BRIEF_INSTRUCTION names
# but does not spell out. It lives here rather than in prompts.py because every
# rule in it is a property of the schema and of scorecard.py, so the two have to
# be edited together or not at all.
BRIEF_SCHEMA_NOTE = """\
# How a recommendation commits itself

Each recommendation names the one snapshot metric that should move, which way,
and by when. Those are a promise made in advance, not a description written
afterwards, and the threshold is computed from the snapshot before you see any
result — you cannot choose it and you cannot revise it.

- `metric` and `guardrail_metric` must be exact dotted paths from the list in
  the schema. If nothing in the snapshot would move, use "none": that is graded
  as unmeasurable, which is not a failure. Do not reach for a nearby metric.
- `direction` is what the metric should do, and it has to be the direction that
  makes the server *better*. A recommendation whose success would mean more
  members going quiet, or more people leaving, is rejected outright: name the
  metric that should improve instead. "unchanged" is a claim that the metric
  holds, and is graded as one.
- `horizon_days` is how long before that is a fair question. A metric with a
  window in its name cannot be judged inside that window — two readings of a
  30-day count taken a week apart are mostly the same messages — so a horizon
  shorter than the metric's own window is raised to it before anything is
  stored.
- `guardrail_metric` is what must not get worse meanwhile. "none" is allowed.

Return zero recommendations when nothing is worth doing. If you do,
`no_action_reason` is required and must cite a number from the snapshot.

# Past recommendations

Any verdict below was computed from the numbers by code you did not run and
cannot argue with. Do not re-grade one, do not add your own assessment of
whether earlier advice worked, and do not restate a stored action in your own
words — it is quoted verbatim for a reason. If you refer to one, name it by its
reference, as in "R-14", never as your own advice.
"""


def _turn(
    instruction: str,
    snap: Dict[str, Any],
    question: Optional[str],
    extra: Optional[str] = None,
) -> str:
    parts = [
        instruction,
        "# Server snapshot\n\n```json\n%s\n```" % json.dumps(snap, indent=2, default=str),
    ]
    if extra:
        parts.append(extra)
    if question:
        parts.append("# The founder's question\n\n%s\n\n%s" % (REFRAME, question))
    return "\n\n".join(parts)


def _given_verdicts(
    verdicts: List[Dict[str, Any]], open_row: Optional[Dict[str, Any]]
) -> Optional[str]:
    """The scorecard, as immovable input. Capped, because it grows with time."""
    if not verdicts and not open_row:
        return None
    block: Dict[str, Any] = {"closed": scorecard.capped(verdicts)}
    if open_row:
        block["still_open"] = {
            "ref": open_row["ref"],
            "action_text": open_row["action"],
            "prediction": open_row["prediction"],
            "days_left": open_row["days_left"],
            "note": "no verdict yet — do not guess at one, and do not open a "
            "second recommendation while this is running",
        }
    return "# Verdicts (computed, not yours)\n\n```json\n%s\n```" % json.dumps(
        block, indent=2, default=str
    )


# --- the self-grading guard ------------------------------------------------

# The same shape as notify._guard: a structural refusal rather than a rule the
# model is asked to follow. The schema carries no field for assessing past
# advice; this catches the assessment arriving inside a free-text field anyway.
# Narrow on purpose. It has to catch "my last recommendation worked" without
# eating "the recommendation is to DM u4", which is a new recommendation
# describing itself.
_SELF_ASSESSMENT = re.compile(
    r"\b(my|our|your advisor'?s)\s+(last|previous|earlier|prior)?\s*"
    r"(recommendation|advice|suggestion)s?\b"
    r"|\bthe (last|previous|earlier|prior) (recommendation|advice|suggestion)s?\b"
    r"|\bas (i|we) (recommended|suggested|advised|predicted)\b"
    r"|\b(i|we) (recommended|suggested|advised|predicted|told you)\b"
    r"|\b(advice|recommendation)s? (i|we) gave\b"
    # A reflection is invited to talk about its own past output, so it reaches
    # for these two shapes in a way a brief never did.
    r"|\b(my|our|the) (earlier|previous|prior|last) "
    r"(note|question|reflection|reading|thought)s?\b"
    r"|\b(i|we) (was|were) (right|wrong) (about|to)\b",
    re.IGNORECASE,
)

_SENTENCE = re.compile(r"[^.!?]+[.!?]?")


def _drop_self_assessment(text: Optional[str]) -> Optional[str]:
    """Remove any sentence in which the model grades its own past advice."""
    if not text:
        return text
    kept = [s for s in _SENTENCE.findall(text) if not _SELF_ASSESSMENT.search(s)]
    cleaned = "".join(kept).strip()
    return cleaned or None


def _guard_self_grading(b: Brief) -> None:
    b.headline = _drop_self_assessment(b.headline) or b.headline
    b.dont = _drop_self_assessment(b.dont)
    # no_action_reason is load-bearing for validity, so it is only cleaned when
    # something survives the cut.
    cleaned = _drop_self_assessment(b.no_action_reason)
    if cleaned:
        b.no_action_reason = cleaned
    for rec in b.recommendations:
        rec.reasoning = _drop_self_assessment(rec.reasoning) or rec.reasoning
        rec.evidence = _drop_self_assessment(rec.evidence) or rec.evidence


# --- the evidence verifier -------------------------------------------------

_NUMERAL = re.compile(r"-?\d+(?:\.\d+)?")
_TIMESTAMP = re.compile(r"^\d{4}-\d{2}-\d{2}[T ]")


def _numeric_literals(snap: Dict[str, Any]) -> Set[str]:
    """Every number that appears as a value in the snapshot, as text.

    Values and strings only — never keys. "messages_30d" is a field name, and
    counting the 30 in it as a known number would license "engagement is down
    30%" forever, along with every other number that happens to appear in a
    window label. Member and channel names are mined, because "chan-12" and
    "u4" are numbers a founder would recognise.
    """
    found: Set[str] = set()

    def add(token: str) -> None:
        found.add(_normalise(token))

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for value in node.values():
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                walk(value)
        elif isinstance(node, bool):
            return
        elif isinstance(node, int):
            add(str(node))
        elif isinstance(node, float):
            add(repr(node))
            add(str(int(node)))
            for places in (0, 1, 2):
                add(("%." + str(places) + "f") % node)
        elif isinstance(node, str):
            # A timestamp is six arbitrary two-digit numbers. Mining them would
            # make most small integers "known" and the check would pass
            # everything.
            if _TIMESTAMP.match(node):
                return
            for token in _NUMERAL.findall(node):
                add(token)

    walk(snap)
    return found


# How a model refers to a channel: "#general", "the 'general' channel", or
# "in the general channel". Members get quoted or @-prefixed.
_CHANNEL_REF = re.compile(
    r"#([A-Za-z0-9_-]{2,})"
    r"|['\u2018\u2019\"\u201c\u201d]([A-Za-z0-9_ -]{2,32})['\u2018\u2019\"\u201c\u201d]\s+channel"
    r"|\bthe\s+([A-Za-z0-9_-]{2,32})\s+channel\b"
)


def _known_names(snap: Dict[str, Any]) -> Set[str]:
    """Every channel and person the snapshot actually names.

    Collected from the whole tree rather than from a fixed list of keys, because
    names appear in channels, dead_channels, top_posters_30d, never_posted,
    gone_quiet, unanswered_questions and reply_dyads_30d, and a new block would
    otherwise silently start producing false positives.
    """
    found: Set[str] = set()

    def walk(node: Any) -> None:
        if isinstance(node, dict):
            for key, value in node.items():
                if key in ("channel", "author", "name", "member", "display_name") and isinstance(value, str):
                    found.add(value.strip().lower())
                walk(value)
        elif isinstance(node, (list, tuple)):
            for value in node:
                if isinstance(value, str):
                    found.add(value.strip().lower())
                else:
                    walk(value)

    walk(snap)
    return found


def verify_entities(snap: Dict[str, Any], text: Optional[str]) -> List[str]:
    """Channel names cited in `text` that the snapshot does not contain.

    verify_evidence catches an invented number and nothing else, so "post in the
    #introductions channel" — on a server whose only channel is #hermes — passed
    every check cadybot had. A small model is more prone to this than to
    inventing a statistic, because a plausible channel name is exactly the kind
    of detail it will fill in from the shape of the sentence.

    Reported the same way numbers are: advisory everywhere except an unprompted
    message, which is held to a higher bar because nobody asked for it.
    """
    if not text:
        return []
    known = _known_names(snap)
    missing: List[str] = []
    for match in _CHANNEL_REF.finditer(text):
        name = next((g for g in match.groups() if g), "").strip().lower()
        if not name or name in known or name in missing:
            continue
        missing.append(name)
    return missing


def _normalise(token: str) -> str:
    token = token.lstrip("+")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token or "0"


# What counts as a number the model *cited*, as opposed to a digit that happens
# to sit inside a word. `_NUMERAL` is deliberately greedy when mining the
# snapshot — more known tokens can only reduce false positives — but running the
# same pattern over prose reads "3" out of "r/3Dprinting", "4" out of "Q4" and
# "-1" out of "R-1".
#
# That is not hypothetical. On the first real local run the desk produced sound
# advice — go participate in r/3Dprinting rather than over-read one message —
# and suppressed it, because "3" appears in no snapshot. The single most likely
# phrase this founder's advisor could ever use silenced it permanently.
_CITED_NUMERAL = re.compile(r"(?<![A-Za-z0-9])-?\d+(?:\.\d+)?(?![A-Za-z])")

# Dates are not statistics. `_numeric_literals` already refuses to mine a
# timestamp on the snapshot side, for the stated reason that six arbitrary
# two-digit numbers would make most small integers "known" — but the text side
# had no matching rule, so a reflection that correctly said "nobody answered
# them on 2026-05-09" was flagged for inventing 2026, 05 and 09, and suppressed.
# Citing when something happened is the opposite of making a number up.
_DATEISH = re.compile(
    r"\d{4}-\d{2}-\d{2}(?:[T ]\d{2}:\d{2}(?::\d{2})?)?"
    r"|\b\d{1,2}/\d{1,2}(?:/\d{2,4})?\b"
    r"|\b\d{1,2}:\d{2}\b"
)


def verify_cited(known: Set[str], text: Optional[str]) -> List[str]:
    """Numbers cited in `text` that are not in `known`.

    Split out of verify_evidence so a caller can widen the universe with figures
    that came from a lookup rather than from the snapshot. The rule is the same
    either way: a figure is citable when code computed it, and not otherwise.
    """
    if not text:
        return []
    missing: List[str] = []
    for token in _CITED_NUMERAL.findall(_DATEISH.sub(" ", text)):
        normalised = _normalise(token)
        if normalised not in known and normalised not in missing:
            missing.append(normalised)
    return missing


def verify_evidence(snap: Dict[str, Any], text: str) -> List[str]:
    """Numbers cited in `text` that appear nowhere in the snapshot.

    "engagement is down about 30%" passes every other check cadybot has, and 30
    exists nowhere. This is reported, never acted on: values derived by
    arithmetic and prose like "a third" will trip it, and the false-positive
    rate has not been measured across the scenarios yet. Auto-retracting on an
    unmeasured detector would be worse than the failure it catches.
    """
    return verify_cited(_numeric_literals(snap), text)


# --- entry points ----------------------------------------------------------


def ask(
    question: str,
    snap: Dict[str, Any],
    guild_id: Optional[int] = None,
    backend: Optional[str] = None,
    inq: Any = None,
) -> Verdict:
    result = llm.generate(
        prompts.stable_prefix(),
        _turn(prompts.ASK_INSTRUCTION, snap, question,
              getattr(inq, "digest", None) or None),
        Verdict,
        "ask",
        guild_id or config.GUILD_ID,
        backend=backend,
    )
    # Brief has had this since the beginning; ask did not, and a verdict is the
    # answer the founder is most likely to act on immediately. Observed on the
    # live server: the snapshot said one human and four bots, and the model
    # wrote "the server has 7 members" -- a number it read out of an
    # illustrative example in the prompt rather than out of the data.
    # would_change_my_mind is rendered to the founder and reasoning is not, so
    # the checked set follows what is shown rather than what is generated.
    # Figures a lookup computed are citable; figures inside a quoted message are
    # somebody's words and stay out of the set.
    known = _numeric_literals(snap)
    for block in (getattr(inq, "facts", None) or []):
        known |= _numeric_literals(block)
    result._unverified = sorted(
        set(
            token
            for text in (result.evidence, result.instead, result.would_change_my_mind)
            if text
            for token in verify_cited(known, text)
        )
    )
    return result


def brief(
    snap: Dict[str, Any],
    guild_id: Optional[int] = None,
    verdicts: Optional[List[Dict[str, Any]]] = None,
    backend: Optional[str] = None,
    register_now: bool = True,
) -> Brief:
    """One brief. Grading has already happened by the time this runs.

    `verdicts` comes from scorecard.score, which loop.py commits before calling
    here. When it is None — the slash-command path — already-closed verdicts are
    read back instead, so the report is the same shape either way.

    `register_now=False` returns the brief without opening the tracked bet, for
    callers that can tell whether the founder actually received it. See
    `register` below.
    """
    guild_id = guild_id or config.GUILD_ID
    open_row = scorecard.open_row(guild_id) if guild_id else None
    if verdicts is None and guild_id:
        verdicts = scorecard.recent_verdicts(guild_id, limit=4)
    verdicts = verdicts or []

    # The verdict block and the desk's own notes ride together in `extra`. Notes
    # go last so the block the founder's outcome is judged against is the one
    # nearest the instruction.
    extra = "\n\n".join(
        p for p in (
            plays.render(snap),
            _given_verdicts(verdicts, open_row),
            _notes_block(guild_id),
        ) if p
    ) or None

    result = llm.generate(
        prompts.stable_prefix(),
        _turn(
            prompts.BRIEF_INSTRUCTION + "\n\n" + BRIEF_SCHEMA_NOTE,
            snap,
            None,
            extra,
        ),
        _brief_model_for(snap),
        "brief",
        guild_id,
        backend=backend,
    )

    _guard_self_grading(result)
    result._backend = backend
    result._verdicts = verdicts
    result._open = open_row
    # scorecard.pre_register raises a horizon shorter than the metric's own
    # window, because two readings closer together than that are the same events
    # counted twice. Mirrored here so the "_Watch:_" line the founder reads is
    # the promise that will actually be stored.
    for rec in result.recommendations:
        rec.horizon_days = max(rec.horizon_days, scorecard.window_days(rec.metric))
    # no_action_reason is checked too: it is required to cite a number, and a
    # required number is exactly the kind that gets invented to satisfy a rule.
    cited = [rec.evidence for rec in result.recommendations]
    if result.no_action_reason:
        cited.append(result.no_action_reason)
    result._unverified = sorted(
        set(token for text in cited for token in verify_evidence(snap, text))
    )

    if register_now:
        register(result, snap, guild_id)
    return result


def register(result: Brief, snap: Dict[str, Any], guild_id: Optional[int] = None) -> List[int]:
    """Open the tracked bet. Called once the founder has the brief in hand.

    Split out of `brief` because pre-registration is a claim that advice was
    given. Run before delivery, a channel that has been deleted or a permission
    that has been revoked leaves a fortnight-long bet open on a recommendation
    nobody read, which then grades against a founder who was never told what to
    do.
    """
    guild_id = guild_id or config.GUILD_ID
    if not guild_id:
        return []

    # Only the top-ranked recommendation becomes a tracked bet. The founder may
    # usefully hear three things; three simultaneous pre-registrations over a
    # fortnight and seven members cannot be told apart, and a grader handed
    # three overlapping claims on one delta will find a way to credit all of
    # them. The rest are advice, and are rendered as advice.
    ids: List[int] = []
    if result.recommendations and result._open is None:
        ids = scorecard.pre_register(
            guild_id,
            snap,
            [result.recommendations[0].model_dump()],
            llm.describe(result._backend),
        )
    # Only rows this pass closed. Everything that renders a brief reads closed
    # verdicts back for context, and narrating those again relabels a row months
    # after the sentence it names was written.
    narrated = [v["ref"] for v in result._verdicts if v.get("newly_closed")]
    if narrated:
        scorecard.record_narration(narrated, llm.describe(result._backend))
    return ids


def chat(
    guild_id: int, channel_id: int, message: str, speaker: str, snap: Dict[str, Any],
    inq: Any = None,
) -> str:
    """One conversational turn, with the running history and a fresh snapshot.

    The snapshot rides on the newest message rather than the system prompt so
    the cached prefix stays byte-identical between turns.

    Deliberately not gated by an open recommendation: the lockout is on opening
    a second pre-registered bet, never on answering a question.
    """
    history = db.recent_turns(guild_id, channel_id)
    parts = [
        prompts.CHAT_INSTRUCTION,
        "```json\n%s\n```" % json.dumps(snap, indent=2, default=str),
    ]
    if inq is not None and getattr(inq, "digest", ""):
        parts.append(inq.digest)
    parts.append("%s: %s" % (speaker, message))
    reply = llm.converse(
        prompts.stable_prefix(),
        history + [{"role": "user", "content": "\n\n".join(parts)}],
        guild_id,
    )

    # Chat has never had any verification at all. It gets it now, because a
    # lookup makes provenance answerable: a figure is citable when the snapshot
    # or a lookup computed it. Quotes are excluded on purpose — member text and
    # stored advice contain figures like "spend 90 minutes" that are not facts
    # about the server. Labelled rather than suppressed: this is a conversation
    # the founder started, not something volunteered at him.
    known = _numeric_literals(snap)
    for block in (getattr(inq, "facts", None) or []):
        known |= _numeric_literals(block)
    unchecked = verify_cited(known, reply)
    if unchecked:
        reply += "\n\n_not from your data: %s_" % ", ".join(unchecked)
    if inq is not None and getattr(inq, "footer", ""):
        reply += "\n" + inq.footer

    db.add_turn(guild_id, channel_id, "user", message, speaker)
    db.add_turn(guild_id, channel_id, "assistant", reply)
    return reply


def _brief_model_for(snap: Dict[str, Any]):
    """A Brief whose `play` field only accepts plays that are actually eligible.

    The enum is computed from the snapshot by plays.py and injected per request,
    which is the same mechanism that already makes an invented metric name
    undecodable: ollama compiles the JSON Schema to a grammar, so an ineligible
    play cannot be emitted at all rather than merely being discouraged in prose.

    This exists because prose preconditions do not hold. The catalogue says
    "Merge channels down — if more than about three channels exist", the live
    server has one channel already named hermes, and the model recommended
    merging down to one channel named hermes. Eligible, satisfied and stage are
    arithmetic now.
    """
    allowed = plays.choices(snap)

    @field_validator("play")
    def _known_play(cls, value: str) -> str:
        # The enum is enforced by ollama's grammar. The Anthropic path needs this.
        if value not in allowed:
            raise ValueError(
                "%r is not available for this server right now. Choose one of: %s"
                % (value, ", ".join(allowed))
            )
        return value

    rec = create_model(
        "EligibleRecommendation",
        __base__=Recommendation,
        __validators__={"_known_play": _known_play},
        # Required, with no default. A defaulted field is one the decoder may
        # skip, and it did: the first run returned play="none" on a
        # recommendation whose own headline was "Show a failure: ...". Naming
        # the play is how the recommendation becomes checkable at all.
        play=(
            str,
            Field(
                description=Recommendation.model_fields["play"].description,
                json_schema_extra={"enum": allowed},
            ),
        ),
    )
    return create_model(
        "EligibleBrief",
        __base__=Brief,
        recommendations=(List[rec], Field(default_factory=list, max_length=3)),
    )


def _notes_block(guild_id: Optional[int]) -> Optional[str]:
    """What cadybot noticed between reports, handed to the next brief.

    This is the only channel by which thinking done unprompted reaches anything
    the founder sees, and it is the one place the "the model never sees what it
    produced" property is softened. The header is doing real work: these are
    labelled as the model's own notes, not as facts, and the schema forbids them
    from containing a number, so the worst a stale note can do is be wrong in
    prose next to a snapshot that is right in figures.
    """
    if not guild_id:
        return None
    notes = agenda.live_notes(guild_id)
    if not notes:
        return None
    return "\n".join(
        [
            "# Your own notes since the last brief",
            "",
            "You wrote these to yourself when something happened. They are not "
            "facts and they are not numbers — the snapshot above is the only "
            "thing that counts. No verdict on past advice is yours to give.",
            "",
        ]
        + ["- %s" % n for n in notes]
    )


def reflect(
    prov: Any,
    snap: Dict[str, Any],
    guild_id: Optional[int] = None,
    backend: Optional[str] = None,
) -> Reflection:
    """Answer a question cadybot put to itself.

    `prov` is an agenda.Provocation. Its `self_prompt` was composed by code from
    stored rows — no model-authored text is ever stored as a question, because a
    question is durable prompt input and a number invented inside one becomes a
    fact cadybot believes for as long as it survives.

    Note what is *not* here: no verdict field, no way to re-grade a past row, and
    no argument selecting a different backend to narrate with. All three were
    available and all three are the shapes that went wrong before.
    """
    result = llm.generate(
        prompts.stable_prefix(),
        _turn(prompts.REFLECT_INSTRUCTION, snap, None, prov.self_prompt),
        Reflection,
        "reflect",
        guild_id or config.GUILD_ID,
        backend=backend,
    )
    # Only the two fields that survive this call get cleaned. `reasoning` is
    # neither shown to anyone nor replayed, so rewriting it would be editing a
    # private note for an audience that does not exist.
    result.to_founder = _drop_self_assessment(result.to_founder)
    # No `or result.note_to_self` fallback here. _drop_self_assessment returns
    # None when every sentence was self-assessment, and elsewhere the original
    # is restored because the field is load-bearing. This field is not: the
    # verdict provocation asks the model what it got wrong, so a note that is
    # *entirely* self-grading is the expected answer — and restoring it would
    # feed exactly the thing the guard exists to strip into the next sixty days
    # of briefs. An empty note is the correct outcome.
    result.note_to_self = _drop_self_assessment(result.note_to_self) or ""
    # Figures the provocation itself supplied are not inventions: cadybot handed
    # the model "threshold 1, reading 0" and must not then flag it for saying so.
    # Numbers and names, together. A small model invents a plausible channel far
    # more readily than it invents a statistic — "post in #introductions" on a
    # server whose only channel is #hermes — and until verify_entities existed
    # nothing looked at that at all.
    result._unverified = sorted(
        (set(verify_evidence(snap, result.to_founder)) - agenda.known_numbers(prov))
        | set("#" + name for name in verify_entities(snap, result.to_founder))
    )
    return result


# --- rendering -------------------------------------------------------------

_MARK = {"yes": "YES", "no": "NO", "not_yet": "NOT YET"}

# What each verdict is allowed to be called in front of the founder. "worked"
# is written as movement rather than as success everywhere it is shown: the
# scorer records that a metric moved, not that the advice caused it.
_VERDICT_LABEL = {
    "worked": "moved as predicted",
    "failed": "did not move",
    "harmful": "guardrail broke",
    "revoked": "moved, then fell back",
    "inconclusive": "no separable change",
    "not_attempted": "no sign it was done",
    "unmeasurable": "not measurable",
}


def render_verdict(v: Verdict) -> str:
    """Conclusion first for the reader, whatever order it was generated in."""
    lines = ["**%s**" % _MARK.get(v.verdict, v.verdict.upper()), "", v.evidence]
    if v.instead:
        lines += ["", "**Instead:** %s" % v.instead]
    if v.would_change_my_mind:
        lines += ["", "_Would change this:_ %s" % v.would_change_my_mind]
    if v.confidence == "low":
        lines += ["", "_Low confidence (%d%%) — not enough data yet to be sure._" % v.confidence_pct]
    else:
        lines += ["", "_Confidence: %s (%d%%)._" % (v.confidence, v.confidence_pct)]
    if v._unverified:
        lines += [
            "",
            "_These numbers appear nowhere in the snapshot: %s. Check them before "
            "acting._" % ", ".join(v._unverified),
        ]
    lines += ["", "_%s_" % llm.describe()]
    return "\n".join(lines)


def fmt_number(value: Optional[float]) -> str:
    if value is None:
        return "n/a"
    if float(value).is_integer():
        return "%d" % int(value)
    return "%.3g" % value


def render_scorecard(verdicts: List[Dict[str, Any]]) -> List[str]:
    """The closed-verdict block. Numbers only — no framing, no narrative."""
    lines: List[str] = ["**Scorecard**", ""]
    for v in verdicts:
        # Authorship is stripped on purpose: "R-14: <action>", never "I said".
        lines.append("**%s: %s**" % (v["ref"], v["action_text"]))
        if v["metric"] and v["metric"] != "none":
            lines.append(
                "%s — baseline %s, now %s, delta %s%s"
                % (
                    v["metric"],
                    fmt_number(v["baseline"]),
                    fmt_number(v["current"]),
                    fmt_number(v["delta"]),
                    ", p=%.3f" % v["p_value"] if v["p_value"] is not None else "",
                )
            )
        lines.append(
            "verdict: **%s** (%s)"
            % (v["verdict"], _VERDICT_LABEL.get(v["verdict"], v["verdict"]))
        )
        if v.get("revoked_at") and not v.get("note"):
            # Only when nothing else says it. The pass that revokes a row hands
            # over a note carrying the day it was re-checked, which is strictly
            # more than this line; rows read back later have no note at all.
            lines.append("_revoked on re-check: the metric did not hold._")
        if v.get("note"):
            lines.append("_%s_" % v["note"])
        lines.append("")
    return lines


def render_brief(b: Brief) -> str:
    """Scorecard first, then advice.

    Order is not cosmetic. A verdict block written after the new advice gets
    written into whatever frame the advice established; generated first, from
    code, it is the thing the new advice has to be consistent with.
    """
    lines: List[str] = []
    if b._verdicts:
        lines += render_scorecard(b._verdicts)

    if b._open:
        lines += [
            "**%s is still open** (day %.0f of %d): %s"
            % (
                b._open["ref"],
                b._open["age_days"],
                b._open["horizon_days"] or config.RECOMMENDATION_HORIZON_DAYS,
                b._open["action"],
            ),
            "No verdict yet, and no new recommendation until there is one.",
            "",
            b.headline,
            "",
        ]
        if b.dont:
            lines += ["**Don't:** %s" % b.dont, ""]
        lines += ["_%s_" % llm.describe(b._backend)]
        return "\n".join(lines).strip()

    lines += [b.headline, ""]
    for i, r in enumerate(b.recommendations, 1):
        lines += ["**%d. %s**" % (i, r.headline), r.action, "_Why:_ %s" % r.evidence]
        if i == 1 and r.metric != "none":
            # Only the first is pre-registered, so only the first gets a
            # promise. Printing a watch line under the others would advertise
            # tracking that is not happening.
            lines.append(
                "_Watch:_ %s %s within %d days (guardrail: %s)"
                % (r.metric, r.direction, r.horizon_days, r.guardrail_metric)
            )
        elif i == 1:
            lines.append("_Watch:_ nothing in the snapshot moves if this works.")
        else:
            lines.append("_Not tracked: only the first recommendation is scored._")
        lines += ["_Fails when:_ %s" % r.play_fails_when, ""]
    if not b.recommendations and b.no_action_reason:
        lines += ["**Nothing to do this week.** %s" % b.no_action_reason, ""]
    if b.dont:
        lines += ["**Don't:** %s" % b.dont, ""]
    if b._unverified:
        lines += [
            "_Unverified numbers: %s — cited above but not found in the snapshot. "
            "Check before acting on them._" % ", ".join(b._unverified),
            "",
        ]
    lines += ["_%s_" % llm.describe(b._backend)]
    return "\n".join(lines).strip()


def render_reflection(r: Reflection, prov: Any) -> str:
    """A volunteered thought. Deliberately unlike a brief.

    No headline, no ranking, no watch line, no pre-registered bet — this is
    cadybot saying one thing it noticed, and it has to look like that rather
    than like a report the founder forgot he scheduled. The provenance line says
    what set it off, so an unprompted message never arrives without a reason
    attached.
    """
    lines = ["**A thought.**", ""]
    # The finding first, and written by code. A model that has to *notice* the
    # fact buries it about two runs in three; one that only has to write around
    # a stated fact cannot lose it.
    found = getattr(prov, "finding", None)
    if found:
        lines += [found, ""]
    lines += [(r.to_founder or "").strip(), ""]
    because = {
        "verdict": "a recommendation closed",
        "life": "somebody posted after a silent month",
        "joined": "somebody joined after a quiet fortnight",
        "drift": "a count moved over the last fortnight",
        "backlog": "reading this server's history for the first time",
    }.get(prov.kind, prov.kind)
    trail = "_Prompted by %s" % because
    if prov.about_ref:
        trail += " (%s)" % prov.about_ref
    lines.append(trail + ". Nobody asked me; `/quiet 7` stops this._")
    if r._unverified:
        lines.append(
            "_Unverified numbers: %s — not found in the snapshot._"
            % ", ".join(r._unverified)
        )
    return "\n".join(lines).strip()


def render_stored(row: Any) -> str:
    """Re-render a thought that was written earlier and held for a quiet moment.

    The desk composes a sentence when something happens, but it may only speak
    inside a narrow window, so the two are usually different ticks. This puts
    the stored row back into the same shape render_reflection produces, without
    a second model call — the sentence was written and checked once.
    """
    stub = type("_Held", (), {"kind": row["kind"], "about_ref": row["about_ref"]})()
    held = Reflection(
        restated="", reasoning="", evidence="",
        note_to_self=row["note_to_self"] or "held",
        watch_metric=row["watch_metric"] or "none",
        worth_telling_founder=True,
        to_founder=row["to_founder"],
    )
    return render_reflection(held, stub)
