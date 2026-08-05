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

from pydantic import BaseModel, Field, PrivateAttr, field_validator, model_validator

from . import config, db, llm, prompts, scorecard, snapshot

# Re-exported so callers catch one name regardless of backend.
Refused = llm.Refused
BackendError = llm.BackendError

# The exact metric names a recommendation may commit to. Rendered into the
# schema as an enum, so an invented metric is rejected by the decoder rather
# than discovered at grading time, when the only remaining options are to guess
# or to drop the row.
METRIC_CHOICES: List[str] = list(snapshot.SCOREABLE_METRICS) + ["none"]


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


class Brief(BaseModel):
    headline: str
    # Zero is allowed. playbooks/seed.md calls doing nothing to the server "a
    # legitimate and frequently correct recommendation", and a schema with a
    # floor of one manufactures something to do on every run.
    recommendations: List[Recommendation] = Field(default_factory=list, max_length=3)
    dont: Optional[str] = None
    no_action_reason: Optional[str] = Field(
        default=None,
        description="Required when there are no recommendations. Must cite a "
        "number from the snapshot.",
    )

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

# Overrides the paragraph in BRIEF_INSTRUCTION that asks the model to assess its
# own past advice, and describes the fields that replaced `prediction`.
BRIEF_SCHEMA_NOTE = """\
# How a recommendation commits itself

Each recommendation names the one snapshot metric that should move, which way,
and by when. Those are a promise made in advance, not a description written
afterwards, and the threshold is computed from the snapshot before you see any
result — you cannot choose it and you cannot revise it.

- `metric` and `guardrail_metric` must be exact dotted paths from the list in
  the schema. If nothing in the snapshot would move, use "none": that is graded
  as unmeasurable, which is not a failure. Do not reach for a nearby metric.
- `direction` is what the metric should do. `horizon_days` is how long before
  that is a fair question.
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
    r"|\b(advice|recommendation)s? (i|we) gave\b",
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


def _normalise(token: str) -> str:
    token = token.lstrip("+")
    if "." in token:
        token = token.rstrip("0").rstrip(".")
    return token or "0"


def verify_evidence(snap: Dict[str, Any], text: str) -> List[str]:
    """Numbers cited in `text` that appear nowhere in the snapshot.

    "engagement is down about 30%" passes every other check cadybot has, and 30
    exists nowhere. This is reported, never acted on: values derived by
    arithmetic and prose like "a third" will trip it, and the false-positive
    rate has not been measured across the scenarios yet. Auto-retracting on an
    unmeasured detector would be worse than the failure it catches.
    """
    if not text:
        return []
    known = _numeric_literals(snap)
    missing: List[str] = []
    for token in _NUMERAL.findall(text):
        normalised = _normalise(token)
        if normalised not in known and normalised not in missing:
            missing.append(normalised)
    return missing


# --- entry points ----------------------------------------------------------


def ask(
    question: str,
    snap: Dict[str, Any],
    guild_id: Optional[int] = None,
    backend: Optional[str] = None,
) -> Verdict:
    return llm.generate(
        prompts.stable_prefix(),
        _turn(prompts.ASK_INSTRUCTION, snap, question),
        Verdict,
        "ask",
        guild_id or config.GUILD_ID,
        backend=backend,
    )


def brief(
    snap: Dict[str, Any],
    guild_id: Optional[int] = None,
    verdicts: Optional[List[Dict[str, Any]]] = None,
    backend: Optional[str] = None,
) -> Brief:
    """One brief. Grading has already happened by the time this runs.

    `verdicts` comes from scorecard.score, which loop.py commits before calling
    here. When it is None — the slash-command path — already-closed verdicts are
    read back instead, so the report is the same shape either way.
    """
    guild_id = guild_id or config.GUILD_ID
    open_row = scorecard.open_row(guild_id) if guild_id else None
    if verdicts is None and guild_id:
        verdicts = scorecard.recent_verdicts(guild_id, limit=4)
    verdicts = verdicts or []

    result = llm.generate(
        prompts.stable_prefix(),
        _turn(
            prompts.BRIEF_INSTRUCTION + "\n\n" + BRIEF_SCHEMA_NOTE,
            snap,
            None,
            _given_verdicts(verdicts, open_row),
        ),
        Brief,
        "brief",
        guild_id,
        backend=backend,
    )

    _guard_self_grading(result)
    result._backend = backend
    result._verdicts = verdicts
    result._open = open_row
    # no_action_reason is checked too: it is required to cite a number, and a
    # required number is exactly the kind that gets invented to satisfy a rule.
    cited = [rec.evidence for rec in result.recommendations]
    if result.no_action_reason:
        cited.append(result.no_action_reason)
    result._unverified = sorted(
        set(token for text in cited for token in verify_evidence(snap, text))
    )

    # Only the top-ranked recommendation becomes a tracked bet. The founder may
    # usefully hear three things; three simultaneous pre-registrations over a
    # fortnight and seven members cannot be told apart, and a grader handed
    # three overlapping claims on one delta will find a way to credit all of
    # them. The rest are advice, and are rendered as advice.
    if guild_id and result.recommendations and open_row is None:
        scorecard.pre_register(
            guild_id,
            snap,
            [result.recommendations[0].model_dump()],
            llm.describe(backend),
        )
    if guild_id and verdicts:
        scorecard.record_narration([v["ref"] for v in verdicts], llm.describe(backend))
    return result


def chat(
    guild_id: int, channel_id: int, message: str, speaker: str, snap: Dict[str, Any]
) -> str:
    """One conversational turn, with the running history and a fresh snapshot.

    The snapshot rides on the newest message rather than the system prompt so
    the cached prefix stays byte-identical between turns.

    Deliberately not gated by an open recommendation: the lockout is on opening
    a second pre-registered bet, never on answering a question.
    """
    history = db.recent_turns(guild_id, channel_id)
    latest = "%s\n\n```json\n%s\n```\n\n%s: %s" % (
        prompts.CHAT_INSTRUCTION,
        json.dumps(snap, indent=2, default=str),
        speaker,
        message,
    )
    reply = llm.converse(
        prompts.stable_prefix(), history + [{"role": "user", "content": latest}], guild_id
    )
    db.add_turn(guild_id, channel_id, "user", message, speaker)
    db.add_turn(guild_id, channel_id, "assistant", reply)
    return reply


# --- rendering -------------------------------------------------------------

_MARK = {"yes": "YES", "no": "NO", "not_yet": "NOT YET"}

# What each verdict is allowed to be called in front of the founder. "worked"
# is written as movement rather than as success everywhere it is shown: the
# scorer records that a metric moved, not that the advice caused it.
_VERDICT_LABEL = {
    "worked": "moved as predicted",
    "failed": "did not move",
    "harmful": "guardrail broke",
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
        if v.get("revoked_at"):
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
