"""The intervention catalogue, with preconditions that are code rather than prose.

`playbooks/*.md` describes every play and the condition that makes it the right
call. Those conditions were only ever read by a model, and a model walks through
them. Observed on the live server, verbatim:

    "immediately merge down all channels, leaving only one channel named
     `hermes`. This makes the small server look less abandoned."

The play it reached for is "Merge channels down", whose stated precondition is
*"If more than about three channels exist and most have no activity in 30 days."*
The server has exactly one channel, and it is already called hermes. So the
advice was ineligible, and the state it demanded already held. Two different
kinds of wrong in one sentence, and nothing in the codebase could see either.

This file makes both checkable:

- `eligible(snap)` returns the plays whose precondition actually holds. The
  result becomes a schema enum, so on the ollama path it is compiled into the
  grammar and an ineligible play is not merely discouraged, it is undecodable.
  Same mechanism that already stops an invented metric name.
- `satisfied` is the second half, and the sharper one. A play can be eligible in
  principle and pointless in fact, because the founder has already done it.
  Recommending a state that already holds is worse than saying nothing: it reads
  as confident and costs the founder the trust that the rest of the advice
  needs.

Where a precondition genuinely cannot be computed — "if the founder has
bandwidth for outbound" is not in any table — the play says so and stays
available. Faking a check would be worse than admitting there isn't one.

Model-free, and bound by the same ban as scorecard.py, ledger.py and agenda.py:
nothing here may import advisor or llm. What a server is allowed to be told is
exactly the judgement a model makes generously.
"""

import dataclasses
from typing import Any, Callable, Dict, List, Optional, Tuple

from . import snapshot

# Stage names as _stage() produces them, weakest first.
STAGES = ("seed", "sprout", "growing", "community")


def _m(snap: Dict[str, Any], path: str, default: Optional[float] = None) -> Optional[float]:
    value = snapshot.resolve_metric(snap, path)
    return default if value is None else value


def _count(snap: Dict[str, Any], *path: str) -> int:
    node: Any = snap
    for part in path:
        if not isinstance(node, dict):
            return 0
        node = node.get(part)
    if isinstance(node, dict):
        return int(node.get("count") or 0)
    if isinstance(node, list):
        return len(node)
    return int(node or 0)


@dataclasses.dataclass
class Play:
    """One intervention, and the arithmetic that decides whether to offer it."""

    id: str
    stages: Tuple[str, ...]
    title: str
    when: str                                   # the precondition, in words
    # True when the precondition holds. None means "not computable from the
    # snapshot" — the play stays available and says so rather than pretending.
    precondition: Optional[Callable[[Dict[str, Any]], bool]] = None
    # True when the founder has already done this. Eligible-but-satisfied plays
    # are withheld: advising a state that already holds is the failure this
    # file was written for.
    satisfied: Optional[Callable[[Dict[str, Any]], bool]] = None

    def status(self, snap: Dict[str, Any]) -> str:
        if self.satisfied is not None and self.satisfied(snap):
            return "already true"
        if self.precondition is None:
            return "available"
        return "eligible" if self.precondition(snap) else "precondition not met"


# --- seed: under 25 members -------------------------------------------------

PLAYS: Tuple[Play, ...] = (
    Play(
        id="answer_unanswered",
        stages=("seed", "sprout", "growing", "community"),
        title="Answer the unanswered message",
        when="a member question is sitting unanswered",
        precondition=lambda s: _count(s, "unanswered_questions") > 0,
        satisfied=lambda s: _count(s, "unanswered_questions") == 0,
    ),
    Play(
        id="ask_one_silent_member",
        stages=("seed", "sprout"),
        title="Ask one question of one silent member",
        when="somebody has never posted, or has gone quiet",
        precondition=lambda s: (_count(s, "members", "never_posted")
                                + _count(s, "members", "gone_quiet")) > 0,
    ),
    Play(
        id="merge_channels",
        stages=("seed", "sprout"),
        title="Merge channels down",
        when="more than three channels exist and most are dead",
        # The play that produced the bad advice. Both halves of its stated
        # condition are now arithmetic: more than three channels, and most of
        # them dead. One channel fails the first clause outright.
        precondition=lambda s: (_count(s, "channels") > 3
                                and _count(s, "dead_channels") * 2 >= _count(s, "channels")),
        satisfied=lambda s: _count(s, "channels") <= 1,
    ),
    Play(
        id="post_the_work",
        stages=("seed", "sprout"),
        title="Post the work, not an announcement",
        when="the founder has gone quiet",
        precondition=lambda s: (_m(s, "activity.days_since_owner_posted", 0) or 0) >= 7,
        satisfied=lambda s: (_m(s, "activity.days_since_owner_posted", 999) or 999) < 3,
    ),
    Play(
        id="show_a_failure",
        stages=("seed", "sprout"),
        title="Show a failure",
        when="the product is unfinished and posting feels premature",
        # Whether the founder is reluctant is not in any table. Available, and
        # honest about why there is no check.
        precondition=None,
    ),
    Play(
        id="recruit_against_request",
        stages=("seed", "sprout"),
        title="Fulfil one specific request by hand, in public, where the audience already is",
        when="there is bandwidth for outbound",
        precondition=None,
    ),
    Play(
        id="convert_a_lurker",
        stages=("seed", "sprout"),
        title="Convert a lurker with a gift",
        when="somebody joined and never spoke",
        precondition=lambda s: _count(s, "members", "never_posted") > 0,
    ),
    Play(
        id="do_nothing_to_the_server",
        stages=("seed", "sprout", "growing", "community"),
        title="Do nothing to the server; the bottleneck is upstream",
        when="the server is quiet and the product has no users",
        precondition=lambda s: (_m(s, "activity.messages_30d", 0) or 0) == 0,
    ),

    # --- sprout: 25-99 -----------------------------------------------------
    Play(
        id="start_one_ritual",
        stages=("sprout", "growing"),
        title="Start one recurring ritual",
        when="activity is regular but shapeless",
        precondition=lambda s: (_m(s, "activity.messages_30d", 0) or 0) >= 30,
    ),
    Play(
        id="introduce_two_members",
        stages=("sprout", "growing"),
        title="Introduce two members to each other",
        when="people talk only to the founder",
        precondition=lambda s: _count(s, "structure", "mentioned_only_by_owner") > 0,
    ),
    Play(
        id="give_regulars_a_role",
        stages=("sprout", "growing"),
        title="Give the first regulars a role",
        when="a handful of members post consistently",
        precondition=lambda s: (_m(s, "activity.unique_posters_30d", 0) or 0) >= 3,
    ),
    Play(
        id="purpose_in_the_topic",
        stages=("sprout", "growing"),
        title="Write the one-line purpose into the channel topic",
        when="new members join and never post",
        precondition=lambda s: _count(s, "members", "never_posted") > 0,
    ),

    # --- growing: 100-499 --------------------------------------------------
    Play(
        id="fix_time_to_first_response",
        stages=("growing", "community"),
        title="Fix time-to-first-response before anything else",
        when="newcomer questions sit unanswered",
        precondition=lambda s: _count(s, "unanswered_questions") > 0,
    ),
    Play(
        id="split_an_overloaded_channel",
        stages=("growing", "community"),
        title="Split a channel, but only one that is demonstrably overloaded",
        when="a single channel carries most messages",
        precondition=lambda s: _count(s, "channels") >= 2,
    ),
    Play(
        id="one_question_onboarding",
        stages=("growing", "community"),
        title="Onboarding that asks one question",
        when="joins are healthy but activation is low",
        precondition=lambda s: (_m(s, "membership_flow_30d.joins", 0) or 0) >= 10,
    ),
    Play(
        id="delegate_moderation",
        stages=("growing", "community"),
        title="Delegate moderation before you need it",
        when="the founder is the only one with permissions",
        precondition=None,
    ),
    Play(
        id="engagement_mechanics",
        stages=("growing", "community"),
        title="Now the engagement mechanics work — progression, leaderboard, prestige",
        when="retention is healthy and you want depth",
        precondition=lambda s: (_m(s, "members.humans", 0) or 0) >= 100,
    ),

    # --- community: 500+ ---------------------------------------------------
    Play(
        id="read_the_retention_curve",
        stages=("community",),
        title="Read the retention curve, not the member count",
        when="always, at this size",
        precondition=lambda s: True,
    ),
    Play(
        id="house_a_sub_community",
        stages=("community",),
        title="Find the sub-community and give it a room",
        when="a topic recurs across channels",
        precondition=None,
    ),
    Play(
        id="programme_around_regulars",
        stages=("community",),
        title="Programme around the regulars, not the newcomers",
        when="a small core carries most activity",
        precondition=lambda s: (_m(s, "top_share_30d.top1.share", 0) or 0) >= 0.3,
    ),
    Play(
        id="automate_welcome_keep_reply_human",
        stages=("community",),
        title="Automate the welcome, keep the reply human",
        when="joins outpace the founder's capacity",
        precondition=lambda s: (_m(s, "membership_flow_30d.joins", 0) or 0) >= 50,
    ),
    Play(
        id="prune_on_evidence",
        stages=("community",),
        title="Prune the channel list on evidence",
        when="dead channels outnumber live ones",
        precondition=lambda s: (_count(s, "dead_channels") * 2 > _count(s, "channels")
                                and _count(s, "channels") > 3),
        satisfied=lambda s: _count(s, "channels") <= 1,
    ),
)

BY_ID = dict((p.id, p) for p in PLAYS)


def for_stage(stage: str) -> List[Play]:
    return [p for p in PLAYS if stage in p.stages]


def eligible(snap: Dict[str, Any]) -> List[Play]:
    """The plays worth offering for this server, right now.

    A play is offered when its stage matches, its precondition holds (or cannot
    be computed), and the founder has not already done it.
    """
    stage = snap.get("stage") or "seed"
    out = []
    for play in for_stage(stage):
        if play.satisfied is not None and play.satisfied(snap):
            continue
        if play.precondition is not None and not play.precondition(snap):
            continue
        out.append(play)
    return out


def withheld(snap: Dict[str, Any]) -> List[Tuple[Play, str]]:
    """Plays deliberately not offered, and why. For `cadybot plays`."""
    stage = snap.get("stage") or "seed"
    out = []
    for play in for_stage(stage):
        status = play.status(snap)
        if status != "eligible" and status != "available":
            out.append((play, status))
    return out


def choices(snap: Dict[str, Any]) -> List[str]:
    """The enum a recommendation's `play` field is constrained to."""
    return [p.id for p in eligible(snap)] + ["none"]


def render(snap: Dict[str, Any]) -> str:
    """The eligible plays, for the prompt.

    Only the eligible ones. The full catalogue stays in playbooks/*.md for the
    reasoning to sit on, but the list the model chooses from is computed, so a
    play whose condition does not hold cannot be picked out of prose.
    """
    plays = eligible(snap)
    if not plays:
        return (
            "# Plays available right now\n\n"
            "None of the catalogue's preconditions hold for this server today. "
            "That is a real answer: recommend nothing to the server and say why, "
            "citing a number."
        )
    lines = [
        "# Plays available right now",
        "",
        "Computed from the snapshot, not chosen by you. A play missing from this "
        "list either has an unmet precondition or describes something the founder "
        "has already done — in both cases recommending it is wrong, and `play` "
        "will not accept it.",
        "",
    ]
    for play in plays:
        lines.append("- `%s` — %s _(%s)_" % (play.id, play.title, play.when))
    lines.append("")
    lines.append(
        "Every recommendation must name its `play` from this list. If you find "
        "yourself writing a recommendation whose play is `none`, that is the "
        "signal you are improvising rather than applying the catalogue — either "
        "it is one of these plays and you should say which, or it should not be "
        "a recommendation. `none` is for the case where you genuinely mean it, "
        "and it is common and correct on a small server."
    )
    return "\n".join(lines)
