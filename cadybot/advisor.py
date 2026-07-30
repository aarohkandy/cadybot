"""What to ask the model, and how to render what comes back.

Two entry points: `ask` (a verdict on one question) and `brief` (ranked
recommendations). Both take the deterministic snapshot as input — the model is
only ever asked to interpret numbers, never to produce them.

Output is schema-constrained on both backends, so a malformed response is
rejected rather than merely unlikely.
"""

import json
from typing import Any, Dict, List, Literal, Optional

from pydantic import BaseModel, Field

from . import config, db, llm, prompts

# Re-exported so callers catch one name regardless of backend.
Refused = llm.Refused
BackendError = llm.BackendError


class Verdict(BaseModel):
    # Literal becomes a JSON Schema enum, so an invalid verdict is rejected by
    # the decoder rather than merely discouraged by the description.
    verdict: Literal["yes", "no", "not_yet"]
    reason: str = Field(description="At most three sentences, citing a number or name.")
    instead: Optional[str] = Field(
        default=None, description="Concrete alternative action. Required unless verdict is yes."
    )
    confidence: Literal["low", "medium", "high"]


class Recommendation(BaseModel):
    headline: str
    action: str
    evidence: str
    metric: str
    prediction: str


class Brief(BaseModel):
    headline: str
    recommendations: List[Recommendation]
    dont: Optional[str] = None
    follow_up: Optional[str] = None


def _turn(instruction: str, snap: Dict[str, Any], question: Optional[str]) -> str:
    parts = [
        instruction,
        "# Server snapshot\n\n```json\n%s\n```" % json.dumps(snap, indent=2, default=str),
    ]
    if question:
        parts.append("# The founder's question\n\n%s" % question)
    return "\n\n".join(parts)


def ask(question: str, snap: Dict[str, Any]) -> Verdict:
    return llm.generate(
        prompts.stable_prefix(),
        _turn(prompts.ASK_INSTRUCTION, snap, question),
        Verdict,
        "ask",
    )


def brief(snap: Dict[str, Any]) -> Brief:
    result = llm.generate(
        prompts.stable_prefix(),
        _turn(prompts.BRIEF_INSTRUCTION, snap, None),
        Brief,
        "brief",
    )
    if config.GUILD_ID:
        db.save_recommendations(
            config.GUILD_ID, [r.model_dump() for r in result.recommendations]
        )
    return result


# --- rendering -------------------------------------------------------------

_MARK = {"yes": "YES", "no": "NO", "not_yet": "NOT YET"}


def render_verdict(v: Verdict) -> str:
    lines = ["**%s**" % _MARK.get(v.verdict, v.verdict.upper()), "", v.reason]
    if v.instead:
        lines += ["", "**Instead:** %s" % v.instead]
    if v.confidence == "low":
        lines += ["", "_Low confidence — not enough data yet to be sure._"]
    lines += ["", "_%s_" % llm.describe()]
    return "\n".join(lines)


def render_brief(b: Brief) -> str:
    lines = [b.headline, ""]
    for i, r in enumerate(b.recommendations, 1):
        lines += [
            "**%d. %s**" % (i, r.headline),
            r.action,
            "_Why:_ %s" % r.evidence,
            "_Watch:_ %s — %s" % (r.metric, r.prediction),
            "",
        ]
    if b.dont:
        lines += ["**Don't:** %s" % b.dont, ""]
    if b.follow_up:
        lines += ["**Last time:** %s" % b.follow_up, ""]
    lines += ["_%s_" % llm.describe()]
    return "\n".join(lines).strip()
