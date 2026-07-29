"""Claude calls.

Two entry points: `ask` (a direct verdict on one question) and `brief` (ranked
recommendations). Both take the deterministic snapshot as input — Claude is only
ever asked to interpret numbers, never to produce them.

Structured output goes through `client.messages.parse()` with Pydantic models, so
a malformed response is impossible rather than merely unlikely.
"""

import json
from typing import Any, Dict, List, Literal, Optional

import anthropic
from pydantic import BaseModel, Field

from . import config, db, prompts

_client: Optional[anthropic.Anthropic] = None

MAX_TOKENS = 8000


def client() -> anthropic.Anthropic:
    """A bare constructor also picks up an `ant auth login` profile."""
    global _client
    if _client is None:
        _client = anthropic.Anthropic()
    return _client


class Verdict(BaseModel):
    # Literal becomes a JSON Schema enum, so an invalid verdict is rejected by
    # the API rather than merely discouraged by the description.
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


class Refused(Exception):
    """Claude's safety classifiers declined the request."""


def _call(schema, instruction: str, snap: Dict[str, Any], question: Optional[str], kind: str):
    """One request. The stable prefix is cached; the volatile part goes last.

    Server-side refusal fallbacks would need the beta `messages.create` path,
    which does not carry Pydantic validation. For a Discord growth advisor a
    refusal is effectively impossible, so this trades that safety net for
    guaranteed-valid output and handles the refusal explicitly instead.
    """
    turn = [
        instruction,
        "# Server snapshot\n\n```json\n%s\n```" % json.dumps(snap, indent=2, default=str),
    ]
    if question:
        turn.append("# The founder's question\n\n%s" % question)

    response = client().messages.parse(
        model=config.MODEL,
        max_tokens=MAX_TOKENS,
        system=prompts.stable_prefix(),
        messages=[{"role": "user", "content": "\n\n".join(turn)}],
        output_format=schema,
    )

    if response.stop_reason == "refusal":
        raise Refused(
            "Claude declined this request (%s). Nothing was billed."
            % getattr(response.stop_details, "category", "unknown")
        )
    if response.stop_reason == "max_tokens":
        raise RuntimeError("Response hit max_tokens; raise MAX_TOKENS in advisor.py.")

    if config.GUILD_ID:
        db.record_run(config.GUILD_ID, kind, response.usage, config.MODEL)

    parsed = response.parsed_output
    if parsed is None:
        raise RuntimeError("Claude returned no parseable output.")
    return parsed


def ask(question: str, snap: Dict[str, Any]) -> Verdict:
    return _call(Verdict, prompts.ASK_INSTRUCTION, snap, question, "ask")


def brief(snap: Dict[str, Any]) -> Brief:
    result = _call(Brief, prompts.BRIEF_INSTRUCTION, snap, None, "brief")
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
        lines += ["**Last time:** %s" % b.follow_up]
    return "\n".join(lines).strip()
