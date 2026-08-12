"""Look things up before answering.

This is the small loop that turns cadybot from something that interprets one
fixed summary into something that goes and checks. It runs before the reply, it
runs at most twice, and everything it finds is appended to the prompt the
existing code already builds.

Three decisions carry it, and each was arrived at by measurement rather than
taste:

**A gather round is a small round.** It sends `prompts.GATHER_SYSTEM` — a few
hundred tokens naming the six lookups — and never `prompts.stable_prefix()`,
which is about 4,100 tokens of persona and stage gates. Nothing in the stable
prefix helps choose a lookup, and paying for it on every round is the single
thing that makes a tool loop unaffordable on CPU inference. The *final* answer
still gets the full prefix and the full snapshot, unchanged.

**The final answer is produced by today's code with the tools array absent.**
The snapshot stays in that call. It is tempting to drop it once lookups exist,
but `prompts.SYSTEM` says "The snapshot tells you the stage. Obey it", and none
of the six lookups returns a member count or a stage — removing it would leave
the hardest rule in the system reading a number the model can no longer see.

**A failure here is silence, not an error.** `investigate` catches everything
and returns an empty result, and an empty result reproduces exactly today's
behaviour. A lookup layer that can break the reply is worse than no lookup
layer.

Ollama only. On the anthropic backend this short-circuits at step 0 and the
answer path is untouched — partly because the loop was measured against local
inference, and partly because a tools array sits upstream of the cache
breakpoint on `stable_prefix()`'s last block and would miss the prompt cache.
"""

import dataclasses
import json
import time
from typing import Any, Dict, List, Optional

from . import config, llm, probe, prompts

# At most two lookups per round, so one confused round cannot drain the budget.
CALLS_PER_ROUND = 2


@dataclasses.dataclass
class Inquiry:
    findings: List[probe.Finding]
    digest: str                    # appended to the final prompt
    facts: List[Dict[str, Any]]    # every Finding.facts, for the verifier
    footer: str                    # the code-written "checked:" line
    stopped: str                   # answered | rounds | budget | no_progress | error | off
    seconds: float = 0.0

    def __bool__(self) -> bool:
        return bool(self.findings)


def _empty(stopped: str) -> Inquiry:
    return Inquiry([], "", [], "", stopped)


def _sniff(content: Optional[str]) -> List[Dict[str, Any]]:
    """Recover a tool call the model wrote as prose instead of as a tool call.

    Ollama returns text it could not match against the tools array as ordinary
    content, so an empty `tool_calls` is not evidence the model declined to call
    anything — often it emitted a bare {"name":..., "arguments":{...}} object.
    Small models do this often enough that not looking is throwing away rounds.
    """
    if not content or "{" not in content:
        return []
    chunk = content[content.find("{"):content.rfind("}") + 1]
    try:
        blob = json.loads(chunk)
    except ValueError:
        return []
    if isinstance(blob, dict) and isinstance(blob.get("name"), str):
        args = blob.get("arguments") or blob.get("parameters") or {}
        return [{"function": {"name": blob["name"], "arguments": args}}]
    return []


def investigate(
    guild_id: int,
    question: str,
    max_rounds: int = 2,
    budget_s: float = 120.0,
) -> Inquiry:
    """Run up to `max_rounds` lookups against this guild's own records.

    Never raises. `guild_id` is passed positionally to probe.run and is not a
    parameter of any tool, so the model cannot name a different server.
    """
    if (config.BACKEND or "").lower() != "ollama":
        return _empty("off")
    if max_rounds <= 0:
        return _empty("off")

    started = time.monotonic()
    deadline = started + budget_s
    transcript: List[Dict[str, Any]] = [{"role": "user", "content": question}]
    findings: List[probe.Finding] = []
    seen: Dict[str, str] = {}
    repeats = 0
    stopped = "rounds"
    tools = probe.schemas()

    for _round in range(max_rounds):
        remaining = deadline - time.monotonic()
        if remaining <= 5:
            stopped = "budget"
            break
        try:
            message = llm.tool_round(
                prompts.GATHER_SYSTEM, transcript, tools, guild_id,
                keep_alive="120s",
                timeout=min(config.OLLAMA_TIMEOUT, remaining),
            )
        except Exception:                       # noqa: BLE001 - a lookup is optional
            stopped = "error"
            break

        calls = message.get("tool_calls") or _sniff(message.get("content"))
        if not calls:
            stopped = "answered"
            break

        transcript.append({"role": "assistant",
                           "content": message.get("content") or "",
                           "tool_calls": message.get("tool_calls") or []})

        progressed = False
        for call in calls[:CALLS_PER_ROUND]:
            fn = (call or {}).get("function") or {}
            name = fn.get("name") or ""
            args = fn.get("arguments") or {}
            if isinstance(args, str):
                try:
                    args = json.loads(args)
                except ValueError:
                    args = {}
            key = "%s(%s)" % (name, json.dumps(args, sort_keys=True, default=str))
            if key in seen:
                transcript.append({"role": "tool", "tool_name": name,
                                   "content": "identical lookup, already run above."})
                continue
            progressed = True
            finding = probe.run(guild_id, name, args,
                                deadline_s=max(1.0, deadline - time.monotonic()))
            finding.tag = "F%d" % (len(findings) + 1)
            findings.append(finding)
            seen[key] = finding.tag
            transcript.append({"role": "tool", "tool_name": name,
                               "content": finding.body[:probe.PROBE_BODY_CHARS]})

        if not progressed:
            repeats += 1
            if repeats >= 2:
                stopped = "no_progress"
                break

    return Inquiry(
        findings=findings,
        digest=probe.render(findings),
        facts=[f.facts for f in findings if f.facts],
        footer=footer(findings),
        stopped=stopped,
        seconds=time.monotonic() - started,
    )


def footer(findings: List[probe.Finding]) -> str:
    """The line saying what was actually checked.

    Written by code, never by the model, so it cannot claim a lookup it did not
    run. This is the whole difference between an assistant that says it checked
    and one that did.
    """
    if not findings:
        return ""
    parts = []
    for f in findings:
        args = ", ".join("%s=%s" % kv for kv in sorted(f.args.items()))
        parts.append("`%s(%s)`%s" % (f.tool, args, " — nothing" if not f.rows else ""))
    return "_checked: %s_" % ", ".join(parts)
