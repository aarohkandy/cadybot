"""Model backends.

Two of them, same interface: hand over system blocks, a user turn, and a
Pydantic model; get a validated instance back.

- `ollama`    — local, free, private, and noticeably worse at judgment. Fine for
                exercising the plumbing, which is what it is here for.
- `anthropic` — Claude Opus 5. Prompt caching on the stable prefix, schema
                enforced by the API.

Switch with CADYBOT_BACKEND.
"""

import json
from typing import Any, Dict, List, Optional, Type, TypeVar

import httpx
from pydantic import BaseModel, ValidationError

from . import config, db

T = TypeVar("T", bound=BaseModel)

_anthropic_client = None


class Refused(Exception):
    """The model declined the request."""


class BackendError(RuntimeError):
    pass


def _flatten(system_blocks: List[Dict[str, Any]]) -> str:
    return "\n\n".join(b["text"] for b in system_blocks)


def _ollama_error(response) -> None:
    """Turn an Ollama HTTP failure into a BackendError callers already catch.

    httpx's raise_for_status raises HTTPStatusError, which is not a
    BackendError, so it escaped every `except BackendError` in the codebase and
    reached the user as the generic "that failed" message. A 500 here usually
    means the model runner died mid-request — worth saying, because the fix is
    different from every other failure in this module.
    """
    if response.status_code == 404:
        raise BackendError(
            "Ollama has no model %r. Run `ollama pull %s`."
            % (config.OLLAMA_MODEL, config.OLLAMA_MODEL)
        )
    if response.status_code >= 500:
        raise BackendError(
            "Ollama returned %d. The model runner usually died mid-request — "
            "check `ollama ps`, then ask again." % response.status_code
        )
    if response.status_code >= 400:
        raise BackendError(
            "Ollama rejected the request (%d): %s"
            % (response.status_code, response.text[:300])
        )


# --- ollama ----------------------------------------------------------------


def _ollama(system_blocks, user_text: str, schema: Type[T], kind: str, guild_id) -> T:
    """Local inference via Ollama's schema-constrained decoding.

    num_ctx matters more than anything else here. Ollama's default context is
    small enough to silently truncate the system prompt, and a truncated stage
    gate is worse than no stage gate — the model would cheerfully recommend a
    tournament to seven people.
    """
    payload = {
        "model": config.OLLAMA_MODEL,
        "stream": False,
        # Ollama evicts an idle model after 5 minutes. Reloading 9GB from disk
        # on every question adds minutes to a reply that should take seconds,
        # so keep it resident between commands.
        "keep_alive": config.OLLAMA_KEEP_ALIVE,
        "options": {"num_ctx": config.OLLAMA_NUM_CTX, "temperature": 0.2},
        "format": schema.model_json_schema(),
        "messages": [
            {"role": "system", "content": _flatten(system_blocks)},
            {"role": "user", "content": user_text},
        ],
    }

    try:
        response = httpx.post(
            "%s/api/chat" % config.OLLAMA_HOST,
            json=payload,
            timeout=httpx.Timeout(config.OLLAMA_TIMEOUT),
        )
    except httpx.ConnectError as exc:
        raise BackendError(
            "Can't reach Ollama at %s. Is it running? `ollama serve`" % config.OLLAMA_HOST
        ) from exc
    except httpx.ReadTimeout as exc:
        raise BackendError(
            "Ollama timed out after %ds. Try a smaller model or raise "
            "CADYBOT_OLLAMA_TIMEOUT." % config.OLLAMA_TIMEOUT
        ) from exc

    _ollama_error(response)
    body = response.json()

    prompt_tokens = body.get("prompt_eval_count") or 0
    if prompt_tokens >= config.OLLAMA_NUM_CTX - 64:
        raise BackendError(
            "Prompt used %d tokens against a %d-token context — it was almost "
            "certainly truncated. Raise CADYBOT_OLLAMA_NUM_CTX."
            % (prompt_tokens, config.OLLAMA_NUM_CTX)
        )

    content = (body.get("message") or {}).get("content") or ""
    if guild_id:
        db.record_local_run(
            guild_id, kind, config.OLLAMA_MODEL, prompt_tokens, body.get("eval_count") or 0
        )

    try:
        return schema.model_validate_json(content)
    except ValidationError as exc:
        raise BackendError(
            "%s returned output that did not match the schema. Local models are "
            "worse at this than the API; try a different model.\n\n%s\n\n%s"
            % (config.OLLAMA_MODEL, content[:600], exc)
        ) from exc


# --- anthropic -------------------------------------------------------------


def _claude(system_blocks, user_text: str, schema: Type[T], kind: str, guild_id) -> T:
    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()

    response = _anthropic_client.messages.parse(
        model=config.MODEL,
        max_tokens=8000,
        system=system_blocks,
        messages=[{"role": "user", "content": user_text}],
        output_format=schema,
    )

    if response.stop_reason == "refusal":
        raise Refused(
            "Claude declined this request (%s)."
            % getattr(response.stop_details, "category", "unknown")
        )
    if response.stop_reason == "max_tokens":
        raise BackendError("Response hit max_tokens; raise it in llm.py.")

    if guild_id:
        db.record_run(guild_id, kind, response.usage, config.MODEL)

    parsed = response.parsed_output
    if parsed is None:
        raise BackendError("Claude returned no parseable output.")
    return parsed


# --- dispatch --------------------------------------------------------------


BACKENDS = ("ollama", "anthropic")


def generate(
    system_blocks,
    user_text: str,
    schema: Type[T],
    kind: str,
    guild_id: Optional[int] = None,
    backend: Optional[str] = None,
) -> T:
    chosen = backend or config.BACKEND
    if chosen == "ollama":
        return _ollama(system_blocks, user_text, schema, kind, guild_id)
    if chosen == "anthropic":
        return _claude(system_blocks, user_text, schema, kind, guild_id)
    raise BackendError("Unknown CADYBOT_BACKEND %r (use 'ollama' or 'anthropic')." % chosen)


def describe(backend: Optional[str] = None) -> str:
    if (backend or config.BACKEND) == "ollama":
        return "%s (local)" % config.OLLAMA_MODEL
    return config.MODEL


# There is deliberately no "narrate this with the other backend" helper. It
# existed, and because a backend argument selects the model for a whole call, it
# swapped the model that writes the recommendations rather than the model that
# writes one sentence about a verdict — so a week with a verdict due generated
# its entire brief on the local 4B model while CADYBOT_BACKEND said anthropic.
# Panickssery et al. (NeurIPS 2024) is still the reason the scorecard renders
# without authorship, but that effect is addressed by scorecard.py owning the
# verdict outright, not by quietly downgrading the advisor.


def converse(
    system_blocks,
    messages: List[Dict[str, str]],
    guild_id=None,
    backend: Optional[str] = None,
) -> str:
    """Free-text reply for ordinary conversation. No schema — this is talking.

    `messages` is the running exchange, oldest first, each {role, content}.
    """
    if (backend or config.BACKEND) == "ollama":
        payload = {
            "model": config.OLLAMA_MODEL,
            "stream": False,
            "keep_alive": config.OLLAMA_KEEP_ALIVE,
            "options": {"num_ctx": config.OLLAMA_NUM_CTX, "temperature": 0.4},
            "messages": [{"role": "system", "content": _flatten(system_blocks)}] + messages,
        }
        try:
            response = httpx.post(
                "%s/api/chat" % config.OLLAMA_HOST,
                json=payload,
                timeout=httpx.Timeout(config.OLLAMA_TIMEOUT),
            )
        except httpx.ConnectError as exc:
            raise BackendError(
                "Can't reach Ollama at %s. Is it running?" % config.OLLAMA_HOST
            ) from exc
        except httpx.ReadTimeout as exc:
            raise BackendError("Ollama timed out after %ds." % config.OLLAMA_TIMEOUT) from exc
        _ollama_error(response)
        body = response.json()
        if guild_id:
            db.record_local_run(
                guild_id, "chat", config.OLLAMA_MODEL,
                body.get("prompt_eval_count") or 0, body.get("eval_count") or 0,
            )
        return ((body.get("message") or {}).get("content") or "").strip()

    global _anthropic_client
    import anthropic

    if _anthropic_client is None:
        _anthropic_client = anthropic.Anthropic()
    response = _anthropic_client.messages.create(
        model=config.MODEL,
        max_tokens=2000,
        system=system_blocks,
        messages=messages,
    )
    if response.stop_reason == "refusal":
        raise Refused("Claude declined that one.")
    if guild_id:
        db.record_run(guild_id, "chat", response.usage, config.MODEL)
    return "".join(b.text for b in response.content if b.type == "text").strip()


def warm() -> None:
    """Load the local model into memory so the first question isn't a cold start.

    Best effort — a failure here only costs latency, never correctness.
    """
    if config.BACKEND != "ollama":
        return
    # Warming is a trade: hold memory now to save a reload later. At keep_alive
    # 0 there is no later — the model unloads the moment this call returns — so
    # warming would allocate several gigabytes purely to free them again.
    if config.OLLAMA_KEEP_ALIVE in (0, "0"):
        return
    try:
        httpx.post(
            "%s/api/generate" % config.OLLAMA_HOST,
            json={"model": config.OLLAMA_MODEL, "keep_alive": config.OLLAMA_KEEP_ALIVE},
            timeout=httpx.Timeout(config.OLLAMA_TIMEOUT),
        )
    except Exception:
        pass


def preflight() -> Optional[str]:
    """Return a human-readable problem with the configured backend, or None."""
    if config.BACKEND != "ollama":
        return None
    try:
        r = httpx.get("%s/api/tags" % config.OLLAMA_HOST, timeout=5)
        r.raise_for_status()
    except Exception:
        return "Ollama is not reachable at %s. Start it with `ollama serve`." % config.OLLAMA_HOST
    names = {m.get("name", "") for m in r.json().get("models", [])}
    if config.OLLAMA_MODEL not in names and ("%s:latest" % config.OLLAMA_MODEL) not in names:
        return "Ollama has no model %r. Available: %s" % (
            config.OLLAMA_MODEL,
            ", ".join(sorted(names)) or "none",
        )
    return None
