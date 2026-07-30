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

    if response.status_code == 404:
        raise BackendError(
            "Ollama has no model %r. Run `ollama pull %s`."
            % (config.OLLAMA_MODEL, config.OLLAMA_MODEL)
        )
    response.raise_for_status()
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


def generate(
    system_blocks, user_text: str, schema: Type[T], kind: str, guild_id: Optional[int] = None
) -> T:
    if config.BACKEND == "ollama":
        return _ollama(system_blocks, user_text, schema, kind, guild_id)
    if config.BACKEND == "anthropic":
        return _claude(system_blocks, user_text, schema, kind, guild_id)
    raise BackendError("Unknown CADYBOT_BACKEND %r (use 'ollama' or 'anthropic')." % config.BACKEND)


def describe() -> str:
    if config.BACKEND == "ollama":
        return "%s (local)" % config.OLLAMA_MODEL
    return config.MODEL


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
