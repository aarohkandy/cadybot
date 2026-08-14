"""One pass of the desk: notice, think, record, and usually say nothing.

This is the loop that makes cadybot self-prompting, and it is deliberately the
smallest file in the feature. Everything that decides *whether* to think lives
in agenda.py, which no model touches; everything that decides *what to say* is
one boolean the model may only ever set to false. What is left here is the
sequence, and the sequence is fixed:

  1. is there budget                     — agenda.affordable, else stop
  2. has anything happened               — agenda.next_provocation, else stop
  3. claim it                            — agenda.open_attempt, else stop
  4. think about it                      — one model call, exactly one
  5. write down what came back           — journal, whether or not it is said
  6. may it be said                      — agenda.may_surface, deterministic
  7. does the model also want it said    — the veto
  8. say it                              — notify.deliver, which logs itself

Steps 1-3 are indexed SELECTs and cost nothing. On a quiet server the pass ends
at step 2 every time, forever, and that is the intended steady state rather than
a degenerate case.

**This module must never call scorecard.score().** `newly_closed` is set in
memory by `scorecard._grade` and never persisted, so a second pass that grades
silently swallows the verdict `loop.nightly` was about to narrate, and the
founder never hears the one thing cadybot exists to tell him. Two of the three
designs this feature was chosen from shipped exactly that bug. The desk reads
`scorecard.open_row` and `scorecard.recent_verdicts` — both pure SELECTs — and
nothing else.

Step 3 before step 4 is the same discipline loop.py applies in the mirror: the
journal row is written *before* the model call, so a thought that crashes still
cost its budget and still consumed its provocation. A failure that is free to
retry is a failure that retries forever.
"""

import functools
from typing import Any, Dict, Optional, Tuple

from . import advisor, agenda, config, db, llm, loop, notify, snapshot


async def think(client, guild_id: int) -> Optional[str]:
    """One pass. Returns the text delivered, or None — usually None.

    `client` may be None, which thinks and journals without delivering. Unlike
    `loop.nightly`, that really is side-effect-free with respect to anything the
    founder sees: no bet is opened, no baseline is advanced, and the only trace
    is a journal row saying what was thought.
    """
    agenda.reap_stale(guild_id)

    # Anything already thought about and still unsaid gets first refusal. The
    # speaking gates are mostly clocks — an afternoon window, a 20-hour gap —
    # and only one tick in four falls inside them, so a thought that has to be
    # written and said in the same pass is usually lost. The sentence is already
    # composed and already checked, so this costs no model call.
    waiting = agenda.unsaid(guild_id)
    if waiting is not None:
        held = agenda.Provocation(
            waiting["kind"], waiting["provoked_by"], "", waiting["about_ref"]
        )
        allowed, _why = agenda.may_surface(guild_id, held)
        if allowed:
            text = advisor.render_stored(waiting)
            if client is not None:
                spoke = await notify.deliver(
                    client, guild_id, text, "thought", waiting["id"]
                )
                if not spoke:
                    return None
            agenda.mark_surfaced(waiting["id"])
            return text

    if not agenda.affordable(guild_id):
        return None

    # No generator reads the snapshot — they all query stored rows — so building
    # it before knowing there is anything to think about was a few hundred
    # milliseconds of SQL on the event loop, four times a day, to answer a
    # question that is almost always "nothing".
    prov = agenda.next_provocation(guild_id, None)
    if prov is None:
        return None                       # the common case, by design

    journal_id = agenda.open_attempt(guild_id, prov)
    if journal_id is None:
        return None                       # someone else has it

    try:
        snap = await loop._offload(client, snapshot.build, guild_id)
        result = await loop._offload(
            client, functools.partial(advisor.reflect, prov, snap, guild_id)
        )
    except Exception as exc:              # noqa: BLE001 - recorded, then re-raised
        agenda.record_failure(journal_id, exc)
        raise

    agenda.record_result(
        journal_id, result, result._unverified, llm.describe(None)
    )

    if not result.worth_telling_founder:
        return None
    if not (result.to_founder or "").strip():
        return None
    if result._unverified:
        # An unprompted message is held to a higher bar than an answer the
        # founder asked for: there is no question here to anchor a number
        # against, so a figure that is not in the snapshot is not worth the
        # interruption. verify_evidence stays advisory everywhere else.
        return None

    allowed, _why = agenda.may_surface(guild_id, prov)
    if not allowed:
        # Kept, not discarded: `unsaid` will offer it again on a tick that is
        # allowed to speak, for the next 48 hours.
        return None

    text = advisor.render_reflection(result, prov)
    if client is not None:
        spoke = await notify.deliver(client, guild_id, text, "thought", journal_id)
        if not spoke:
            return None
    agenda.mark_surfaced(journal_id)
    return text


def preview(guild_id: int) -> Dict[str, Any]:
    """What the next pass would do, without doing any of it.

    Reads only. No journal row, no model call, no budget spent — `cadybot
    reflect` is genuinely free, which matters because the whole point of it is
    to be run repeatedly on a live server to watch the desk decline to act.
    """
    snap = snapshot.build(guild_id)
    prov = agenda.next_provocation(guild_id, snap, create_mark=False)
    spent = db.scalar(
        "SELECT COUNT(*) FROM journal WHERE guild_id=? AND started_at>=?",
        (guild_id, db.hours_ago(24)),
    )
    out = {
        "affordable": agenda.affordable(guild_id),
        "spent_today": spent,
        "budget": config.THINK_CALLS_PER_DAY,
        "provocation": None,
        "self_prompt": None,
        "would_surface": None,
        "why": "nothing has happened that I have not already thought about",
    }
    if agenda.installed_at(guild_id, create=False) is None:
        out["why"] = ("the desk has not started on this server yet — it begins "
                      "on the first scheduled pass, and everything before that "
                      "is history rather than news")
        return out
    if prov is None:
        return out
    allowed, why = agenda.may_surface(guild_id, prov)
    out.update(
        {
            "provocation": prov.kind,
            "provoked_by": prov.provoked_by,
            "about": prov.about_ref,
            "self_prompt": prov.self_prompt,
            "would_surface": allowed,
            "why": why,
        }
    )
    return out
