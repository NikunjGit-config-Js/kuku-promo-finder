"""Hard checks every output promo must pass. These are the answer to 'how do you know the timestamps are valid?'"""
from __future__ import annotations

from . import config as C
from .parser import Episode


def check_segment(ep: Episode, start: float, end: float) -> list[str]:
    errs = []
    if not (0 <= start < end <= ep.duration + C.EPS):
        errs.append(f"segment {start}-{end} outside episode bounds 0-{ep.duration}")
    tev = ep.timed_events
    starts = {round(e.start, 2) for e in tev} | {round(s.start, 2) for s in ep.scenes if s.start is not None}
    ends = {round(e.end, 2) for e in tev} | {round(s.end, 2) for s in ep.scenes if s.end is not None}
    if round(start, 2) not in starts:
        errs.append(f"start {start} is not an event/scene boundary in the source")
    if round(end, 2) not in ends:
        errs.append(f"end {end} is not an event/scene boundary in the source")
    for e in tev:
        if e.kind == "dialogue" and (e.start + C.EPS < start < e.end - C.EPS or e.start + C.EPS < end < e.end - C.EPS):
            errs.append(f"cuts through dialogue {e.eid}")
    return errs


def check_promo(p: dict, episodes: dict[int, Episode]) -> list[str]:
    """p is an output promo record (see pipeline.promo_record)."""
    errs, total = [], 0.0
    prev = None
    for seg in p["segments"]:
        ep = episodes.get(seg["episode"])
        if ep is None:
            errs.append(f"episode {seg['episode']} does not exist")
            continue
        errs += check_segment(ep, seg["start_s"], seg["end_s"])
        total += seg["end_s"] - seg["start_s"]
        if prev and (seg["episode"], seg["start_s"]) == prev:
            errs.append("duplicate segment")
        prev = (seg["episode"], seg["start_s"])
    if not (C.MIN_DUR - C.DUR_TOL <= total <= C.MAX_DUR + C.DUR_TOL):
        errs.append(f"total duration {total:.2f}s outside {C.MIN_DUR}-{C.MAX_DUR}s")
    if abs(total - p["total_duration_s"]) > C.EPS:
        errs.append("reported duration does not match segments")
    first = p["segments"][0]
    ep = episodes.get(first["episode"])
    for role in ("hook", "cliffhanger"):
        ref = p[role]
        src = next((e for e in ep.events if e.eid == ref["event_id"]), None) if ep else None
        if src is None:
            errs.append(f"{role} event {ref['event_id']} not found in source")
            continue
        if ref.get("line") and ref["line"] != src.text:
            errs.append(f"{role} text does not match source event {src.eid}")
    if ep and p["hook"].get("offset_s", 99) > C.HOOK_WINDOW + C.EPS:
        errs.append(f"hook starts {p['hook']['offset_s']}s in (> {C.HOOK_WINDOW}s)")
    return errs
