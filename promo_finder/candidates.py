"""Candidate promo windows: legal cut points, heuristic features, scoring and de-duplication.

Everything here is deterministic code. Start/end times are always copied from source
event boundaries; nothing is interpolated or invented.
"""
from __future__ import annotations

import bisect
import re
from dataclasses import dataclass, field

from . import config as C
from .parser import Episode, Event, fmt_ts

_DELIV = {w: wt for wt, s in C.DELIVERY.items() for w in s.split()}
_ACTION = {wt: set(s.split()) for wt, s in C.ACTION_STRONG.items()}
_SFX = {wt: set(s.split()) for wt, s in C.SFX_STRONG.items()}
_THREAT = {"kill", "dead", "die", "burn", "hang", "ruin", "destroy", "never", "gallows", "poison", "prison",
           "punish", "ash", "end", "warn", "nothing", "curse"}


def _words(text: str) -> list[str]:
    return re.findall(r"[a-z']+", text.lower())


def _tier(text: str, tiers: dict[float, set]) -> float:
    ws = set(_words(text))
    return max((wt for wt, vocab in tiers.items() if ws & vocab), default=0.0)


def delivery_score(delivery: str | None) -> float:
    if not delivery:
        return C.DELIVERY_DEFAULT
    vals = []
    for w in re.findall(r"[a-z]+", delivery.lower()):
        if w not in _DELIV and w.endswith("ly") and w[:-2] in _DELIV:
            w = w[:-2]
        vals.append(_DELIV.get(w, C.DELIVERY_DEFAULT))
    vals = vals or [C.DELIVERY_DEFAULT]
    return 0.7 * max(vals) + 0.3 * sum(vals) / len(vals)


def line_text_score(text: str) -> float:
    ws = _words(text)
    s = min(0.6, 0.3 * len(set(ws) & C.LINE_STRONG))
    s += 0.2 if "!" in text else 0.0
    s += 0.15 if "?" in text else 0.0
    s += 0.05 if text.strip().endswith("...") or text.strip().startswith("...") else 0.0
    s += 0.05 if len(ws) <= 6 and ("!" in text or "?" in text) else 0.0
    return min(1.0, s)


def event_intensity(ev: Event) -> float:
    """0..1 'would this stop a scroll' score for one event, from its own text only."""
    if ev.kind == "dialogue":
        return min(1.0, 0.6 * delivery_score(ev.delivery) + 0.4 * line_text_score(ev.text) + 0.05)
    if ev.kind == "action":
        return max(0.15, _tier(ev.text, _ACTION))
    if ev.kind == "sfx":
        return max(0.15, _tier(ev.text, _SFX), 0.8 * _tier(ev.text, _ACTION))
    if ev.kind == "camera":
        base = 0.35 if any(k in ev.text.lower() for k in C.CAMERA_DRAMATIC) else 0.1
        return max(base, 0.6 * _tier(ev.text, _ACTION))
    return 0.1


@dataclass
class Window:
    episode: int
    start: float
    end: float
    events: list[Event]
    feats: dict = field(default_factory=dict)
    score: float = 0.0
    why: str = ""
    hook_event: Event | None = None
    cliff_event: Event | None = None
    hook_desc: str = ""
    cliff_desc: str = ""
    method: str = "heuristic"

    @property
    def duration(self) -> float:
        return round(self.end - self.start, 2)

    def iou(self, other: "Window") -> float:
        if other.episode != self.episode:
            return 0.0
        inter = max(0.0, min(self.end, other.end) - max(self.start, other.start))
        union = max(self.end, other.end) - min(self.start, other.start)
        return inter / union if union > 0 else 0.0


def legal_cuts(ep: Episode) -> tuple[list[tuple[float, bool]], list[tuple[float, bool]]]:
    """Cut-in and cut-out times taken from event boundaries.

    A time is legal only if no dialogue line is playing across it (never cut mid-sentence).
    'clean' additionally means it is not inside a camera shot or an action/SFX event.
    """
    tev = ep.timed_events
    dlg = [(e.start, e.end) for e in tev if e.kind == "dialogue"]
    cams = [(e.start, e.end) for e in tev if e.kind == "camera"]
    acts = [(e.start, e.end) for e in tev if e.kind in ("action", "sfx")]

    def inside(t, spans):
        return any(a + C.EPS < t < b - C.EPS for a, b in spans)

    ins = {round(e.start, 2) for e in tev} | {round(s.start, 2) for s in ep.scenes if s.start is not None}
    outs = {round(e.end, 2) for e in tev} | {round(s.end, 2) for s in ep.scenes if s.end is not None}
    cut_in = [(t, not inside(t, cams) and not inside(t, acts)) for t in sorted(ins) if not inside(t, dlg)]
    cut_out = [(t, not inside(t, cams) and not inside(t, acts)) for t in sorted(outs) if not inside(t, dlg)]
    return cut_in, cut_out


def _short(ev: Event, n: int = 90) -> str:
    t = ev.text if len(ev.text) <= n else ev.text[: n - 1] + "…"
    if ev.kind == "dialogue":
        who = ev.speaker + (f" ({ev.delivery})" if ev.delivery else "")
        return f'{who}: "{t}"'
    return f"[{ev.kind.upper()}] {t}"


def _hook_desc(ev: Event) -> str:
    if ev.kind == "dialogue":
        if set(_words(ev.text)) & C.LINE_STRONG:
            kind = "shocking line"
        elif "?" in ev.text:
            kind = "confrontational question"
        elif "!" in ev.text:
            kind = "outburst"
        else:
            kind = "charged line"
        return f"{kind} — {_short(ev)}"
    return f"{'physical moment' if ev.kind == 'action' else 'sound sting' if ev.kind == 'sfx' else 'striking shot'} — {_short(ev)}"


def _a(phrase: str) -> str:
    return ("an " if phrase[:1].lower() in "aeiou" else "a ") + phrase


def _cliff_kind(ev: Event, tail_strong: bool) -> str:
    ws = set(_words(ev.text))
    if "?" in ev.text:
        return "unanswered question"
    if ws & _THREAT:
        return "threat / stakes line"
    if ev.text.strip().endswith("..."):
        return "line cut off mid-thought"
    if ws & C.LINE_STRONG:
        return "reveal line"
    if "!" in ev.text:
        return "outburst"
    return "ominous reaction" if tail_strong else "charged line"


def _name_tokens(ep: Episode) -> set[str]:
    generic = {"the", "lady", "lord", "sir", "former", "in", "of", "gold", "gown", "with", "cap", "young", "old",
               "elder", "first", "second", "man", "woman", "voice", "guard", "maid", "o.s.", "v.o."}
    toks = set()
    for e in ep.events:
        if e.kind == "dialogue" and e.speaker:
            toks |= {w for w in re.findall(r"[a-z]+", e.speaker.lower()) if w not in generic and len(w) > 2}
    return toks


_CLAUSE = re.compile(r"[^?!.\u2026]+[?!.\u2026]*")
_ANSWER_PHRASES = ("that is", "of course")


def _first_clause(text: str, names: set[str] = frozenset()) -> str:
    """What the viewer hears first: the first sentence or clause. A lone vocative name ('Ethan?') is skipped."""
    parts = [x.strip() for x in _CLAUSE.findall(text) if x.strip(" .?!\u2026")]
    if len(parts) > 1 and len(_words(parts[0])) == 1 and _words(parts[0])[0] in names:
        parts = parts[1:]
    return parts[0] if parts else text.strip()


def _is_reply(first: Event, prev: Event | None, s: float, names: set[str] = frozenset()) -> bool:
    """The clip opens on a reaction to a line the viewer never hears.

    Within 2 s of another speaker's line, any of: the whole line is short (<= 4 words); it starts with an answer
    word ("Because", "No", "Not"...); the previous line was a question; or the FIRST CLAUSE heard is a short
    question/exclamation ("You really mean that?", "Months?"), even if more words follow."""
    if prev is None or prev.speaker == first.speaker or first.start - prev.end > 2.0 or first.start > s + C.HOOK_WINDOW:
        return False
    w = _words(first.text)
    first_word = w[0] if w else ""
    answer = first_word in {m.strip() for m in C.ANSWER_MARKERS} or " ".join(w[:2]) in _ANSWER_PHRASES
    clause = _first_clause(first.text, names)
    short_reaction = len(_words(clause)) <= 4 and clause.rstrip().endswith(("?", "!"))
    return len(w) <= 4 or answer or prev.text.rstrip().endswith("?") or short_reaction


def _quote(text: str, n: int = 9) -> str:
    w = text.split()
    return " ".join(w[:n]) + ("…" if len(w) > n else "")


def score_window(ep: Episode, s: float, e: float, s_clean: bool, e_clean: bool, tev: list[Event],
                 starts: list[float], inten: dict[str, float], names: set[str]) -> Window | None:
    """Heuristic features + clip-quality score for window [s, e]; None if it fails an eligibility rule."""
    i0 = bisect.bisect_left(starts, s - C.EPS)
    i1 = bisect.bisect_left(starts, e - C.EPS)
    evs = tev[i0:i1]
    dl = [x for x in evs if x.kind == "dialogue"]
    if len(dl) < C.MIN_DIALOGUE_LINES:
        return None
    last = dl[-1]
    tail = e - last.end
    if tail > C.MAX_DEAD_TAIL or tail < -C.EPS:
        return None
    prev_dl = next((x for x in reversed(tev[:i0]) if x.kind == "dialogue"), None)
    reply_open = _is_reply(dl[0], prev_dl, s, names)

    # Hook: strongest event in the first 3 s. Full credit only if it lands in the first second.
    best_hook, hook_ev = 0.0, None
    for x in (x for x in evs if x.start <= s + C.HOOK_WINDOW + C.EPS):
        v = inten[x.eid] * (1.0 - 0.3 * max(0.0, (x.start - s) - 1.0) / (C.HOOK_WINDOW - 1.0))
        if x.kind == "camera":
            v *= 0.8
        elif x.kind == "dialogue":   # 1-3 word lines ("Not anymore.") carry little meaning for a stranger
            v *= min(1.0, 0.6 + 0.1 * len(_words(x.text)))
        if v > best_hook:
            best_hook, hook_ev = v, x
    if hook_ev is None or best_hook < C.HOOK_MIN_INTENSITY:
        return None
    hook = min(1.0, best_hook + (0.1 if hook_ev.kind == "dialogue" else 0.0))
    if reply_open and hook_ev.kind == "dialogue":
        hook *= 0.75                 # an answer without its question is not a hook for a stranger

    # Cliffhanger: what is left open at the cut. Structure first, punctuation second.
    after = tev[i1:]
    nxt = next((x for x in after if x.kind == "dialogue"), None)
    pending = bool(nxt and nxt.speaker != last.speaker and nxt.start - e <= 5.0)
    tail_text = " ".join(x.text.lower() for x in evs if x.start >= last.end - 0.5)
    scene_end = any(abs(e - sc.end) <= C.EPS for sc in ep.scenes if sc.end is not None)
    beat = scene_end or any(w in tail_text for w in C.TAIL_CLIFF_WORDS)   # writer marked a beat at the cut
    lw = set(_words(last.text))
    cliff = 0.35 * inten[last.eid]
    cliff += 0.12 if "?" in last.text else 0.0
    cliff += 0.08 if (lw & C.LINE_STRONG or "!" in last.text) else 0.0
    cliff += 0.08 if last.text.strip().endswith("...") or last.text.strip().endswith("…") else 0.0
    cliff += 0.18 if pending else 0.0
    cliff += 0.15 if beat else 0.0
    cliff += 0.05 if e >= ep.duration - 0.5 else 0.0
    cliff += 0.05 if tail <= 4.0 else (-0.1 if tail > 6.0 else 0.0)
    if last.delivery and any(w in last.delivery.lower() for w in C.RESOLVED_DELIVERY) and inten[last.eid] < 0.6:
        cliff -= 0.25
    cliff = max(0.0, min(1.0, cliff))

    # Standalone context: clear sides, named people/relationships, open conflict, little backstory.
    speakers = {x.speaker for x in dl}
    text = " ".join(x.text.lower() for x in dl)
    tw = set(_words(text))
    spk = {1: 0.5, 2: 1.0, 3: 1.0, 4: 0.7}.get(len(speakers), 0.4)
    rel = min(0.3, 0.15 * len(tw & C.RELATION_TERMS))
    named = 0.15 if tw & names else 0.0
    hot = {x.speaker for x in dl if inten[x.eid] >= 0.65}
    conflict = len(hot) >= 2
    dep = min(0.3, 0.1 * sum(1 for p in C.CONTEXT_DEPENDENT if p in text))
    context = max(0.0, min(1.0, 0.5 * spk + rel + named + (0.2 if conflict else 0.0) - dep
                           - (0.15 if reply_open else 0.0)))

    intensity = sum(inten[x.eid] for x in dl) / len(dl)
    top_share = max(sum(1 for x in dl if x.speaker == sp) for sp in speakers) / len(dl)
    monologue = len(dl) >= 6 and top_share > 0.7
    intensity = max(0.0, intensity - (0.15 if monologue else 0.0))

    # Escalation: several beats (turns / strong actions) and the second half at least as hot as the first.
    turns = sum(1 for a_, b_ in zip(dl, dl[1:]) if a_.speaker != b_.speaker)
    beats = turns + sum(1 for x in evs if x.kind in ("action", "sfx") and inten[x.eid] >= 0.75)
    mid = s + (e - s) / 2
    h1 = [inten[x.eid] for x in evs if x.kind != "camera" and x.start < mid] or [0.0]
    h2 = [inten[x.eid] for x in evs if x.kind != "camera" and x.start >= mid] or [0.0]
    escalation = 0.5 * min(1.0, beats / 5) + 0.5 * max(0.0, min(1.0, 0.5 + 1.5 * (max(h2) - sum(h1) / len(h1))))

    gaps = [b_.start - a_.start for a_, b_ in zip(dl, dl[1:])] + [dl[0].start - s]
    edit = 1.0 - (0.0 if s_clean else 0.25) - (0.0 if e_clean else 0.25)
    edit -= 0.1 if not (25.0 <= e - s <= 45.0) else 0.0
    edit -= 0.2 if max(gaps) > 8.0 else 0.0
    edit = max(0.0, edit)

    f = {"hook": round(hook, 3), "cliff": round(cliff, 3), "context": round(context, 3),
         "escalation": round(escalation, 3), "intensity": round(intensity, 3)}
    score = sum(C.W_HEUR[k] * f[k] for k in C.W_HEUR)
    w = Window(ep.number, s, e, evs, feats=f, score=round(score, 4), hook_event=hook_ev, cliff_event=last)
    w.feats.update({"edit_effort": round(edit, 3), "hook_offset": round(hook_ev.start - s, 2), "tail": round(tail, 2),
                    "pending_reply": pending, "beat_at_cut": beat, "speakers": len(speakers), "conflict": conflict,
                    "clean_in": s_clean, "clean_out": e_clean, "opens_on_reply": reply_open, "monologue": monologue,
                    "beats": beats, "dialogue_lines": len(dl)})
    w.hook_desc = _hook_desc(hook_ev)
    ck = _cliff_kind(last, beat)
    w.cliff_desc = f"{ck} — {_short(last)}"
    # One specific line built from the script itself (no templates that read the same for every clip).
    who = hook_ev.speaker.title() if hook_ev.kind == "dialogue" else hook_ev.kind
    opener = (f'{who} opens with "{_quote(hook_ev.text)}"' if hook_ev.kind == "dialogue"
              else f"opens on {dict(action='an action beat', sfx='a sound cue').get(hook_ev.kind, 'a shot')} "
                   f"({_quote(hook_ev.text, 7)})")
    ender = f'cuts on {last.speaker.title()}\'s "{_quote(last.text)}"'
    reply = f", before {nxt.speaker.title()} can answer" if pending and nxt else (", on a marked beat" if beat else "")
    w.why = f"{opener[0].upper() + opener[1:]}; {ender}{reply}."
    return w


def episode_windows(ep: Episode) -> list[Window]:
    """All eligible 20-60 s windows for one episode, scored (unsorted)."""
    tev = sorted(ep.timed_events, key=lambda x: (x.start, x.idx))
    starts = [x.start for x in tev]
    inten = {x.eid: event_intensity(x) for x in tev}
    names = _name_tokens(ep)
    cut_in, cut_out = legal_cuts(ep)
    outs = [t for t, _ in cut_out]
    out_clean = dict(cut_out)
    wins = []
    for s, s_clean in cut_in:
        lo = bisect.bisect_left(outs, s + C.MIN_DUR - C.DUR_TOL)
        hi = bisect.bisect_right(outs, s + C.MAX_DUR + C.DUR_TOL)
        for e in outs[lo:hi]:
            w = score_window(ep, s, e, s_clean, out_clean[e], tev, starts, inten, names)
            if w:
                wins.append(w)
    return wins


def top_diverse(wins: list[Window], k: int, overlap: float = C.NMS_OVERLAP) -> list[Window]:
    """Greedy non-maximum suppression: best windows that are not near-duplicates."""
    kept: list[Window] = []
    for w in sorted(wins, key=lambda x: -x.score):
        if all(w.iou(o) < overlap for o in kept):
            kept.append(w)
        if len(kept) >= k:
            break
    return kept


def transcript_lines(ep: Episode, start: float | None = None, end: float | None = None,
                     camera_chars: int = 110) -> list[str]:
    """Compact, ID-tagged transcript used for the LLM prompt and for human review packets."""
    out = []
    for x in sorted(ep.timed_events, key=lambda z: (z.start, z.idx)):
        if start is not None and x.start < start - C.EPS:
            continue
        if end is not None and x.start >= end - C.EPS:
            continue
        txt = x.text if x.kind != "camera" or len(x.text) <= camera_chars else x.text[: camera_chars - 1] + "…"
        if x.kind == "dialogue":
            body = f'{x.speaker}{f" ({x.delivery})" if x.delivery else ""}: "{txt}"'
        else:
            body = f"[{x.kind.upper()}] {txt}"
        out.append(f"{x.eid} | {fmt_ts(x.start)}-{fmt_ts(x.end)} | {body}")
    return out
