"""Editorial policy: story angle, spoiler tier, and the two rankings.

This layer encodes business choices (what to post first), not clip quality. Every label carries its
evidence and a confidence, because text-only rules are a proxy: an editor makes the final call.
"""
from __future__ import annotations

import re
from collections import Counter

from . import config as C
from .parser import Episode

_ANGLE = {k: set(v.split()) for k, v in C.ANGLES.items()}
_ANGLE["secret_identity"] |= set(C.STATUS_TERMS)
_PAYOFF = [re.compile(p.format(status="|".join(C.STATUS_TERMS)), re.I) for p in C.PAYOFF_PATTERNS]
_TIER = {"green": 0, "amber": 1, "red": 2}


def tag_angle(events) -> tuple[str, list[str]]:
    """Primary story angle by matched words (dialogue, actions, delivery notes). 'conflict' if nothing matches."""
    words = []
    for x in events:
        if x.kind in ("dialogue", "action"):
            words += re.findall(r"[a-z']+", x.text.lower())
        if x.kind == "dialogue" and x.delivery:
            words += re.findall(r"[a-z]+", x.delivery.lower())
    hits = {a: sorted({w for w in words if w in vocab}) for a, vocab in _ANGLE.items()}
    best = max(hits, key=lambda a: len(hits[a]))
    return (best, hits[best][:5]) if hits[best] else ("conflict", [])


def early_vocab(episodes: list[Episode], window: int) -> set[str]:
    """Words spoken in the launch-window episodes (what a new viewer will have heard)."""
    return {w for ep in episodes if ep.number <= window for x in ep.events if x.kind == "dialogue"
            for w in re.findall(r"[a-z']+", x.text.lower())}


def spoiler_tier(ep_num: int, events, window: int, early: set[str], llm_risk: str | None = None) -> dict:
    """Green: inside the launch window. Red: later AND the clip contains payoff wording or a status term never
    heard in the window. Amber: later, nothing found (a tease at most). LLM 'high'/'medium' can only raise it."""
    text = " ".join(x.text for x in events if x.kind == "dialogue")
    hits = [m.group(0) for rx in _PAYOFF for m in rx.finditer(text)]
    evidence = []
    if window and ep_num <= window:
        tier, conf = "green", "medium"
        evidence.append(f"Ep {ep_num} is inside the launch window (Ep 1-{window})")
    elif not window:                                   # position rule disabled: content only
        tier, conf = ("red" if hits else "green"), "low"
        evidence.append(f'payoff wording: "{hits[0]}"' if hits else "no payoff wording found")
    else:
        novel = sorted({t for t in C.STATUS_TERMS if re.search(rf"\b{t}\b", text, re.I) and t not in early})
        if hits or novel:
            tier = "red"
            evidence += [f'payoff wording: "{h}"' for h in hits[:2]]
            evidence += [f'"{t}" is never said in Ep 1-{window}' for t in novel[:2]]
        else:
            tier = "amber"
            evidence.append(f"Ep {ep_num} is after the launch window; no payoff wording found")
        conf = "low"
    if llm_risk in ("medium", "high"):
        llm_tier = "red" if llm_risk == "high" else "amber"
        if _TIER[llm_tier] > _TIER[tier]:
            tier = llm_tier
        evidence.append(f"LLM spoiler risk: {llm_risk}")
        conf = "medium"
    elif llm_risk == "low":
        evidence.append("LLM spoiler risk: low")
    return {"tier": tier, "confidence": conf, "evidence": evidence}


def quality_ranking(promos: list[dict]) -> list[dict]:
    """Unrestricted: best clips across all episodes. Editing effort only breaks near-ties."""
    ordered = sorted(promos, key=lambda p: -(p["score"] + C.EDIT_TIEBREAK * p["features"]["edit_effort"]))
    for i, p in enumerate(ordered, 1):
        p["rank"] = i
    return ordered


def launch_ranking(promos: list[dict], n_episodes: int, window: int) -> list[dict]:
    """First posting batch: never Red, at most LAUNCH_MAX_AMBER Amber (later ones ranked lower), one clip per
    episode, top half by quality only (a shorter batch beats filler), and a small penalty for repeating an angle."""
    chosen, used, ambers, angles = [], set(), 0, Counter()
    scores = sorted(p["score"] for p in promos)
    bar = scores[min(len(scores) - 1, int(len(scores) * C.LAUNCH_MIN_PCTL / 100))] if scores else 0.0
    while len(chosen) < C.LAUNCH_BATCH:
        best, best_val = None, None
        for p in promos:
            ep, tier = p["episodes"][0], p["spoiler"]["tier"]
            if (p in chosen or ep in used or tier == "red" or p["score"] < bar
                    or (tier == "amber" and ambers >= C.LAUNCH_MAX_AMBER)):
                continue
            val = p["score"] + C.EDIT_TIEBREAK * p["features"]["edit_effort"]
            if tier == "amber" and window:
                val -= C.AMBER_DISTANCE_PENALTY * (ep - window) / max(1, n_episodes - window)
            val -= C.ANGLE_REPEAT_PENALTY * angles[p["angle"]]
            if best_val is None or val > best_val:
                best, best_val = p, val
        if best is None:
            break
        chosen.append(best)
        used.add(best["episodes"][0])
        ambers += best["spoiler"]["tier"] == "amber"
        angles[best["angle"]] += 1
        best["launch_score"] = round(best_val, 4)
    for p in promos:
        p.setdefault("launch_rank", None)
    for i, p in enumerate(chosen, 1):
        p["launch_rank"] = i
    return chosen
