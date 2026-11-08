"""Baselines, objective structural checks, and a blind human-review kit.

Why a human review: the heuristic optimises its own features, so scoring its picks with those same
features would be circular. Structural checks (length, cuts, dead air) are objective; storytelling
quality is judged blind by a person against two simple baselines.
"""
from __future__ import annotations

import csv
import io
import json
import random
from pathlib import Path

from . import config as C
from .candidates import Window, episode_windows, legal_cuts, top_diverse, transcript_lines
from .parser import Episode, fmt_ts
from .validate import check_segment


def baseline_window(ep: Episode, kind: str, rng: random.Random) -> Window | None:
    cut_in, cut_out = legal_cuts(ep)
    ins, outs = [t for t, _ in cut_in], [t for t, _ in cut_out]
    pairs = [(s, e) for s in ins for e in outs if C.MIN_DUR - C.DUR_TOL <= e - s <= C.MAX_DUR + C.DUR_TOL]
    if not pairs:
        return None
    if kind == "episode_end":       # what an editor might grab first: the episode's own cliffhanger
        e = max(e for _, e in pairs)
        s = min((s for s, e2 in pairs if e2 == e), key=lambda s: abs((e - s) - 40))
    elif kind == "episode_start":
        s = min(s for s, _ in pairs)
        e = min((e for s2, e in pairs if s2 == s), key=lambda e: abs((e - s) - 40))
    elif kind == "random":
        s, e = rng.choice(pairs)
    else:
        raise ValueError(kind)
    evs = [x for x in sorted(ep.timed_events, key=lambda z: (z.start, z.idx)) if s - C.EPS <= x.start < e - C.EPS]
    return Window(ep.number, s, e, evs, method=kind)


def structural(ep: Episode, w: Window) -> dict:
    dl = [x for x in w.events if x.kind == "dialogue"]
    errs = check_segment(ep, w.start, w.end)
    return {
        "duration_ok": C.MIN_DUR - C.DUR_TOL <= w.end - w.start <= C.MAX_DUR + C.DUR_TOL,
        "no_mid_line_cut": not any("cuts through dialogue" in x for x in errs),
        "boundaries_from_source": not any("not an event" in x for x in errs),
        "speech_in_first_3s": any(x.start <= w.start + C.HOOK_WINDOW + C.EPS for x in dl),
        "ends_within_2s_of_a_line": bool(dl) and (w.end - dl[-1].end) <= 2.0 + C.EPS,
        "dialogue_lines": len(dl),
    }


def compare_methods(episodes: list[Episode], system_promos: dict[int, dict], seed: int = 7) -> dict:
    """Objective structural metrics for system picks vs baselines, one clip per episode per method."""
    rng = random.Random(seed)
    rows = {m: [] for m in ("system", "episode_end", "episode_start", "random")}
    for ep in episodes:
        p = system_promos.get(ep.number)
        if p:
            seg = p["segments"][0]
            evs = [x for x in ep.timed_events if seg["start_s"] - C.EPS <= x.start < seg["end_s"] - C.EPS]
            rows["system"].append(structural(ep, Window(ep.number, seg["start_s"], seg["end_s"], evs)))
        for m in ("episode_end", "episode_start", "random"):
            w = baseline_window(ep, m, rng)
            if w:
                rows[m].append(structural(ep, w))
    summary = {}
    for m, rs in rows.items():
        n = len(rs)
        summary[m] = {"clips": n, **{k: round(sum(r[k] for r in rs) / n, 3) for k in rs[0]}} if n else {"clips": 0}
    return summary


# ------------------------------------------------------------------ blind human review kit
ARMS = ("current", "previous", "episode_end")
SHEET_COLS = ["clip_id", "hook_1to5", "cliffhanger_1to5", "standalone_1to5", "would_post_yes_no", "notes"]
_OLD_COLS = {"makes_sense_1to5": "standalone_1to5", "would_post_y_n": "would_post_yes_no"}   # v5.1 kits
_SCORES = ("hook_1to5", "cliffhanger_1to5", "standalone_1to5")
INSTRUCTIONS = """# Blind promo review (about 15-25 minutes)

You will read {n} short clips, as scripts. Some were picked by the current tool, some by the previous version, and
some are simply the last ~40 s of an episode. The method is hidden; **do not open review_key.json until you finish.**

How to rate: read each clip once, top to bottom, imagining it appearing in your feed with no knowledge of the show.
Fill one row of review_sheet.csv per clip, straight away. Do not go back and change earlier answers. Allow about a minute per clip.

| Column | 1 | 3 | 5 |
|---|---|---|---|
| hook_1to5: the first 3 seconds | nothing happens yet | mildly interesting | I would stop scrolling |
| cliffhanger_1to5: the ending | the scene feels finished | a little curious | I need the next scene |
| standalone_1to5: as a stranger | I am lost | I get the gist | clear who, what and what is at stake |
| would_post_yes_no | yes or no: would you spend ad money on this clip as it is? | | |

Excel is fine: keep the file as CSV when saving ("CSV UTF-8" or "CSV (Comma delimited)").

Lines show seconds from the clip's start, then the speaker and their delivery note. [shot] lines describe the camera,
and are shown only for the first and last 3 seconds. Then run: `python run.py score-review --run <run folder>`.
"""


def make_review_kit(episodes: list[Episode], promos: list[dict], out_dir: Path, n_episodes: int = 5,
                    seed: int = 11, previous: list[dict] | None = None) -> Path:
    """For each sampled episode: the current version's best clip, the previous version's best clip (if it differs)
    and an episode-ending baseline. Clips are shuffled and the method is hidden in review_key.json."""
    rng = random.Random(seed)
    by_num = {e.number: e for e in episodes}

    def best_of(ps):
        best = {}
        for p in sorted(ps, key=lambda p: p["rank"]):
            best.setdefault(p["episodes"][0], p)
        return best

    def win(p):
        return p["segments"][0]["start_s"], p["segments"][0]["end_s"]
    cur, prev = best_of(promos), best_of(previous or [])
    pool = sorted(n for n in cur if n in prev and win(prev[n]) != win(cur[n])) if prev else sorted(cur)
    if len(pool) < n_episodes:
        pool = sorted(cur)
    # Stratified: one episode from each equal slice of the series, so early and late episodes are both rated.
    k = min(n_episodes, len(pool))
    strata = [pool[i * len(pool) // k:(i + 1) * len(pool) // k] for i in range(k)]
    chosen = sorted(rng.choice(st) for st in strata if st)
    clips = []
    for n in chosen:
        arms = [("current", win(cur[n]))] + ([("previous", win(prev[n]))] if n in prev else [])
        w = baseline_window(by_num[n], "episode_end", rng)
        if w:
            arms.append(("episode_end", (w.start, w.end)))
        seen = set()
        for m, (s, e) in arms:
            k = (round(s, 2), round(e, 2))
            if k not in seen:                    # an identical clip is shown once
                seen.add(k)
                clips.append((m, n, s, e))
    rng.shuffle(clips)
    out_dir.mkdir(parents=True, exist_ok=True)
    key, md = {}, [INSTRUCTIONS.format(n=len(clips))]
    for i, (m, n, s, e) in enumerate(clips, 1):
        cid = f"R{i:02d}"
        key[cid] = {"method": m, "episode": n, "start": s, "end": e}
        md += [f"## {cid} — {e - s:.0f} s", "```"]
        for x in sorted(by_num[n].timed_events, key=lambda z: (z.start, z.idx)):
            if not (s - C.EPS <= x.start < e - C.EPS):
                continue
            off = x.start - s
            if x.kind == "camera" and C.HOOK_WINDOW < off < (e - s) - C.HOOK_WINDOW:
                continue
            if x.kind == "dialogue":
                body = f'{x.speaker}{f" ({x.delivery})" if x.delivery else ""}: "{x.text}"'
            else:
                body = f"[{'shot' if x.kind == 'camera' else x.kind}] {x.text}"
            md.append(f"+{off:4.1f}  {body}")
        md += ["```", ""]
    (out_dir / "review_packet.md").write_text("\n".join(md), encoding="utf-8")
    (out_dir / "review_key.json").write_text(json.dumps(key, indent=2), encoding="utf-8")
    with open(out_dir / "review_sheet.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(SHEET_COLS)
        for cid in key:
            wr.writerow([cid, "", "", "", "", ""])
    return out_dir


def _read_sheet(sheet: Path) -> list[dict]:
    """Rows of the rating sheet, accepting what Excel writes on Windows: 'CSV UTF-8' (with a BOM) or
    'CSV (Comma delimited)' (the ANSI code page), and the column names of v5.1 kits."""
    raw = Path(sheet).read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("cp1252", errors="replace")
    rows = []
    for r in csv.DictReader(io.StringIO(text, newline="")):
        row = {}
        for k, v in r.items():
            if isinstance(k, str):
                k = k.strip().lower()
                row[_OLD_COLS.get(k, k)] = v.strip() if isinstance(v, str) else ""
        rows.append(row)
    return rows


def _yes_no(v: str) -> bool | None:
    v = v.strip().lower()
    return True if v in ("y", "yes", "true", "1") else False if v in ("n", "no", "false", "0") else None


def score_review(sheet: Path, key_path: Path) -> dict:
    """Per-method averages and per-episode head-to-heads (sum of the three 1-5 scores). Rows that are blank or
    out of range are skipped and counted. A blank would-post answer is 'not answered', not 'no'.
    Small samples: read the verdicts, not the decimals."""
    key = json.loads(Path(key_path).read_text(encoding="utf-8"))
    rated, skipped = {}, 0
    for r in _read_sheet(sheet):
        k = key.get(r.get("clip_id", ""))
        try:
            vals = [float(r[c]) for c in _SCORES]
        except (TypeError, ValueError, KeyError):
            skipped += 1
            continue
        if not k or not all(1 <= v <= 5 for v in vals):
            skipped += 1
            continue
        rated[r["clip_id"]] = {**k, "hook": vals[0], "cliff": vals[1], "standalone": vals[2],
                               "total": sum(vals), "post": _yes_no(r.get("would_post_yes_no", ""))}
    per = {}
    for m in ARMS:
        rs = [v for v in rated.values() if v["method"] == m]
        if rs:
            answered = [v["post"] for v in rs if v["post"] is not None]
            per[m] = {"clips": len(rs), **{c: round(sum(v[c] for v in rs) / len(rs), 2)
                                           for c in ("hook", "cliff", "standalone", "total")},
                      "would_post_rate": round(sum(answered) / len(answered), 2) if answered else None,
                      "would_post_answered": len(answered)}
    by_ep = {}
    for v in rated.values():
        by_ep.setdefault(v["episode"], {})[v["method"]] = v["total"]
    pairs = {}
    for other in ("previous", "episode_end"):
        d = [x["current"] - x[other] for x in by_ep.values() if "current" in x and other in x]
        if not d:
            continue
        w, l = sum(x > 0 for x in d), sum(x < 0 for x in d)
        mean = sum(d) / len(d)
        verdict = ("current better" if w >= 0.8 * len(d) and mean >= 1.5 else
                   f"{other} better" if l >= 0.8 * len(d) and mean <= -1.5 else "no clear difference")
        pairs[f"current_vs_{other}"] = {"episodes": len(d), "wins": w, "ties": len(d) - w - l, "losses": l,
                                        "mean_total_diff": round(mean, 2), "verdict": verdict}
    res = {"per_method": per, "head_to_head": pairs, "rated_clips": len(rated), "skipped_rows": skipped,
           "note": "One rater, few clips: treat as directional evidence, not a measurement of audience response."}
    if not rated:
        res["warning"] = (f"no row could be scored; the sheet needs the columns {', '.join(SHEET_COLS[:5])} "
                          "with 1-5 scores, saved as comma-separated CSV")
    return res


# ------------------------------------------------------------------ old vs new run on the same input
def compare_runs(episodes: list[Episode], old: dict, new: dict, window: int = C.LAUNCH_WINDOW) -> dict:
    """Same detectors applied to both runs' first posting batches, so the comparison is like-for-like.
    These are structural/policy measurements, not proof of creative quality."""
    from collections import Counter
    from .candidates import _is_reply, _name_tokens
    from .editorial import early_vocab, spoiler_tier, tag_angle
    by = {e.number: e for e in episodes}
    early = early_vocab(episodes, window)

    def batch(run):
        ps = run["promos"]
        if any("launch_rank" in p for p in ps):
            return sorted([p for p in ps if p.get("launch_rank")], key=lambda p: p["launch_rank"])
        return [p for p in ps if p.get("post_batch") == "first"]

    def describe(run):
        rows = []
        for p in batch(run):
            s = p["segments"][0]
            ep = by[s["episode"]]
            tev = sorted(ep.timed_events, key=lambda z: (z.start, z.idx))
            evs = [x for x in tev if s["start_s"] - C.EPS <= x.start < s["end_s"] - C.EPS]
            dl = [x for x in evs if x.kind == "dialogue"]
            prev = next((x for x in reversed(tev) if x.kind == "dialogue" and x.end <= s["start_s"] + C.EPS), None)
            rows.append({"episode": ep.number, "tier": spoiler_tier(ep.number, evs, window, early)["tier"],
                         "angle": tag_angle(evs)[0],
                         "opens_on_reply": bool(dl) and _is_reply(dl[0], prev, s["start_s"], _name_tokens(ep)),
                         "hook_after_1s": p["hook"]["offset_s"] > 1.0, "duration": p["total_duration_s"]})
        n = len(rows) or 1
        return {"batch_size": len(rows), "episodes": [r["episode"] for r in rows],
                "tiers": dict(Counter(r["tier"] for r in rows)), "angles": dict(Counter(r["angle"] for r in rows)),
                "distinct_angles": len({r["angle"] for r in rows}),
                "mean_episode": round(sum(r["episode"] for r in rows) / n, 1),
                "opens_on_reply": sum(r["opens_on_reply"] for r in rows),
                "hook_after_1s": sum(r["hook_after_1s"] for r in rows),
                "mean_duration_s": round(sum(r["duration"] for r in rows) / n, 1)}

    def overall(run):
        m = run["run"]
        return {"promos": m["promos"], "episodes_covered": len({p["episodes"][0] for p in run["promos"]}),
                "validation_failures": m["validation_failures"], "seconds": m["seconds"]}
    key = lambda p: (p["segments"][0]["episode"], p["segments"][0]["start_s"], p["segments"][0]["end_s"])

    def decisions(run):   # what an editor acts on; scores may differ in the 4th decimal across Python versions
        return [(p["promo_id"], *key(p), p.get("rank"), p.get("launch_rank")) for p in run["promos"]]
    return {"same_recommendations": decisions(old) == decisions(new),
            "window_used_for_both": window,
            "old": {**overall(old), "first_batch": describe(old)},
            "new": {**overall(new), "first_batch": describe(new)},
            "first_batch_overlap": len({key(p) for p in batch(old)} & {key(p) for p in batch(new)})}
