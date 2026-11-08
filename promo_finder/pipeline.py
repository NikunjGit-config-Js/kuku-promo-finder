"""End-to-end run: parse -> candidates -> (optional LLM judge) -> rank -> validate -> write outputs."""
from __future__ import annotations

import csv
import json
import platform
import time
from datetime import datetime, timezone
from pathlib import Path

from . import config as C
from .candidates import Window, episode_windows, top_diverse
from .editorial import early_vocab, launch_ranking, quality_ranking, spoiler_tier, tag_angle
from .llm_judge import SYSTEM, Judge, build_prompt, make_provider, prompt_key
from . import __version__
from .parser import Episode, fmt_ts, require_episodes
from .validate import check_promo


def _norm(v: int) -> float:
    return (v - 1) / 4.0


def apply_judgement(w: Window, j: dict, model: str) -> None:
    """Replace heuristic scores with the LLM rubric; keep code-measured editing + heuristic as a prior."""
    heur = w.score
    parts = {"hook": _norm(j["hook_score"]), "cliff": _norm(j["cliffhanger_score"]),
             "standalone": _norm(j["standalone_score"]), "stakes": _norm(j["stakes_score"]), "heuristic": heur}
    w.score = round(sum(C.W_LLM[k] * parts[k] for k in C.W_LLM), 4)
    w.feats.update({"heuristic_score": heur, "llm": {k: j[k] for k in (
        "hook_score", "cliffhanger_score", "standalone_score", "stakes_score", "spoiler_risk", "angle", "verdict")}})
    if j.get("hook_event_id"):
        w.hook_event = next(e for e in w.events if e.eid == j["hook_event_id"])
    if j.get("cliffhanger_event_id"):
        w.cliff_event = next(e for e in w.events if e.eid == j["cliffhanger_event_id"])
    w.hook_desc = j["hook"] or w.hook_desc
    w.cliff_desc = j["cliffhanger"] or w.cliff_desc
    w.why = j["why"] or w.why
    w.method = f"llm:{model}"


def _event_ref(ev, start: float, desc: str) -> dict:
    return {"event_id": ev.eid, "at": fmt_ts(ev.start), "offset_s": round(ev.start - start, 2),
            "speaker": ev.speaker, "delivery": ev.delivery, "kind": ev.kind, "line": ev.text, "description": desc}


def promo_record(w: Window, ep: Episode, n: int) -> dict:
    llm = w.feats.get("llm", {})
    return {
        "promo_id": f"E{ep.number:02d}-P{n}",
        "episodes": [ep.number],
        "segments": [{"order": 1, "episode": ep.number, "start": fmt_ts(w.start), "end": fmt_ts(w.end),
                      "start_s": w.start, "end_s": w.end, "duration_s": w.duration,
                      "first_event_id": w.events[0].eid, "last_event_id": w.events[-1].eid, "source_file": ep.source}],
        "total_duration_s": w.duration,
        "hook": _event_ref(w.hook_event, w.start, w.hook_desc),
        "cliffhanger": _event_ref(w.cliff_event, w.start, w.cliff_desc),
        "why": w.why,
        "score": round(w.score, 4),
        "features": {k: v for k, v in w.feats.items() if k != "llm"},
        "llm": llm or None,
        "angle": llm.get("angle"),
        "spoiler_risk": llm.get("spoiler_risk"),
        "method": w.method,
    }


def select_episodes(episodes: list[Episode], spec: str | None) -> list[Episode]:
    """'1,4,10-12' -> those episode numbers; None/'' -> all."""
    if not spec:
        return episodes
    want = set()
    for part in spec.split(","):
        part = part.strip()
        if "-" in part:
            a, b = part.split("-", 1)
            want |= set(range(int(a), int(b) + 1))
        elif part:
            want.add(int(part))
    return [e for e in episodes if e.number in want]


def export_prompts(input_path: str, out_path: str, show: str | None = None, episodes: str | None = None) -> int:
    """Write the exact judge prompts as JSONL, for a batch API or any chat model; answers come back via
    --provider replay:<file> with records {"key": ..., "response": ...}."""
    eps, _ = require_episodes(input_path)
    show = show or Path(input_path).resolve().name
    n = 0
    with open(out_path, "w", encoding="utf-8") as f:
        for ep in select_episodes(eps, episodes):
            cands = top_diverse(episode_windows(ep), C.CANDIDATES_PER_EPISODE)
            if cands:
                user = build_prompt(ep, cands, show)
                f.write(json.dumps({"key": prompt_key(SYSTEM, user), "episode": ep.number, "candidates": len(cands),
                                    "system": SYSTEM, "user": user}, ensure_ascii=False) + "\n")
                n += 1
    return n


def run(input_path: str, out_dir: str, provider: str = "heuristic", model: str | None = None,
        show: str | None = None, episodes: str | None = None, launch_window: int | None = None) -> dict:
    t0 = time.time()
    all_episodes, load_errors = require_episodes(input_path)   # wrong path / .zip / no .txt -> InputError
    out = Path(out_dir)
    out.mkdir(parents=True, exist_ok=True)
    episodes_sel = select_episodes(all_episodes, episodes)
    by_num = {e.number: e for e in all_episodes}
    show = show or Path(input_path).resolve().name
    window = C.LAUNCH_WINDOW if launch_window is None else launch_window
    early = early_vocab(all_episodes, window)
    judge = None
    if provider != "heuristic":
        p = make_provider(provider, model)
        judge = Judge(p, out / "llm_cache", out / "llm_calls.jsonl", show)

    promos, rejected, ep_rows, llm_warnings = [], [], [], {}
    for ep in episodes_sel:
        wins = episode_windows(ep)
        cands = top_diverse(wins, C.CANDIDATES_PER_EPISODE)
        status = "heuristic"
        if judge and cands:
            res, warns = judge.judge(ep, cands)
            if warns:
                llm_warnings[ep.number] = warns
            if res is None:
                replay_miss = bool(warns) and "no recorded response" in warns[0]
                status = "no_recorded_response_used_heuristic" if replay_miss else "llm_failed_used_heuristic"
            else:
                status = "llm" if len(res) == len(cands) else "llm_partial"
                for i, w in enumerate(cands):
                    if i in res:
                        apply_judgement(w, res[i], judge.p.model)
        cands.sort(key=lambda w: -w.score)
        kept = 0
        for w in cands:
            verdict = w.feats.get("llm", {}).get("verdict", "use")
            reason = None
            llm = w.feats.get("llm")
            weak = ([k for k, v in C.FLOORS_LLM.items() if llm[k] < v] if llm
                    else [k for k, v in C.FLOORS_HEUR.items() if w.feats[k] < v])
            if verdict == "reject":
                reason = "LLM verdict: reject"
            elif weak:
                reason = "below floor on " + ", ".join(weak)
            elif w.score < C.MIN_FINAL_SCORE:
                reason = f"score {w.score:.2f} < {C.MIN_FINAL_SCORE}"
            elif kept >= C.PROMOS_PER_EPISODE:
                reason = f"episode already has {C.PROMOS_PER_EPISODE} better promos"
            if reason:
                r = promo_record(w, ep, 0)
                r["reject_reason"] = reason
                rejected.append(r)
                continue
            kept += 1
            rec = promo_record(w, ep, kept)
            angle, angle_ev = tag_angle(w.events)
            llm = w.feats.get("llm") or {}
            use_llm_angle = llm.get("angle") not in (None, "other")
            rec["angle"] = llm["angle"] if use_llm_angle else angle
            rec["angle_evidence"] = [f"LLM angle: {llm['angle']}"] if use_llm_angle else angle_ev
            rec["spoiler"] = spoiler_tier(ep.number, w.events, window, early, llm.get("spoiler_risk"))
            promos.append(rec)
        ep_rows.append({
            "episode": ep.number, "file": ep.source, "duration": fmt_ts(ep.duration), "events": len(ep.events),
            "timed_events": len(ep.timed_events), "parse_warnings": len(ep.warnings), "eligible_windows": len(wins),
            "candidates": len(cands), "promos": kept, "best_score": round(cands[0].score, 3) if cands else None,
            "status": status if cands else "no eligible window",
        })

    ranked = quality_ranking(promos)
    launch = launch_ranking(ranked, max(by_num) if by_num else 1, window)
    failures = 0
    for p in ranked:
        p["validation_errors"] = check_promo(p, by_num)
        failures += bool(p["validation_errors"])

    meta = {
        "version": __version__,
        "run_at": datetime.now(timezone.utc).isoformat(timespec="seconds"),
        "input": str(Path(input_path).name), "show": show, "files_found": len(all_episodes) + len(load_errors),
        "episodes_parsed": len(all_episodes), "episodes_processed": len(episodes_sel), "load_errors": load_errors,
        "provider": provider, "model": judge.p.model if judge else None,
        "llm_usage": judge.usage if judge else None,
        "promos": len(ranked), "rejected_candidates": len(rejected), "validation_failures": failures,
        "seconds": round(time.time() - t0, 2), "python": platform.python_version(),
        "config": {k: getattr(C, k) for k in ("MIN_DUR", "MAX_DUR", "HOOK_WINDOW", "MIN_DIALOGUE_LINES",
                                               "MAX_DEAD_TAIL", "CANDIDATES_PER_EPISODE", "PROMOS_PER_EPISODE",
                                               "MIN_FINAL_SCORE", "FLOORS_HEUR", "FLOORS_LLM", "LAUNCH_BATCH", "LAUNCH_MIN_PCTL",
                                               "LAUNCH_MAX_AMBER", "AMBER_DISTANCE_PENALTY", "ANGLE_REPEAT_PENALTY",
                                               "EDIT_TIEBREAK", "W_HEUR", "W_LLM")} | {"launch_window": window},
    }
    posting = (f"Two lists. QUALITY ranks every promo across all episodes, unrestricted. LAUNCH is the first batch "
               f"to post: up to {C.LAUNCH_BATCH} clips from the top half by quality, one per episode, never a Red spoiler, at most "
               f"{C.LAUNCH_MAX_AMBER} Amber teases (later ones ranked lower), and a small push toward different story "
               f"angles. Green = Ep 1-{window}; this window is an editorial assumption to set per show "
               f"(--launch-window), not an industry rule. Spoiler and angle labels are text-based proxies with "
               f"evidence attached; an editor approves every clip before posting.")
    meta["launch_batch_size"] = len(launch)
    _write(out, meta, posting, ranked, rejected, ep_rows, all_episodes, llm_warnings)
    return meta


def _write(out: Path, meta, posting, ranked, rejected, ep_rows, episodes, llm_warnings):
    # Always UTF-8: the OS default (cp1252 on most Windows PCs) cannot hold every script character, e.g. "₹".
    (out / "promos.json").write_text(json.dumps({"run": meta, "posting_plan": posting, "promos": ranked},
                                                indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "rejected_candidates.json").write_text(json.dumps(rejected, indent=2, ensure_ascii=False), encoding="utf-8")
    (out / "run_meta.json").write_text(json.dumps(meta, indent=2), encoding="utf-8")
    (out / "parse_report.json").write_text(json.dumps(
        {"load_errors": meta["load_errors"], "episodes": {e.number: e.warnings for e in episodes if e.warnings},
         "llm_warnings": llm_warnings}, indent=2), encoding="utf-8")
    with open(out / "episodes.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.DictWriter(f, fieldnames=list(ep_rows[0].keys()) if ep_rows else ["episode"])
        wr.writeheader()
        wr.writerows(ep_rows)
    cols = ["rank", "launch_rank", "promo_id", "episode", "start", "end", "duration_s", "hook_at", "hook_line",
            "cliffhanger_line", "why", "score", "edit_effort", "spoiler_tier", "spoiler_confidence", "spoiler_evidence",
            "angle", "angle_evidence", "method", "validation_errors"]
    with open(out / "promos.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(cols)
        for p in ranked:
            s = p["segments"][0]
            wr.writerow([p["rank"], p["launch_rank"] or "", p["promo_id"], s["episode"], s["start"], s["end"],
                         p["total_duration_s"], p["hook"]["at"], _line(p["hook"]), _line(p["cliffhanger"]), p["why"],
                         p["score"], p["features"]["edit_effort"], p["spoiler"]["tier"], p["spoiler"]["confidence"],
                         "; ".join(p["spoiler"]["evidence"]), p["angle"], " ".join(p["angle_evidence"]), p["method"],
                         "; ".join(p["validation_errors"])])
    head = ["| # | Ep | Start–End (dur) | Hook (first 3 s) | Cliffhanger (cut) | Why | Score | Spoiler | Angle |",
            "|---|---|---|---|---|---|---|---|---|"]

    def row(p, n):
        s = p["segments"][0]
        return (f"| {n} | {s['episode']} | {s['start']}–{s['end']} ({p['total_duration_s']:.1f}s) | "
                f"{_line(p['hook'], 70)} | {_line(p['cliffhanger'], 70)} | {p['why']} | {p['score']:.2f} | "
                f"{p['spoiler']['tier']} | {p['angle']} |")
    launch = sorted([p for p in ranked if p["launch_rank"]], key=lambda p: p["launch_rank"])
    model_note = f" / `{meta['model']}`" if meta["model"] else ""   # no nested same-quote f-string: Python 3.10+
    md = [f"# Promos — {meta['show']}", "",
          f"Run {meta['run_at']} · provider `{meta['provider']}`{model_note} · "
          f"{meta['episodes_processed']}/{meta['episodes_parsed']} episodes · {meta['promos']} promos · "
          f"validation failures: {meta['validation_failures']}", "", f"**How to read this.** {posting}", "",
          "## Launch batch (post first)", ""] + head + [row(p, p["launch_rank"]) for p in launch] + [
          "", "## Quality ranking (all episodes, top 15)", ""] + head + [row(p, p["rank"]) for p in ranked[:15]]
    (out / "top_promos.md").write_text("\n".join(md) + "\n", encoding="utf-8")


def _line(ref: dict, n: int = 160) -> str:
    t = ref["line"] if len(ref["line"]) <= n else ref["line"][: n - 1] + "…"
    who = f"{ref['speaker']}: " if ref.get("speaker") else f"[{ref['kind'].upper()}] "
    return (who + t).replace("|", "/")
