#!/usr/bin/env python3
"""Promo finder CLI.

  python run.py find   --input <folder of episode .txt files> --out outputs/run1 [--provider gemini]
  python run.py eval   --input <folder> --run outputs/run1          # structural metrics vs baselines
  python run.py review --input <folder> --run outputs/run1          # blind human-review kit
  python run.py score-review --run outputs/run1                     # after filling review_sheet.csv
"""
import argparse
import json
import os
import sys
from pathlib import Path

sys.path.insert(0, os.path.dirname(os.path.abspath(__file__)))   # works with `python -I run.py` too

from promo_finder import pipeline, evaluate              # noqa: E402
from promo_finder.llm_judge import LLMError             # noqa: E402
from promo_finder.parser import InputError, require_episodes  # noqa: E402


def _read_json(path) -> dict:
    return json.loads(Path(path).read_text(encoding="utf-8"))


def _write_json(path, data) -> None:
    Path(path).write_text(json.dumps(data, indent=2), encoding="utf-8")


def main(argv=None):
    ap = argparse.ArgumentParser(description="Find, rank and check 20-60 s promos in microdrama screenplays.")
    sub = ap.add_subparsers(dest="cmd", required=True)
    f = sub.add_parser("find", help="run the pipeline")
    f.add_argument("--input", required=True, help="folder (searched recursively) or single .txt screenplay")
    f.add_argument("--out", default="outputs/run")
    f.add_argument("--provider", default="heuristic",
                   help="heuristic (free, offline) | gemini | groq | openrouter | ollama | openai | anthropic | mock")
    f.add_argument("--model", default=None, help="override the provider's default model")
    f.add_argument("--show", default=None, help="show title used in prompts (default: input folder name)")
    f.add_argument("--episodes", default=None, help="only these episodes, e.g. 1,4,10-12 (default: all)")
    f.add_argument("--launch-window", type=int, default=None,
                   help="episodes 1..N count as spoiler-safe for the launch batch (default 10; 0 = content-only)")
    ex = sub.add_parser("export-prompts", help="write the exact LLM judge prompts to JSONL (batch / offline use)")
    ex.add_argument("--input", required=True)
    ex.add_argument("--out", required=True)
    ex.add_argument("--show", default=None)
    ex.add_argument("--episodes", default=None)
    for name in ("eval", "review"):
        s = sub.add_parser(name)
        s.add_argument("--input", required=True)
        s.add_argument("--run", required=True, help="output folder of a previous `find` run")
        if name == "review":
            s.add_argument("--episodes", type=int, default=5)
            s.add_argument("--previous", default=None, help="output folder of an earlier version, added as a third arm")
    cp = sub.add_parser("compare", help="old vs new run on the same input (like-for-like detectors)")
    cp.add_argument("--input", required=True)
    cp.add_argument("--old", required=True)
    cp.add_argument("--new", required=True)
    sr = sub.add_parser("score-review")
    sr.add_argument("--run", required=True)
    a = ap.parse_args(argv)
    try:
        return _dispatch(a)
    except (InputError, LLMError) as e:   # wrong input path or a misconfigured provider: fail fast, readably
        print(f"error: {e}", file=sys.stderr)
        return 2


def _dispatch(a) -> int:
    if a.cmd == "export-prompts":
        n = pipeline.export_prompts(a.input, a.out, show=a.show, episodes=a.episodes)
        print(f"{n} prompts -> {a.out}")
        return 0
    if a.cmd == "find":
        meta = pipeline.run(a.input, a.out, provider=a.provider, model=a.model, show=a.show, episodes=a.episodes,
                            launch_window=a.launch_window)
        print(json.dumps({k: meta[k] for k in ("episodes_parsed", "load_errors", "promos", "validation_failures",
                                                "provider", "model", "llm_usage", "seconds")}, indent=2))
        print(f"-> {Path(a.out) / 'top_promos.md'}")
        return 0 if meta["validation_failures"] == 0 else 2
    if a.cmd == "score-review":
        review = Path(a.run) / "review"
        res = evaluate.score_review(review / "review_sheet.csv", review / "review_key.json")
        _write_json(review / "review_results.json", res)
        print(json.dumps(res, indent=2))
        if res.get("warning"):
            print(f"error: {res['warning']}", file=sys.stderr)
            return 2
        return 0
    episodes, _ = require_episodes(a.input)
    if a.cmd == "eval":
        run_dir = Path(a.run)
        best = {}
        for p in sorted(_read_json(run_dir / "promos.json")["promos"], key=lambda p: p["rank"]):
            best.setdefault(p["episodes"][0], p)
        res = evaluate.compare_methods(episodes, best)
        _write_json(run_dir / "eval_structural.json", res)
        print(json.dumps(res, indent=2))
    elif a.cmd == "review":
        run_dir = Path(a.run)
        promos = _read_json(run_dir / "promos.json")["promos"]
        prev = _read_json(Path(a.previous) / "promos.json")["promos"] if a.previous else None
        d = evaluate.make_review_kit(episodes, promos, run_dir / "review", n_episodes=a.episodes, previous=prev)
        print(f"Blind review kit -> {d}/review_packet.md ; fill {d}/review_sheet.csv then run score-review")
    elif a.cmd == "compare":
        res = evaluate.compare_runs(episodes, _read_json(Path(a.old) / "promos.json"),
                                    _read_json(Path(a.new) / "promos.json"))
        _write_json(Path(a.new) / "compare_vs_previous.json", res)
        print(json.dumps(res, indent=2))
    return 0


if __name__ == "__main__":
    sys.exit(main())
