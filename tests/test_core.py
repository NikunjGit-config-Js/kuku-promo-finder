"""Run: python -m pytest -q   (uses only the synthetic example show; no Kuku data needed)."""
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from promo_finder import config as C  # noqa: E402
from promo_finder.candidates import episode_windows, legal_cuts, top_diverse  # noqa: E402
from promo_finder.llm_judge import Judge, MockProvider, validate_judgement  # noqa: E402
from promo_finder.parser import load_show, parse_episode, parse_ts  # noqa: E402
from promo_finder.pipeline import run  # noqa: E402
from promo_finder.validate import check_promo  # noqa: E402

SHOW = ROOT / "examples" / "synthetic_show"


@pytest.fixture(scope="module")
def eps():
    episodes, errors = load_show(SHOW)
    return episodes, errors


def test_timestamp_formats():
    assert parse_ts("00:04.80") == 4.8
    assert parse_ts("0:01:02.50") == 62.5
    assert parse_ts("01:02") == 62.0
    assert parse_ts("12.5") == 12.5
    assert parse_ts("1:2:3:4") is None and parse_ts("abc") is None


def test_messy_file_is_parsed_and_problems_reported(eps):
    episodes, errors = eps
    assert len(episodes) == 2
    ep2 = next(e for e in episodes if e.source == "Episode_two.txt")
    assert ep2.number == 2 and any("renumbered to 2" in x for x in errors)       # no number anywhere
    assert any("no Start/End line" in w for w in ep2.warnings)                    # untimed block kept
    assert any("end <= start" in w for w in ep2.warnings)                         # reversed timestamps
    vo = next(e for e in ep2.events if e.text.startswith("No mistake"))
    assert vo.speaker == "MAYA" and vo.delivery == "Cold, Certain"               # (V.O.) flag stripped
    assert all(not (e.end is not None and e.start is not None and e.end <= e.start) or not e.timed
               for e in ep2.events)


def test_event_ids_unique_across_show(eps):
    ids = [e.eid for ep in eps[0] for e in ep.events]
    assert len(ids) == len(set(ids))


def test_no_cut_inside_a_dialogue_line(eps):
    for ep in eps[0]:
        cut_in, cut_out = legal_cuts(ep)
        dl = [e for e in ep.timed_events if e.kind == "dialogue"]
        for t, _ in cut_in + cut_out:
            assert not any(d.start + C.EPS < t < d.end - C.EPS for d in dl)


def test_overlapping_events_not_double_counted(eps):
    ep1 = next(e for e in eps[0] if e.number == 1)
    w = top_diverse(episode_windows(ep1), 1)[0]
    summed = sum(e.end - e.start for e in w.events)
    assert summed > w.duration                 # camera, action and dialogue overlap in time...
    assert C.MIN_DUR <= w.duration <= C.MAX_DUR  # ...but clip length is end - start on the timeline


def test_every_window_obeys_hard_rules(eps):
    for ep in eps[0]:
        for w in episode_windows(ep):
            assert C.MIN_DUR <= w.duration <= C.MAX_DUR
            assert w.hook_event.start <= w.start + C.HOOK_WINDOW + C.EPS


def test_pipeline_outputs_validate(tmp_path):
    meta = run(str(SHOW), str(tmp_path), provider="heuristic", show="test")
    assert meta["validation_failures"] == 0 and meta["promos"] >= 2
    data = json.loads((tmp_path / "promos.json").read_text())
    assert [p["rank"] for p in data["promos"]] == list(range(1, len(data["promos"]) + 1))
    for name in ("promos.csv", "top_promos.md", "episodes.csv", "parse_report.json", "run_meta.json"):
        assert (tmp_path / name).exists()


def test_validator_catches_fabrication(eps, tmp_path):
    run(str(SHOW), str(tmp_path), provider="heuristic", show="test")
    p = json.loads((tmp_path / "promos.json").read_text())["promos"][0]
    by = {e.number: e for e in eps[0]}
    assert check_promo(p, by) == []
    bad = json.loads(json.dumps(p))
    bad["segments"][0]["start_s"] += 0.37                                       # invented timestamp
    assert any("not an event" in x for x in check_promo(bad, by))
    bad = json.loads(json.dumps(p))
    bad["hook"]["line"] = "A line the script never says."                        # invented dialogue
    assert any("does not match source" in x for x in check_promo(bad, by))
    bad = json.loads(json.dumps(p))
    bad["segments"][0]["end_s"] = bad["segments"][0]["start_s"] + 75            # too long
    bad["total_duration_s"] = 75
    assert any("outside" in x for x in check_promo(bad, by))


@pytest.mark.parametrize("mode,expect", [("good", "llm"), ("malformed", "llm_failed_used_heuristic"),
                                         ("fail", "llm_failed_used_heuristic"), ("bad_ids", "llm")])
def test_llm_modes_never_break_the_run(tmp_path, mode, expect):
    meta = run(str(SHOW), str(tmp_path), provider=f"mock:{mode}", show="test")
    assert meta["validation_failures"] == 0
    rows = (tmp_path / "episodes.csv").read_text()
    assert expect in rows
    if mode == "bad_ids":   # hallucinated event IDs are rejected and logged, not used
        rep = json.loads((tmp_path / "parse_report.json").read_text())
        assert any("not an event in the clip" in w for ws in rep["llm_warnings"].values() for w in ws)


def test_judgement_parser_accepts_fenced_json_and_rejects_bad_values(eps):
    ep1 = next(e for e in eps[0] if e.number == 1)
    cands = top_diverse(episode_windows(ep1), 2)
    raw = "```json\n" + json.dumps({"candidates": [
        {"id": "C1", "hook_score": "5", "cliffhanger_score": 4, "standalone_score": 4, "stakes_score": 9,
         "hook_event_id": cands[0].events[0].eid, "cliffhanger_event_id": "E77-001", "verdict": "use"},
        {"id": "C9", "hook_score": 3, "cliffhanger_score": 3, "standalone_score": 3, "stakes_score": 3},
        {"id": "C2", "hook_score": "4", "cliffhanger_score": 4, "standalone_score": 4, "stakes_score": 4,
         "cliffhanger_event_id": "E77-001"}]}) + "\n```"
    res, warns = validate_judgement(raw, ep1, cands)
    assert 0 not in res                                    # stakes_score 9 is out of range: rejected, not clamped
    assert any("[9]" in w for w in warns)
    assert any("C9" in w for w in warns) and any("C2" in w for w in warns)


def test_cache_makes_reruns_free(tmp_path, eps):
    ep1 = next(e for e in eps[0] if e.number == 1)
    cands = top_diverse(episode_windows(ep1), 3)
    j = Judge(MockProvider("good"), tmp_path / "cache", tmp_path / "log.jsonl", "test")
    j.judge(ep1, cands)
    j.judge(ep1, cands)
    assert j.usage["calls"] == 1 and j.usage["cached"] == 1
