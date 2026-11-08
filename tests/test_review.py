"""Blind review kit: arms, blinding, scoring. Ratings below are test fixtures, not human scores."""
import csv
import json
import sys
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from promo_finder.evaluate import make_review_kit, score_review  # noqa: E402
from promo_finder.parser import load_show  # noqa: E402
from promo_finder.pipeline import run  # noqa: E402

SHOW = ROOT / "examples" / "synthetic_show"


def test_kit_is_blind_and_has_three_arms(tmp_path):
    run(str(SHOW), str(tmp_path / "cur"), show="t")
    cur = json.loads((tmp_path / "cur" / "promos.json").read_text())["promos"]
    prev = json.loads((tmp_path / "cur" / "rejected_candidates.json").read_text())   # different, valid windows
    for i, p in enumerate(prev, 1):
        p["rank"] = i
    eps, _ = load_show(SHOW)
    d = make_review_kit(eps, cur, tmp_path / "rev", n_episodes=2, previous=prev)
    key = json.loads((d / "review_key.json").read_text())
    methods = {v["method"] for v in key.values()}
    assert methods <= {"current", "previous", "episode_end"} and len(methods) >= 2
    packet = (d / "review_packet.md").read_text().lower()
    body = packet.split("## r01", 1)[1]                      # clip text only, after the instructions
    assert all(w not in body for w in ("previous", "episode_end", "baseline", "current version"))
    with open(d / "review_sheet.csv") as f:
        assert next(csv.reader(f))[:4] == ["clip_id", "hook_1to5", "cliffhanger_1to5", "standalone_1to5"]


def test_scoring_head_to_head_and_bad_rows(tmp_path):
    key = {"R01": {"method": "current", "episode": 1}, "R02": {"method": "previous", "episode": 1},
           "R03": {"method": "current", "episode": 2}, "R04": {"method": "previous", "episode": 2},
           "R05": {"method": "episode_end", "episode": 2}}
    (tmp_path / "key.json").write_text(json.dumps(key))
    rows = [["R01", 5, 5, 4, "y", ""], ["R02", 3, 2, 4, "n", ""], ["R03", 4, 4, 4, "y", ""],
            ["R04", 4, 3, 3, "n", ""], ["R05", 9, 1, 1, "n", "out of range"], ["R06", "", "", "", "", ""]]
    with open(tmp_path / "sheet.csv", "w", newline="") as f:
        wr = csv.writer(f)
        wr.writerow(["clip_id", "hook_1to5", "cliffhanger_1to5", "makes_sense_1to5", "would_post_y_n", "notes"])
        wr.writerows(rows)
    res = score_review(tmp_path / "sheet.csv", tmp_path / "key.json")
    assert res["rated_clips"] == 4 and res["skipped_rows"] == 2
    h = res["head_to_head"]["current_vs_previous"]
    assert (h["wins"], h["losses"]) == (2, 0) and h["verdict"] == "current better"
    assert res["per_method"]["current"]["would_post_rate"] == 1.0
