"""Portability regressions found in the final release gate (v5.1.1). Uses only the synthetic show.

Each test reproduces a defect that shipped in v5.1:
1. pipeline.py used a Python 3.12-only f-string, so the tool did not even import on Python 3.10/3.11.
2. JSON outputs used the OS default encoding (cp1252 on most Windows PCs): a script line with a character
   outside that code page (for example the rupee sign or Hindi text) crashed the run at the final write,
   and the other JSON files came out as non-UTF-8.
3. The review scorer silently skipped every row of a sheet saved by Excel as 'CSV UTF-8' (BOM) and crashed
   on Excel's ANSI 'CSV (Comma delimited)'; a blank 'would post' answer was counted as 'no'.
4. A wrong --input (Kuku's .zip, a typo, an empty folder) exited 0. The .zip is stored uncompressed, so it was
   parsed as one giant 'episode' and produced plausible-looking promos that mixed episodes.
"""
import csv
import io
import json
import os
import subprocess
import sys
import tokenize
import py_compile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

import run as cli  # noqa: E402
from promo_finder.evaluate import SHEET_COLS, make_review_kit, score_review  # noqa: E402
from promo_finder.parser import load_show  # noqa: E402
from promo_finder.pipeline import run  # noqa: E402

SHOW = ROOT / "examples" / "synthetic_show"
SOURCES = sorted([*ROOT.glob("promo_finder/*.py"), ROOT / "run.py", *ROOT.glob("tests/*.py")])


# 1 ------------------------------------------------------------------------------------------ Python 3.10+
def test_sources_stay_compatible_with_python_310():
    for f in SOURCES:
        py_compile.compile(str(f), doraise=True)
    if sys.version_info < (3, 12):          # compiling above already proves it on this interpreter
        return
    offenders = []                          # Python 3.12+ accepts these; 3.10 and 3.11 reject the whole file
    for f in SOURCES:
        quotes = []
        for tok in tokenize.generate_tokens(io.StringIO(f.read_text(encoding="utf-8")).readline):
            if tok.type == tokenize.FSTRING_START:
                q = tok.string.lstrip("rRbBfF")[:1]
                if q in quotes:
                    offenders.append(f"{f.name}:{tok.start[0]}")
                quotes.append(q)
            elif tok.type == tokenize.FSTRING_END:
                quotes.pop()
            elif tok.type == tokenize.STRING and quotes and tok.string.lstrip("rRbBuU")[:1] in quotes:
                offenders.append(f"{f.name}:{tok.start[0]}")
    assert not offenders, f"f-string reuses its own quote inside {{...}} (Python 3.12+ only): {offenders}"


# 2 ------------------------------------------------------------------------------- UTF-8 on any system
def _non_utf8_env() -> dict:
    """ASCII on Linux/macOS; the ANSI code page (usually cp1252) on Windows, where LC_ALL is ignored."""
    env = {k: v for k, v in os.environ.items() if k not in ("PYTHONUTF8", "PYTHONIOENCODING", "LC_ALL", "LC_CTYPE",
                                                             "LANG")}
    env.update(PYTHONUTF8="0", PYTHONCOERCECLOCALE="0", LC_ALL="C", LANG="C")
    return env


def _cli(args, env):
    return subprocess.run([sys.executable, str(ROOT / "run.py"), *args], capture_output=True, text=True,
                          encoding="utf-8", errors="replace", env=env, timeout=180)


def _rupee_show(tmp_path: Path) -> Path:
    """The synthetic show with a rupee sign (not in cp1252 or ASCII) at the start of every spoken line."""
    d = tmp_path / "show"
    d.mkdir()
    for f in SHOW.glob("*.txt"):
        lines = f.read_text(encoding="utf-8").splitlines()
        lines = ['"₹ ' + x.lstrip()[1:] if x.lstrip().startswith('"') else x for x in lines]
        (d / f.name).write_text("\n".join(lines) + "\n", encoding="utf-8")
    return d


def test_every_command_writes_utf8_whatever_the_system_encoding(tmp_path):
    show, out, env = _rupee_show(tmp_path), tmp_path / "out", _non_utf8_env()
    for args in (["find", "--input", str(show), "--out", str(out)],
                 ["eval", "--input", str(show), "--run", str(out)],
                 ["review", "--input", str(show), "--run", str(out), "--episodes", "2"],
                 ["compare", "--input", str(show), "--old", str(out), "--new", str(out)]):
        r = _cli(args, env)
        assert r.returncode == 0, (args[0], r.stderr[-1500:])
    ids = list(json.loads((out / "review" / "review_key.json").read_text(encoding="utf-8")))
    with open(out / "review" / "review_sheet.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(SHEET_COLS)
        wr.writerows([[cid, 4, 3, 4, "yes", "₹ note"] for cid in ids])
    r = _cli(["score-review", "--run", str(out)], env)
    assert r.returncode == 0, r.stderr[-1500:]
    for f in out.rglob("*"):
        if f.is_file():
            f.read_bytes().decode("utf-8")          # raises if any output is not UTF-8
    text = (out / "promos.json").read_text(encoding="utf-8")
    assert "₹" in text and json.loads(text)["promos"]
    assert json.loads((out / "compare_vs_previous.json").read_text(encoding="utf-8"))["same_recommendations"] is True


def test_compare_flags_a_changed_recommendation(tmp_path):
    from promo_finder.evaluate import compare_runs
    run(str(SHOW), str(tmp_path / "a"), show="t")
    a = json.loads((tmp_path / "a" / "promos.json").read_text(encoding="utf-8"))
    b = json.loads(json.dumps(a))
    b["promos"][0]["score"] += 0.0001            # a 4th-decimal score change is not a different recommendation
    eps, _ = load_show(SHOW)
    assert compare_runs(eps, a, b)["same_recommendations"] is True
    b["promos"][0]["rank"], b["promos"][-1]["rank"] = b["promos"][-1]["rank"], b["promos"][0]["rank"]
    assert compare_runs(eps, a, b)["same_recommendations"] is False


# 3 ------------------------------------------------------------------------- review sheet from Excel
def _kit(tmp_path: Path) -> Path:
    run(str(SHOW), str(tmp_path / "run"), show="t")
    promos = json.loads((tmp_path / "run" / "promos.json").read_text(encoding="utf-8"))["promos"]
    eps, _ = load_show(SHOW)
    return make_review_kit(eps, promos, tmp_path / "rev", n_episodes=2)


@pytest.mark.parametrize("encoding, header, post, note", [
    ("utf-8-sig", SHEET_COLS, "Yes", "ok"),                         # Excel: Save as 'CSV UTF-8' (adds a BOM)
    ("cp1252", SHEET_COLS, "No", "café — cut late"),      # Excel: 'CSV (Comma delimited)' on Windows
    ("utf-8", ["clip_id", "hook_1to5", "cliffhanger_1to5", "makes_sense_1to5", "would_post_y_n", "notes"],
     "y", ""),                                                      # a sheet from the v5.1 kit
])
def test_review_sheet_saved_by_excel_is_scored(tmp_path, encoding, header, post, note):
    d = _kit(tmp_path)
    ids = list(json.loads((d / "review_key.json").read_text(encoding="utf-8")))
    with open(d / "review_sheet.csv", "w", newline="", encoding=encoding) as f:
        wr = csv.writer(f)
        wr.writerow(header)
        wr.writerows([[cid, 4, 3, 5, post, note] for cid in ids])
    res = score_review(d / "review_sheet.csv", d / "review_key.json")
    assert res["rated_clips"] == len(ids) and res["skipped_rows"] == 0
    want = 1.0 if post.lower().startswith("y") else 0.0
    assert all(m["would_post_rate"] == want for m in res["per_method"].values())


def test_blank_would_post_is_unanswered_and_an_unusable_sheet_says_why(tmp_path):
    key = {"R01": {"method": "current", "episode": 1}, "R02": {"method": "current", "episode": 2}}
    (tmp_path / "key.json").write_text(json.dumps(key), encoding="utf-8")
    with open(tmp_path / "s.csv", "w", newline="", encoding="utf-8") as f:
        wr = csv.writer(f)
        wr.writerow(SHEET_COLS)
        wr.writerows([["R01", 4, 4, 4, "yes", ""], ["R02", 3, 3, 3, "", ""]])
    m = score_review(tmp_path / "s.csv", tmp_path / "key.json")["per_method"]["current"]
    assert m["clips"] == 2 and m["would_post_answered"] == 1 and m["would_post_rate"] == 1.0
    (tmp_path / "semi.csv").write_text("clip_id;hook_1to5\nR01;4\n", encoding="utf-8")
    res = score_review(tmp_path / "semi.csv", tmp_path / "key.json")
    assert res["rated_clips"] == 0 and "clip_id" in res["warning"]


# 4 --------------------------------------------------------------------------------- wrong --input
def test_wrong_input_fails_loudly(tmp_path, capsys):
    zpath = tmp_path / "pack.zip"
    with zipfile.ZipFile(zpath, "w", compression=zipfile.ZIP_STORED) as z:   # Kuku's pack is stored like this
        for f in SHOW.glob("*.txt"):
            z.write(f, f"show/{f.name}")
    empty = tmp_path / "empty"
    empty.mkdir()
    for bad in (zpath, tmp_path / "missing", empty):
        assert cli.main(["find", "--input", str(bad), "--out", str(tmp_path / "o")]) == 2
        assert cli.main(["eval", "--input", str(bad), "--run", str(tmp_path / "o")]) == 2
    err = capsys.readouterr().err
    assert "extract" in err and "not found" in err and "no .txt" in err
    assert not (tmp_path / "o" / "promos.json").exists()


# 5 ------------------------------------------------------------------------- exact 20-60 s bounds
_EDGE_SCRIPT = """EPISODE NUMBER: Episode 1
SCENE 01
Scene heading: INT. HALL - NIGHT
Scene start: 00:00.00
Scene end: 01:05.00
"""
_EDGE_LINES = [("A (Furious)", 0.0, 3.0, "You lied to me!"), ("B (Shocked)", 4.0, 8.0, "I never lied to you."),
               ("A (Furious)", 9.0, 14.0, "Then why is my ring on her finger?"),
               ("B (Cold)", 15.0, 19.95, "Because you were never his wife!"),
               ("A (Shocked)", 25.0, 30.0, "Who told you that?"), ("B (Cruel)", 35.0, 40.0, "Your own mother did."),
               ("A (Sobbing)", 45.0, 50.0, "She would never betray me!"),
               ("B (Mocking)", 55.0, 60.05, "Then ask her at the gallows!")]


def test_durations_are_exactly_20_to_60_seconds(tmp_path):
    """Regression: a +/-0.05 s timestamp tolerance was also applied to the 20-60 s rule, so 19.95 s and
    60.05 s clips could be generated and passed validation (one 19.95 s candidate existed in Kuku's pack)."""
    from promo_finder.candidates import episode_windows
    from promo_finder.parser import fmt_ts, parse_episode
    from promo_finder.validate import check_promo
    body = "".join(f'\n**{who}:**\nStart: ({fmt_ts(a)}) | End: ({fmt_ts(b)})\n"{line}"\n' for who, a, b, line in _EDGE_LINES)
    (tmp_path / "ep.txt").write_text(_EDGE_SCRIPT + body, encoding="utf-8")
    ep = parse_episode(tmp_path / "ep.txt")
    durations = sorted({w.duration for w in episode_windows(ep)})
    assert durations and min(durations) >= 20.0 and max(durations) <= 60.0, durations
    promo = {"segments": [{"episode": 1, "start_s": 0.0, "end_s": 19.95}], "total_duration_s": 19.95,
             "hook": {"event_id": ep.events[0].eid, "offset_s": 0.0}, "cliffhanger": {"event_id": ep.events[3].eid}}
    assert any("total duration" in e for e in check_promo(promo, {1: ep}))
