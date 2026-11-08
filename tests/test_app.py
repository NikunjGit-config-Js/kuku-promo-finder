"""Web app (app.py): upload handling, parity with the command line, and a headless run of the page.
Uses only the synthetic show. The page test is skipped if Streamlit is not installed."""
import io
import json
import sys
import zipfile
from pathlib import Path

import pytest

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import app  # noqa: E402  (imports no Streamlit at module level)
from promo_finder.pipeline import run  # noqa: E402

SHOW = ROOT / "examples" / "synthetic_show"


def _zip(files: dict) -> bytes:
    buf = io.BytesIO()
    with zipfile.ZipFile(buf, "w", compression=zipfile.ZIP_STORED) as z:
        for name, data in files.items():
            z.writestr(name, data)
    return buf.getvalue()


def test_uploads_keep_only_txt_files_and_stay_inside_the_upload_folder(tmp_path):
    pack = _zip({"show/ep-01.txt": (SHOW / "ep-01.txt").read_bytes(), "../../escape.txt": b"x",
                 "C:evil.txt": b"x", "__MACOSX/show/._ep-01.txt": b"junk", "notes.pdf": b"%PDF"})
    names = app.collect_scripts([("pack.zip", pack), ("Episode_two.txt", (SHOW / "Episode_two.txt").read_bytes())],
                                tmp_path / "in")
    assert names == sorted(["ep-01.txt", "escape.txt", "C_evil.txt", "Episode_two.txt"])
    assert sorted(p.name for p in (tmp_path / "in").iterdir()) == names
    assert not (tmp_path / "escape.txt").exists() and not (tmp_path.parent / "escape.txt").exists()


def test_oversized_upload_is_refused(tmp_path, monkeypatch):
    monkeypatch.setattr(app, "MAX_UPLOAD_BYTES", 100)
    with pytest.raises(ValueError, match="larger than"):
        app.collect_scripts([("big.txt", b"x" * 101)], tmp_path / "in")


def test_app_gives_the_same_promos_as_the_command_line(tmp_path):
    res = app.run_promo_finder(SHOW, "demo", 10)
    run(str(SHOW), str(tmp_path / "cli"), show="demo", launch_window=10)
    cli = json.loads((tmp_path / "cli" / "promos.json").read_text(encoding="utf-8"))["promos"]
    key = lambda ps: [(p["promo_id"], p["segments"][0]["start_s"], p["segments"][0]["end_s"], p["rank"],
                       p["launch_rank"]) for p in ps]
    assert res["promos"] and key(res["promos"]) == key(cli)
    for p in res["promos"]:                       # the script view shows the promo's own hook and ending lines
        s = p["segments"][0]
        texts = [r["Text"] for r in app.clip_transcript(res["episodes"][s["episode"]], s["start_s"], s["end_s"])]
        assert p["hook"]["line"] in texts and p["cliffhanger"]["line"] in texts
    assert len(app.table_rows(res["promos"])) == len(res["promos"])


def test_page_runs_on_the_demo_show_and_on_an_uploaded_zip():
    pytest.importorskip("streamlit")
    from streamlit.testing.v1 import AppTest

    at = AppTest.from_file(str(ROOT / "app.py"), default_timeout=120).run()
    assert not at.exception
    at.button[0].click().run()
    assert not at.exception
    assert any(h.value.startswith("Post first") for h in at.subheader) and len(at.dataframe) >= 3
    assert {m.label: m.value for m in at.metric}["Promos found"] == "3"

    at.radio[0].set_value("Upload my own").run()
    pack = _zip({f"show/{f.name}": f.read_bytes() for f in SHOW.glob("*.txt")})
    at.file_uploader[0].upload("pack.zip", pack, "application/zip").run()
    at.button[0].click().run()
    assert not at.exception
    assert {m.label: m.value for m in at.metric}["Episodes read"] == "2"
