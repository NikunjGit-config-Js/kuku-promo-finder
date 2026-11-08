"""Promo Finder web app: upload timed screenplays (or use the invented demo show) and get 20-60 s promo clips with
exact timestamps, a hook in the first 3 s, a cliffhanger cut and a spoiler label.

It runs the same pipeline as `python run.py find` in code-only mode (no API key, nothing stored).
Run locally:  streamlit run app.py
"""
from __future__ import annotations

import io
import json
import re
import tempfile
import zipfile
from pathlib import Path

ROOT = Path(__file__).resolve().parent
DEMO = ROOT / "examples" / "synthetic_show"
MAX_UPLOAD_BYTES = 25 * 1024 * 1024          # a 50-episode pack is about 0.5 MB
_UNSAFE = re.compile(r'[<>:"/\\|?*\x00-\x1f]')


def collect_scripts(uploads, dest: Path) -> list[str]:
    """Save uploaded .txt screenplays, and the .txt files inside uploaded .zip files, into `dest`.

    `uploads` is a list of (file name, bytes). Only the base file name is kept and unsafe characters are replaced,
    so an archive path can never write outside `dest`. Anything that is not .txt is ignored."""
    dest.mkdir(parents=True, exist_ok=True)
    written, total = [], 0

    def add(name: str, size: int, read) -> None:
        nonlocal total
        base = _UNSAFE.sub("_", name.replace("\\", "/").rsplit("/", 1)[-1])
        if not base.lower().endswith(".txt") or base.startswith("."):
            return
        total += size
        if total > MAX_UPLOAD_BYTES:
            raise ValueError(f"upload is larger than {MAX_UPLOAD_BYTES // 2 ** 20} MB")
        target, n = dest / base, 1
        while target.exists():                           # same name in two folders of the archive
            n += 1
            target = dest / f"{base[:-4]} ({n}).txt"
        target.write_bytes(read())
        written.append(target.name)

    for name, data in uploads:
        if name.lower().endswith(".zip"):
            with zipfile.ZipFile(io.BytesIO(data)) as z:
                for info in z.infolist():
                    if not info.is_dir():
                        add(info.filename, info.file_size, lambda info=info: z.read(info))
        else:
            add(name, len(data), lambda data=data: data)
    return sorted(written)


def run_promo_finder(src: Path, show: str, launch_window: int) -> dict:
    """Run the `run.py find` pipeline (code-only) in a temporary folder and return what the page shows."""
    from promo_finder.parser import load_show
    from promo_finder.pipeline import run

    with tempfile.TemporaryDirectory() as out:
        meta = run(str(src), out, show=show, launch_window=launch_window)
        files = {n: (Path(out) / n).read_bytes() for n in ("promos.csv", "promos.json", "top_promos.md")}
    promos = json.loads(files["promos.json"].decode("utf-8"))["promos"]
    episodes, _ = load_show(src)
    return {"meta": meta, "promos": promos, "files": files, "episodes": {e.number: e for e in episodes}}


def table_rows(promos: list[dict]) -> list[dict]:
    rows = []
    for p in promos:
        s, h, c = p["segments"][0], p["hook"], p["cliffhanger"]
        opens = h.get("line") or h.get("description") or ""
        rows.append({"Post #": p["launch_rank"], "Rank": p["rank"], "Episode": s["episode"], "Start": s["start"],
                     "End": s["end"], "Seconds": round(p["total_duration_s"], 1),
                     "Opens on": opens if h.get("kind") == "dialogue" else f"[{h.get('kind')}] {opens}",
                     "Cuts on": c.get("line") or "", "Spoiler": p["spoiler"]["tier"].title(), "Angle": p["angle"],
                     "Score": round(p["score"], 3), "Promo": p["promo_id"]})
    return rows


def clip_transcript(episode, start: float, end: float) -> list[dict]:
    """The script events inside a clip, timed from the clip's first frame: what the editor will cut."""
    return [{"At": f"+{e.start - start:.1f}s", "Type": e.kind, "Who": (e.speaker or "").title(), "Text": e.text}
            for e in sorted(episode.timed_events, key=lambda x: (x.start, x.idx))
            if start - 1e-6 <= e.start < end - 1e-6]


def main() -> None:
    import streamlit as st
    from promo_finder.parser import InputError

    st.set_page_config(page_title="Promo Finder", layout="wide")
    st.title("Promo Finder")
    st.write("Finds 20–60 s promo clips in timed microdrama screenplays: a hook in the first 3 seconds, a cliffhanger "
             "cut, exact timestamps taken from the script, and a spoiler label. Code-only mode, no API key. "
             "The editor makes the final call.")

    with st.sidebar:
        st.header("Input")
        use_demo = st.radio("Screenplays", ["Demo show (invented, 2 episodes)", "Upload my own"]).startswith("Demo")
        files = None
        if not use_demo:
            files = st.file_uploader("Episode .txt files, or one .zip of them", type=["txt", "zip"],
                                     accept_multiple_files=True)
        window = st.number_input("Launch window: first N episodes count as spoiler-safe", min_value=0,
                                 max_value=500, value=10, step=1, help="0 = judge spoilers on content only")
        go = st.button("Find promos", type="primary")
        st.caption("Uploads are processed in a temporary folder that is deleted after the run; nothing is stored. "
                   "The optional LLM judge runs from the command line with your own API key.")

    if go:
        try:
            with tempfile.TemporaryDirectory() as tmp:
                if use_demo:
                    src, show = DEMO, "Demo show (invented)"
                else:
                    if not files:
                        st.warning("Add episode .txt files or a .zip of them first.")
                        st.stop()
                    if not collect_scripts([(f.name, f.getvalue()) for f in files], Path(tmp) / "scripts"):
                        st.error("No .txt screenplay files found in the upload.")
                        st.stop()
                    src = Path(tmp) / "scripts"
                    show = files[0].name.rsplit(".", 1)[0] if len(files) == 1 else "Uploaded show"
                with st.spinner("Reading the scripts and scoring every legal 20–60 s window..."):
                    st.session_state["result"] = run_promo_finder(src, show, int(window))
        except (InputError, ValueError, zipfile.BadZipFile) as e:
            st.error(f"Could not process the upload: {e}")
            st.stop()

    res = st.session_state.get("result")
    if not res:
        st.info("Pick the demo show or upload screenplays in the sidebar, then press **Find promos**.")
        return
    meta, promos = res["meta"], res["promos"]
    cols = st.columns(5)
    cols[0].metric("Episodes read", meta["episodes_parsed"])
    cols[1].metric("Promos found", meta["promos"])
    cols[2].metric("Episodes with a promo", len({p["episodes"][0] for p in promos}))
    cols[3].metric("Failed checks", meta["validation_failures"])
    cols[4].metric("Run time", f"{meta['seconds']} s")
    if meta["load_errors"]:
        with st.expander(f"Input notes ({len(meta['load_errors'])})"):
            st.write("\n".join(f"- {m}" for m in meta["load_errors"]))

    launch = sorted([p for p in promos if p["launch_rank"]], key=lambda p: p["launch_rank"])
    st.subheader(f"Post first: {len(launch)} clip{'' if len(launch) == 1 else 's'}")
    st.caption("One per episode, from the top half by quality, never Red, at most 2 Amber. Green: from the launch "
               "window. Amber: later, no reveal wording found. Red: later, and it confirms a reveal.")
    if launch:
        st.dataframe(table_rows(launch), hide_index=True, width="stretch")
    else:
        st.write("No clip qualified for the first batch.")

    st.subheader("Every promo, ranked by quality")
    tiers = st.multiselect("Spoiler tiers", ["Green", "Amber", "Red"], default=["Green", "Amber", "Red"])
    st.dataframe([r for r in table_rows(promos) if r["Spoiler"] in tiers], hide_index=True, width="stretch")

    if promos:
        st.subheader("Check a clip against the script")
        labels = {p["promo_id"]: f"{p['promo_id']}  ·  Ep {p['segments'][0]['episode']}  ·  "
                                 f"{p['segments'][0]['start']}–{p['segments'][0]['end']}" for p in promos}
        pid = st.selectbox("Promo", list(labels), format_func=labels.get)
        p = next(x for x in promos if x["promo_id"] == pid)
        s = p["segments"][0]
        st.markdown(f"**Episode {s['episode']}, {s['start']}–{s['end']}** ({p['total_duration_s']:.1f} s) · spoiler "
                    f"**{p['spoiler']['tier'].title()}** · angle *{p['angle']}*")
        st.write(p["why"])
        if p["spoiler"]["evidence"]:
            st.caption("Spoiler evidence: " + "; ".join(p["spoiler"]["evidence"]))
        ep = res["episodes"].get(s["episode"])
        if ep:
            st.dataframe(clip_transcript(ep, s["start_s"], s["end_s"]), hide_index=True, width="stretch")

    st.subheader("Download")
    for col, (name, mime) in zip(st.columns(3), (("promos.csv", "text/csv"), ("promos.json", "application/json"),
                                                 ("top_promos.md", "text/markdown"))):
        col.download_button(name, res["files"][name], file_name=name, mime=mime)


if __name__ == "__main__":
    main()
