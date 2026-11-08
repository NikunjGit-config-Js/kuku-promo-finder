"""Screenplay parser: turns one episode text file into timed Event records.

Rules:
- Never invent a timestamp. A block without a readable time line is kept as
  "untimed", reported as a warning, and excluded from all timing logic.
- Field names, scene labels and block labels are matched case-insensitively.
- Every problem is reported in Episode.warnings instead of being skipped silently.
"""
from __future__ import annotations

import re
from dataclasses import dataclass, field
from pathlib import Path

FIELD_RE = re.compile(
    r"^\s*(episode\s+number|episode\s+logline|continuity\s+notes|scene\s+heading|scene\s+start|scene\s+end)\s*:\s*(.*)$",
    re.I,
)
SCENE_RE = re.compile(r"^\s*scene\s+(\d+)\b\s*[:.\-–—]?\s*(.*)$", re.I)
# **CAMERA:**  /  **CAMERA**:  /  **MAYA (Stunned, Breathless):**
BLOCK_RE = re.compile(r"^\s*\*\*\s*(.+?)\s*:?\s*\*\*\s*:?\s*$")
TIME_RE = re.compile(
    r"start\s*:\s*\(?\s*([0-9][0-9:.]*)\s*\)?\s*[|,;]?\s*end\s*:\s*\(?\s*([0-9][0-9:.]*)\s*\)?", re.I
)
TS_RE = re.compile(r"^(?:(\d+):)?(\d{1,3}):(\d{1,2}(?:\.\d+)?)$|^(\d+(?:\.\d+)?)$")
MARKER_RE = re.compile(r"^\s*-{2,}\s*script\s+(start|end)\s*-{2,}\s*$", re.I)

KIND_BY_LABEL = {
    "CAMERA": "camera", "SHOT": "camera", "TRANSITION": "camera",
    "ACTION": "action",
    "SOUND FX": "sfx", "SFX": "sfx", "SOUND": "sfx", "MUSIC": "sfx", "SOUND EFFECT": "sfx",
    "TEXT ON SCREEN": "text", "SUPER": "text", "TITLE": "text", "ON-SCREEN TEXT": "text",
}
SPEAKER_FLAGS = {"O.S.", "V.O.", "O.C.", "OS", "VO", "CONT'D", "CONT’D", "CONTD", "PRE-LAP", "FILTERED"}


def parse_ts(raw: str) -> float | None:
    """'00:04.80' -> 4.8, '1:02:03.5' -> 3723.5, '4.8' -> 4.8. Returns None if unreadable."""
    m = TS_RE.match(raw.strip())
    if not m:
        return None
    if m.group(4) is not None:
        return round(float(m.group(4)), 3)
    h = int(m.group(1) or 0)
    return round(h * 3600 + int(m.group(2)) * 60 + float(m.group(3)), 3)


def fmt_ts(sec: float) -> str:
    """Seconds -> 'MM:SS.cc' (the format used by the source scripts)."""
    m, s = divmod(round(sec, 2), 60)
    return f"{int(m):02d}:{s:05.2f}"


@dataclass
class Event:
    eid: str
    episode: int
    scene: int
    idx: int
    kind: str                      # camera | action | sfx | dialogue | text
    label: str                     # raw block label, e.g. "MAYA (Stunned, Breathless)"
    speaker: str | None
    delivery: str | None           # parenthetical performance note, e.g. "Confused, Breathless"
    start: float | None
    end: float | None
    start_raw: str | None
    end_raw: str | None
    text: str
    line_no: int

    @property
    def timed(self) -> bool:
        return self.start is not None and self.end is not None and self.end > self.start


@dataclass
class Scene:
    number: int
    heading: str = ""
    start: float | None = None
    end: float | None = None


@dataclass
class Episode:
    number: int
    source: str
    logline: str = ""
    continuity: str = ""
    scenes: list[Scene] = field(default_factory=list)
    events: list[Event] = field(default_factory=list)
    warnings: list[str] = field(default_factory=list)

    @property
    def timed_events(self) -> list[Event]:
        return [e for e in self.events if e.timed]

    @property
    def duration(self) -> float:
        ends = [s.end for s in self.scenes if s.end is not None] + [e.end for e in self.timed_events]
        return max(ends) if ends else 0.0

    def event(self, eid: str) -> Event | None:
        for e in self.events:
            if e.eid == eid:
                return e
        return None


def _classify(label: str) -> tuple[str, str | None, str | None]:
    """Return (kind, speaker, delivery) for a block label."""
    parens = re.findall(r"\(([^)]*)\)", label)
    core = re.sub(r"\(.*?\)", "", label).strip(" :").upper()
    core = re.sub(r"\s+", " ", core)
    if core in KIND_BY_LABEL:
        return KIND_BY_LABEL[core], None, None
    notes = [p.strip() for p in parens if p.strip().upper() not in SPEAKER_FLAGS]
    return "dialogue", core, ", ".join(notes) or None


def parse_episode(path: str | Path, fallback_number: int | None = None) -> Episode:
    path = Path(path)
    raw = path.read_bytes()
    try:
        text = raw.decode("utf-8-sig")
    except UnicodeDecodeError:
        text = raw.decode("latin-1")
    lines = text.replace("\r\n", "\n").replace("\r", "\n").split("\n")

    ep = Episode(number=-1, source=path.name)
    scene: Scene | None = None
    pending: dict | None = None    # block being read

    def flush():
        nonlocal pending
        if pending is None:
            return
        kind, speaker, delivery = _classify(pending["label"])
        body = " ".join(pending["text"]).strip()
        if kind == "dialogue":
            body = body.strip().strip('"“”').strip()
        idx = len(ep.events)
        ev = Event(
            eid="", episode=ep.number, scene=scene.number if scene else 0, idx=idx, kind=kind,
            label=pending["label"], speaker=speaker, delivery=delivery,
            start=pending["start"], end=pending["end"], start_raw=pending["start_raw"],
            end_raw=pending["end_raw"], text=body, line_no=pending["line_no"],
        )
        if pending["start_raw"] is None:
            ep.warnings.append(f"line {ev.line_no}: block '{ev.label}' has no Start/End line; kept as untimed")
        elif ev.start is None or ev.end is None:
            ep.warnings.append(f"line {ev.line_no}: unreadable timestamp '{ev.start_raw}'/'{ev.end_raw}'; kept as untimed")
        elif ev.end <= ev.start:
            ep.warnings.append(f"line {ev.line_no}: end <= start ({ev.start_raw} -> {ev.end_raw}); excluded from timing")
        if not body:
            ep.warnings.append(f"line {ev.line_no}: block '{ev.label}' has empty text")
        ep.events.append(ev)
        pending = None

    for no, line in enumerate(lines, 1):
        s = line.strip()
        if not s or MARKER_RE.match(s) or s.lower().startswith("actions and dialogue"):
            if not s and pending is not None and pending["start_raw"] is not None and pending["text"]:
                flush()
            continue
        fm = FIELD_RE.match(s)
        if fm and not s.startswith("**"):
            key, val = re.sub(r"\s+", " ", fm.group(1).lower()), fm.group(2).strip()
            if key == "episode number":
                m = re.search(r"\d+", val)
                ep.number = int(m.group()) if m else -1
            elif key == "episode logline":
                ep.logline = val
            elif key == "continuity notes":
                ep.continuity = val
            elif scene is None:
                ep.warnings.append(f"line {no}: '{key}' outside any SCENE block; ignored")
            elif key == "scene heading":
                scene.heading = val
            else:
                t = parse_ts(val)
                if t is None:
                    ep.warnings.append(f"line {no}: unreadable {key} '{val}'")
                setattr(scene, "start" if key == "scene start" else "end", t)
            continue
        sm = SCENE_RE.match(s)
        if sm:
            flush()
            scene = Scene(number=int(sm.group(1)), heading=sm.group(2).strip())
            ep.scenes.append(scene)
            continue
        bm = BLOCK_RE.match(s)
        if bm:
            flush()
            pending = {"label": bm.group(1).strip(), "start": None, "end": None, "start_raw": None,
                       "end_raw": None, "text": [], "line_no": no}
            continue
        if pending is not None:
            tm = TIME_RE.search(s)
            if tm and pending["start_raw"] is None:
                pending["start_raw"], pending["end_raw"] = tm.group(1), tm.group(2)
                pending["start"], pending["end"] = parse_ts(tm.group(1)), parse_ts(tm.group(2))
            else:
                pending["text"].append(s)
            continue
        ep.warnings.append(f"line {no}: unrecognised line outside any block: {s[:60]!r}")
    flush()

    # Episode number: header -> file name. If both are missing, load_show assigns the next free number.
    if ep.number < 0:
        m = re.search(r"episode[-_ ]?(\d+)", path.stem, re.I)
        if m:
            ep.number = int(m.group(1))
            ep.warnings.append("EPISODE NUMBER missing; taken from file name")
        elif fallback_number is not None:
            ep.number = fallback_number
            ep.warnings.append(f"EPISODE NUMBER missing; assigned {fallback_number}")
    assign_ids(ep)

    # Scene sanity: fill missing bounds from events (flagged), check containment.
    for sc in ep.scenes:
        evs = [e for e in ep.timed_events if e.scene == sc.number]
        if sc.start is None and evs:
            sc.start = min(e.start for e in evs)
            ep.warnings.append(f"scene {sc.number}: missing start, inferred {fmt_ts(sc.start)} from events")
        if sc.end is None and evs:
            sc.end = max(e.end for e in evs)
            ep.warnings.append(f"scene {sc.number}: missing end, inferred {fmt_ts(sc.end)} from events")
        for e in evs:
            if sc.start is not None and sc.end is not None and (e.start < sc.start - 0.05 or e.end > sc.end + 0.05):
                ep.warnings.append(f"{e.eid} ({fmt_ts(e.start)}-{fmt_ts(e.end)}) lies outside scene {sc.number} bounds")
    if not ep.timed_events:
        ep.warnings.append("no timed events found")
    return ep


def assign_ids(ep: Episode) -> None:
    for i, ev in enumerate(ep.events):
        ev.episode = ep.number
        ev.eid = f"E{ep.number:02d}-{i + 1:03d}"


class InputError(ValueError):
    """The input path holds no readable screenplay (missing path, a .zip, a folder without .txt files)."""


def load_show(input_path: str | Path) -> tuple[list[Episode], list[str]]:
    """Load every .txt screenplay under a folder (or a single .txt file). Returns episodes + load errors.

    Episodes without any number, and later duplicates of a number, get the next unused number
    (reported as an error) so two files can never share event IDs."""
    p = Path(input_path)
    if not p.exists():
        return [], [f"{p}: path not found"]
    if p.is_file() and p.suffix.lower() != ".txt":   # e.g. the .zip pack, which is stored uncompressed and
        return [], [f"{p.name}: not a .txt screenplay; if it is the .zip pack, extract it and pass the folder"]
    files = sorted(p.rglob("*.txt")) if p.is_dir() else [p]
    episodes, errors = [], []
    for f in files:
        try:
            episodes.append(parse_episode(f))
        except Exception as exc:  # report, never hide
            errors.append(f"{f.name}: {type(exc).__name__}: {exc}")
    used = {e.number for e in episodes if e.number >= 0}
    seen: dict[int, str] = {}
    for ep in episodes:
        dup = ep.number in seen
        if ep.number < 0 or dup:
            new = max(used | {0}) + 1
            used.add(new)
            msg = (f"duplicate episode number {ep.number} ({seen[ep.number]} and {ep.source}); "
                   if dup else f"{ep.source}: no episode number in header or file name; ")
            errors.append(msg + f"{ep.source} renumbered to {new}")
            ep.warnings.append(f"episode number assigned as {new}")
            ep.number = new
            assign_ids(ep)
        seen[ep.number] = ep.source
    episodes.sort(key=lambda e: e.number)
    return episodes, errors


def require_episodes(input_path: str | Path) -> tuple[list[Episode], list[str]]:
    """load_show, but an input with no episode at all is an error, never an empty 'successful' run."""
    episodes, errors = load_show(input_path)
    if not episodes:
        why = errors[0] if errors else f"no .txt screenplay files under {input_path}"
        raise InputError(f"no screenplay episodes loaded ({why})")
    return episodes, errors
