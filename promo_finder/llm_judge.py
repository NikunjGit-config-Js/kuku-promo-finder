"""LLM judge: one call per episode scores the code-generated candidate windows.

Design rules
- The model never produces timestamps. It sees event IDs and candidate IDs and answers with IDs;
  code maps IDs back to source timestamps and rejects anything that doesn't exist.
- Any provider failure or malformed answer falls back to the heuristic scores for that episode,
  and the raw request/response is logged so a reviewer can audit it.
- Responses are cached by prompt hash: re-running costs nothing and is reproducible.
"""
from __future__ import annotations

import hashlib
import http.client
import json
import math
import os
import re
import time
import urllib.error
import urllib.request
from pathlib import Path

from . import config as C
from .candidates import Window, transcript_lines
from .parser import Episode, fmt_ts

SYSTEM = (
    "You are a senior promo editor for vertical microdrama ads on TikTok, Instagram Reels and YouTube Shorts. "
    "You judge whether a clip cut from a screenplay works as a 20-60 second ad for a viewer who has NEVER seen "
    "the show and is scrolling fast. Use only the transcript given. Never invent or rewrite dialogue. "
    "Refer to events only by their IDs. Reply with JSON only."
)

ANGLES = ["betrayal", "humiliation", "revenge", "romance", "secret_identity", "danger", "villain_scheme",
          "family_conflict", "other"]


def build_prompt(ep: Episode, cands: list[Window], show: str) -> str:
    lines = "\n".join(transcript_lines(ep))
    clist = "\n".join(
        f"C{i}: {fmt_ts(w.start)} -> {fmt_ts(w.end)} ({w.duration:.1f}s); first event {w.events[0].eid}, "
        f"last dialogue {w.cliff_event.eid}" for i, w in enumerate(cands, 1))
    return f"""SHOW: {show} | EPISODE {ep.number}
Logline (for YOUR understanding only; the viewer of the ad will not know it): {ep.logline}

TRANSCRIPT — id | start-end | type or speaker (delivery): text
{lines}

CANDIDATE CLIPS — each plays every event from its start time to its end time. Length and clean cut points
were already checked by code; you judge storytelling only.
{clist}

Judge each candidate AS A STRANGER WOULD SEE IT: only the events inside the clip. Use the rest of the
episode only to understand what the clip leaves unresolved.
Scores are integers 1-5:
- hook_score: does something in the FIRST 3 SECONDS stop a scrolling viewer (slap, reveal, shocking line)? 1 = nothing happens yet.
- cliffhanger_score: does it cut at an unresolved moment the viewer needs resolved? 1 = resolved, calm or flat.
- standalone_score: can a stranger tell who is involved, what is happening and what is at stake? 1 = confusing without earlier episodes.
- stakes_score: how much is at stake emotionally (humiliation, danger, love, identity)?

Return exactly this JSON shape, one object per candidate, in order:
{{"candidates": [{{"id": "C1", "hook_event_id": "<id of the hook event; it must start in the clip's first 3 s, or null>",
"hook": "<=12 words", "hook_score": 1, "cliffhanger_event_id": "<id of the event the clip leaves hanging>",
"cliffhanger": "<=12 words", "cliffhanger_score": 1, "standalone_score": 1, "stakes_score": 1,
"spoiler_risk": "low|medium|high", "angle": "{'|'.join(ANGLES)}",
"why": "<one sentence, <=25 words, why it works or fails as an ad>", "verdict": "use|reject"}}]}}"""


# --------------------------------------------------------------------------- providers
class LLMError(RuntimeError):
    pass


class NoRecordedResponse(LLMError):
    """Replay mode has no stored answer for this prompt (the episode was not part of the offline batch)."""


def prompt_key(system: str, user: str) -> str:
    return hashlib.sha256(f"{system}|{user}".encode()).hexdigest()[:24]


def _post(url: str, headers: dict, payload: dict, timeout: int) -> dict:
    req = urllib.request.Request(url, data=json.dumps(payload).encode(), method="POST",
                                 headers={"Content-Type": "application/json", **headers})
    try:
        with urllib.request.urlopen(req, timeout=timeout) as r:
            return json.loads(r.read().decode())
    except urllib.error.HTTPError as e:
        body = e.read().decode(errors="replace")[:500]
        raise LLMError(f"HTTP {e.code}: {body}") from e
    except (urllib.error.URLError, OSError, http.client.HTTPException, ValueError) as e:
        raise LLMError(f"{type(e).__name__}: {e}") from e


class Provider:
    """Uniform interface: complete(system, user) -> (text, usage)."""

    def __init__(self, name: str, model: str | None = None, base_url: str | None = None, api_key: str | None = None):
        preset = C.PROVIDERS.get(name, {})
        self.name = name
        self.model = model or os.environ.get("LLM_MODEL") or preset.get("model")
        self.base_url = (base_url or os.environ.get("LLM_BASE_URL") or preset.get("base_url", "")).rstrip("/")
        key_env = preset.get("key_env")
        self.api_key = api_key or os.environ.get("LLM_API_KEY") or (os.environ.get(key_env) if key_env else None)
        self._json_mode = True
        if key_env and not self.api_key:
            raise LLMError(f"No API key: set {key_env} (or LLM_API_KEY) in your environment")

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        if self.name == "anthropic":
            data = _post(f"{self.base_url}/messages",
                         {"x-api-key": self.api_key, "anthropic-version": "2023-06-01"},
                         {"model": self.model, "max_tokens": 3000, "temperature": C.LLM_TEMPERATURE,
                          "system": system, "messages": [{"role": "user", "content": user}]}, C.LLM_TIMEOUT)
            text = "".join(b.get("text", "") for b in data.get("content", []))
            u = data.get("usage", {})
            return text, {"in": u.get("input_tokens", 0), "out": u.get("output_tokens", 0)}
        payload = {"model": self.model, "temperature": C.LLM_TEMPERATURE,
                   "messages": [{"role": "system", "content": system}, {"role": "user", "content": user}]}
        if self._json_mode:
            payload["response_format"] = {"type": "json_object"}
        headers = {"Authorization": f"Bearer {self.api_key}"} if self.api_key else {}
        try:
            data = _post(f"{self.base_url}/chat/completions", headers, payload, C.LLM_TIMEOUT)
        except LLMError as e:
            if self._json_mode and "response_format" in str(e):   # provider without JSON mode
                self._json_mode = False
                return self.complete(system, user)
            raise
        try:
            text = data["choices"][0]["message"]["content"] or ""
            if isinstance(text, list):
                text = "".join(part.get("text", "") for part in text if isinstance(part, dict))
        except (KeyError, IndexError, TypeError, AttributeError) as e:
            raise LLMError(f"unexpected response shape: {str(data)[:300]}") from e
        u = data.get("usage") or {}
        return text, {"in": u.get("prompt_tokens", 0), "out": u.get("completion_tokens", 0)}


class MockProvider(Provider):
    """Offline stand-in for tests and demos. Modes: good | malformed | bad_ids | fail.

    'good' answers with plausible scores derived from the heuristic features it can read in the
    prompt, so it exercises every code path without pretending to be a real model.
    """

    def __init__(self, mode: str = "good"):
        self.name, self.model, self.mode = "mock", f"mock-{mode}", mode

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        if self.mode == "fail":
            raise LLMError("mock provider failure")
        if self.mode == "malformed":
            return "Sure! Here are my thoughts: C1 is great, C2 less so.", {"in": 0, "out": 0}
        if self.mode == "null":
            return '{"candidates":[null]}', {"in": 0, "out": 0}
        if self.mode == "chaos":   # syntactically valid JSON, semantically broken in every way we know of
            ok = {"cliffhanger_score": 4, "standalone_score": 4, "stakes_score": 4}
            return json.dumps({"candidates": [
                1, "C1", None, [], {"id": True}, {"id": {"x": 1}},
                {"id": "C2", "hook_score": 1e309, **ok}, {"id": "C2", "hook_score": 9, **ok},
                {"id": "C3", "hook_score": True, **ok},
                {"id": "C1", "hook_score": 5, "cliffhanger_score": 5, "standalone_score": 5, "stakes_score": 5,
                 "hook_event_id": {"a": 1}, "cliffhanger_event_id": ["x"], "why": {"nested": True},
                 "angle": ["romance"], "spoiler_risk": 3, "verdict": None},
                {"id": "C1", "hook_score": 1, "cliffhanger_score": 1, "standalone_score": 1, "stakes_score": 1},
            ]}), {"in": 0, "out": 0}
        cands = re.findall(r"^C(\d+): .*?first event (\S+), last dialogue (\S+)$", user, re.M)
        out = []
        for n, first, last in cands:
            hook_id = "E99-999" if self.mode == "bad_ids" else first.rstrip(",")
            out.append({"id": f"C{n}", "hook_event_id": hook_id, "hook": "mock hook", "hook_score": 4,
                        "cliffhanger_event_id": last, "cliffhanger": "mock cliffhanger", "cliffhanger_score": 3,
                        "standalone_score": 3, "stakes_score": 3, "spoiler_risk": "low", "angle": "other",
                        "why": "Mock judgement for pipeline testing only.", "verdict": "use"})
        return "```json\n" + json.dumps({"candidates": out}) + "\n```", {"in": 0, "out": 0}


class ReplayProvider(Provider):
    """Answers from a JSONL file of recorded replies ({"key": prompt_key, "response": "..."}), e.g. produced
    by a batch API or by pasting exported prompts into any chat model. Unknown prompts raise
    NoRecordedResponse, so those episodes simply keep heuristic scores."""

    def __init__(self, path: str):
        self.name, self.model, self.path = "replay", f"replay:{Path(path).name}", path
        self.answers = {}
        for line in Path(path).read_text(encoding="utf-8").splitlines():
            if line.strip():
                rec = json.loads(line)
                self.answers[rec["key"]] = rec["response"]

    def complete(self, system: str, user: str) -> tuple[str, dict]:
        key = prompt_key(system, user)
        if key not in self.answers:
            raise NoRecordedResponse(f"no recorded response for prompt {key}")
        return self.answers[key], {"in": 0, "out": 0}


def make_provider(name: str, model: str | None = None) -> Provider:
    if name.startswith("replay:"):
        return ReplayProvider(name.split(":", 1)[1])
    if name.startswith("mock"):
        return MockProvider(name.split(":", 1)[1] if ":" in name else "good")
    if name not in C.PROVIDERS:
        raise LLMError(f"unknown provider '{name}'; choose from heuristic, mock, replay:<file>, {', '.join(C.PROVIDERS)}")
    return Provider(name, model=model)


# --------------------------------------------------------------------------- parsing + validation
SCORE_KEYS = ("hook_score", "cliffhanger_score", "standalone_score", "stakes_score")
_CAND_ID = re.compile(r"^\s*C?\s*(\d+)\s*$", re.I)


def _extract_json(raw: str):
    """Parse the reply; tolerate code fences and chatter around one JSON value."""
    s = re.sub(r"^```(?:json)?\s*|\s*```$", "", str(raw).strip())
    try:
        return json.loads(s)
    except json.JSONDecodeError:
        a, b = s.find("{"), s.rfind("}")
        if a < 0 or b <= a:
            raise ValueError("no JSON object in reply")
        return json.loads(s[a:b + 1])


def _score(v) -> int | None:
    """An integer 1-5, or None. Out-of-range, boolean and non-finite values are rejected, not clamped:
    a model answering on a 1-10 scale must not be silently read as a 5."""
    if v is None or isinstance(v, bool) or not isinstance(v, (int, float, str)):
        return None
    try:
        f = float(v)
    except ValueError:
        return None
    if not math.isfinite(f) or not 1.0 <= f <= 5.0:
        return None
    return int(f + 0.5)


def _text(v, n: int) -> str:
    return str(v)[:n] if isinstance(v, (str, int, float)) and not isinstance(v, bool) else ""


def _event_id(v) -> str | None:
    return v.strip() if isinstance(v, str) and v.strip() else None


def validate_judgement(raw: str, ep: Episode, cands: list[Window]) -> tuple[dict[int, dict], list[str]]:
    """Parse the model reply and check every field and ID against the source.

    Never raises on a syntactically valid but malformed reply: bad items are skipped with a warning
    and those candidates keep their heuristic scores. Raises ValueError only if nothing usable exists."""
    data = _extract_json(raw)
    items = data.get("candidates") if isinstance(data, dict) else data if isinstance(data, list) else None
    if not isinstance(items, list):
        raise ValueError("reply has no 'candidates' list")
    warns, out = [], {}
    for n, it in enumerate(items, 1):
        if not isinstance(it, dict):
            warns.append(f"item {n} is {type(it).__name__}, not an object; ignored")
            continue
        rid = it.get("id")
        m = _CAND_ID.match(str(rid)) if isinstance(rid, (str, int)) and not isinstance(rid, bool) else None
        if not m or not 1 <= int(m.group(1)) <= len(cands):
            warns.append(f"item {n}: unknown candidate id {rid!r}; ignored")
            continue
        i = int(m.group(1)) - 1
        if i in out:
            warns.append(f"C{i + 1}: duplicate judgement ignored (first one kept)")
            continue
        j = {k: _score(it.get(k)) for k in SCORE_KEYS}
        bad = [k for k, v in j.items() if v is None]
        if bad:
            warns.append(f"C{i + 1}: invalid/out-of-range {', '.join(bad)} {[it.get(k) for k in bad]!r}; "
                         "candidate keeps heuristic scores")
            continue
        w = cands[i]
        in_win = {e.eid: e for e in w.events}
        hid, cid = _event_id(it.get("hook_event_id")), _event_id(it.get("cliffhanger_event_id"))
        if it.get("hook_event_id") is not None and (hid not in in_win or
                                                     in_win[hid].start > w.start + C.HOOK_WINDOW + C.EPS):
            warns.append(f"C{i + 1}: hook_event_id {it.get('hook_event_id')!r} is not an event in the clip's "
                         "first 3 s; using code's hook event")
            hid = None
            j["hook_score"] = min(j["hook_score"], 3)
        if it.get("cliffhanger_event_id") is not None and cid not in in_win:
            warns.append(f"C{i + 1}: cliffhanger_event_id {it.get('cliffhanger_event_id')!r} is outside the clip; "
                         "using code's last line")
            cid = None
        risk, angle = it.get("spoiler_risk"), it.get("angle")
        j.update({
            "hook_event_id": hid if hid in in_win else None, "cliffhanger_event_id": cid if cid in in_win else None,
            "hook": _text(it.get("hook"), 120), "cliffhanger": _text(it.get("cliffhanger"), 120),
            "why": " ".join(_text(it.get("why"), 400).split()[:40]),
            "spoiler_risk": risk if isinstance(risk, str) and risk in ("low", "medium", "high") else "unknown",
            "angle": angle if isinstance(angle, str) and angle in ANGLES else "other",
            "verdict": "reject" if _text(it.get("verdict"), 20).lower().startswith("rej") else "use",
        })
        out[i] = j
    if not out:
        raise ValueError("no usable judgement in reply: " + "; ".join(warns[:3]))
    missing = [f"C{i + 1}" for i in range(len(cands)) if i not in out]
    if missing:
        warns.append(f"no valid judgement for {', '.join(missing)}; those keep heuristic scores")
    return out, warns


# --------------------------------------------------------------------------- orchestration
class Judge:
    def __init__(self, provider: Provider, cache_dir: Path, log_path: Path, show: str):
        self.p, self.cache_dir, self.log_path, self.show = provider, cache_dir, log_path, show
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.usage = {"calls": 0, "cached": 0, "failed": 0, "in": 0, "out": 0, "seconds": 0.0}
        self._last = 0.0

    def _log(self, rec: dict):
        with open(self.log_path, "a", encoding="utf-8") as f:
            f.write(json.dumps(rec, ensure_ascii=False) + "\n")

    def judge(self, ep: Episode, cands: list[Window]) -> tuple[dict[int, dict] | None, list[str]]:
        if not cands:
            return {}, []
        user = build_prompt(ep, cands, self.show)
        key = hashlib.sha256(f"{self.p.name}|{self.p.model}|{SYSTEM}|{user}".encode()).hexdigest()[:24]
        cache = self.cache_dir / f"{key}.json"
        if cache.exists():
            raw = json.loads(cache.read_text(encoding="utf-8"))["raw"]
            self.usage["cached"] += 1
            try:
                return validate_judgement(raw, ep, cands)
            except Exception:  # stale or corrupt cache entry: drop it and ask again
                cache.unlink()
        errors = []
        for attempt in range(1, C.LLM_RETRIES + 1):
            wait = C.LLM_MIN_INTERVAL - (time.time() - self._last)
            if wait > 0 and self.p.name not in ("mock", "replay"):
                time.sleep(wait)
            t0 = time.time()
            self._last = t0
            try:
                raw, usage = self.p.complete(SYSTEM, user)
                dt = time.time() - t0
                self.usage["calls"] += 1
                self.usage["in"] += usage.get("in", 0)
                self.usage["out"] += usage.get("out", 0)
                self.usage["seconds"] += dt
                self._log({"episode": ep.number, "attempt": attempt, "provider": self.p.name, "model": self.p.model,
                           "seconds": round(dt, 2), "usage": usage, "prompt_chars": len(user), "response": raw})
                res, warns = validate_judgement(raw, ep, cands)
                cache.write_text(json.dumps({"episode": ep.number, "raw": raw}), encoding="utf-8")
                return res, warns
            except NoRecordedResponse as e:          # replay mode: this prompt was never answered offline
                self._log({"episode": ep.number, "provider": self.p.name, "error": str(e)})
                return None, [str(e)]
            except Exception as e:  # provider error or unusable reply -> retry, then heuristic fallback
                if isinstance(e, (KeyboardInterrupt, SystemExit)):
                    raise
                errors.append(f"attempt {attempt}: {type(e).__name__}: {str(e)[:200]}")
                self._log({"episode": ep.number, "attempt": attempt, "provider": self.p.name, "error": errors[-1]})
                if "HTTP 429" in str(e) or "HTTP 5" in str(e):
                    time.sleep(min(60, C.LLM_BACKOFF * attempt))
        self.usage["failed"] += 1
        return None, errors
