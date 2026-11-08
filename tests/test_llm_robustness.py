"""Regression tests for the LLM layer: malformed replies, provider contracts, replay. No network needed."""
import json
import sys
import threading
from http.server import BaseHTTPRequestHandler, HTTPServer
from pathlib import Path

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

import pytest  # noqa: E402

from promo_finder import config as C  # noqa: E402
from promo_finder.candidates import episode_windows, top_diverse  # noqa: E402
from promo_finder.llm_judge import MockProvider, SYSTEM, _score, validate_judgement  # noqa: E402
from promo_finder.parser import load_show  # noqa: E402
from promo_finder.pipeline import export_prompts, run  # noqa: E402

SHOW = ROOT / "examples" / "synthetic_show"


@pytest.fixture(autouse=True)
def no_waiting(monkeypatch):
    monkeypatch.setattr(C, "LLM_MIN_INTERVAL", 0)
    monkeypatch.setattr(C, "LLM_BACKOFF", 0)


@pytest.fixture(scope="module")
def ep1_cands():
    eps, _ = load_show(SHOW)
    ep = next(e for e in eps if e.number == 1)
    return ep, top_diverse(episode_windows(ep), 4)


def _statuses(out):
    return [r.split(",")[-1] for r in (out / "episodes.csv").read_text().splitlines()[1:]]


@pytest.mark.parametrize("value,expected", [
    (4, 4), ("4", 4), (4.4, 4), (4.5, 5), (1, 1), (5, 5),
    (9, None), (0, None), (-2, None), (True, None), (float("inf"), None), (float("nan"), None),
    ("abc", None), (None, None), ([4], None), ({"v": 4}, None)])
def test_score_is_rejected_not_clamped(value, expected):
    assert _score(value) == expected


@pytest.mark.parametrize("reply", [
    '{"candidates":[null]}', '{"candidates":[1,"x",[]]}', '{"candidates":{}}', '"just a string"', "[]",
    '{"candidates":[{"id":{"x":1},"hook_score":4,"cliffhanger_score":4,"standalone_score":4,"stakes_score":4}]}'])
def test_unusable_replies_raise_valueerror_only(reply, ep1_cands):
    ep, cands = ep1_cands
    with pytest.raises(ValueError):            # ValueError => retry, then heuristic fallback; never a crash
        validate_judgement(reply, ep, cands)


def test_chaos_reply_keeps_only_valid_fields(ep1_cands):
    ep, cands = ep1_cands
    raw, _ = MockProvider("chaos").complete(SYSTEM, "")
    res, warns = validate_judgement(raw, ep, cands)
    assert set(res) == {0}                                  # only C1 survives
    assert res[0]["hook_score"] == 3                         # bad hook id -> capped, code's hook used
    assert res[0]["hook_event_id"] is None and res[0]["cliffhanger_event_id"] is None
    assert res[0]["why"] == "" and res[0]["angle"] == "other" and res[0]["spoiler_risk"] == "unknown"
    assert any("duplicate judgement ignored" in w for w in warns)
    assert any("[9]" in w for w in warns)                    # out-of-range score reported, not clamped


def test_top_level_array_is_accepted(ep1_cands):
    ep, cands = ep1_cands
    raw = json.dumps([{"id": "C1", "hook_score": 5, "cliffhanger_score": 4, "standalone_score": 4, "stakes_score": 3}])
    res, _ = validate_judgement(raw, ep, cands)
    assert res[0]["hook_score"] == 5


@pytest.mark.parametrize("mode,allowed", [
    ("null", {"llm_failed_used_heuristic"}), ("malformed", {"llm_failed_used_heuristic"}), ("good", {"llm"}),
    ("chaos", {"llm_partial"})])
def test_pipeline_survives_every_reply_mode(tmp_path, mode, allowed):
    meta = run(str(SHOW), str(tmp_path), provider=f"mock:{mode}", show="t")
    assert meta["validation_failures"] == 0 and meta["promos"] >= 1
    st = _statuses(tmp_path)
    assert set(st) <= allowed and (mode != "chaos" or "llm_partial" in st)


# ---------------------------------------------------------------- provider contract tests (local HTTP server)
class _FakeLLM(BaseHTTPRequestHandler):
    calls = []

    def log_message(self, *a):
        pass

    def do_POST(self):
        body = json.loads(self.rfile.read(int(self.headers["Content-Length"])))
        _FakeLLM.calls.append((self.path, {k.lower(): v for k, v in self.headers.items()}, body))
        if self.path.endswith("/chat/completions") and "response_format" in body:
            return self._send(400, {"error": "response_format is not supported by this server"})
        if self.path.endswith("/chat/completions"):
            text, _ = MockProvider("good").complete(SYSTEM, body["messages"][1]["content"])
            return self._send(200, {"choices": [{"message": {"content": text}}],
                                    "usage": {"prompt_tokens": 1000, "completion_tokens": 200}})
        if self.path.endswith("/messages"):
            text, _ = MockProvider("good").complete(SYSTEM, body["messages"][0]["content"])
            return self._send(200, {"content": [{"type": "text", "text": text}],
                                    "usage": {"input_tokens": 900, "output_tokens": 150}})
        self._send(404, {"error": "not found"})

    def _send(self, code, obj):
        data = json.dumps(obj).encode()
        self.send_response(code)
        self.send_header("Content-Type", "application/json")
        self.send_header("Content-Length", str(len(data)))
        self.end_headers()
        self.wfile.write(data)


@pytest.fixture()
def fake_server():
    srv = HTTPServer(("127.0.0.1", 0), _FakeLLM)
    threading.Thread(target=srv.serve_forever, daemon=True).start()
    _FakeLLM.calls = []
    yield f"http://127.0.0.1:{srv.server_address[1]}/v1"
    srv.shutdown()


@pytest.mark.parametrize("provider", ["openai", "anthropic"])
def test_real_http_adapters_against_local_server(tmp_path, monkeypatch, fake_server, provider):
    monkeypatch.setenv("LLM_BASE_URL", fake_server)
    monkeypatch.setenv("LLM_API_KEY", "test-key")
    meta = run(str(SHOW), str(tmp_path), provider=provider, model="any-model", show="t")
    assert meta["validation_failures"] == 0 and set(_statuses(tmp_path)) == {"llm"}
    assert meta["llm_usage"]["in"] > 0 and meta["llm_usage"]["out"] > 0        # usage parsed from responses
    path, headers, body = _FakeLLM.calls[-1]
    assert body["model"] == "any-model"
    if provider == "openai":
        assert headers["authorization"] == "Bearer test-key" and "response_format" not in body   # JSON-mode fallback
    else:
        assert headers["x-api-key"] == "test-key" and body["system"] == SYSTEM


def test_missing_key_fails_fast_with_message(tmp_path, monkeypatch):
    for k in ("GEMINI_API_KEY", "LLM_API_KEY"):
        monkeypatch.delenv(k, raising=False)
    sys.path.insert(0, str(ROOT))
    import run as cli
    assert cli.main(["find", "--input", str(SHOW), "--out", str(tmp_path), "--provider", "gemini"]) == 2


def test_replay_uses_recorded_answers_and_skips_unrecorded(tmp_path):
    prompts = tmp_path / "prompts.jsonl"
    assert export_prompts(str(SHOW), str(prompts), show="t") == 2
    recs = [json.loads(x) for x in prompts.read_text().splitlines()]
    answered = recs[0]                                       # answer only the first episode's prompt
    text, _ = MockProvider("good").complete(SYSTEM, answered["user"])
    replies = tmp_path / "replies.jsonl"
    replies.write_text(json.dumps({"key": answered["key"], "response": text}) + "\n")
    meta = run(str(SHOW), str(tmp_path / "out"), provider=f"replay:{replies}", show="t")
    assert meta["validation_failures"] == 0
    assert sorted(_statuses(tmp_path / "out")) == ["llm", "no_recorded_response_used_heuristic"]
