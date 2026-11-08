# Promo Finder: Kuku AI Generalist assignment, Problem 1

Reads microdrama screenplays and returns ranked **20–60 s promo clips**. Each clip opens on a hook in its first 3 seconds and cuts on an unresolved moment. Every start and end time is copied from the script and checked before it is written out. It produces two lists:
- **Quality ranking:** the best clips across all episodes.
- **Launch batch:** what to post first, without Red spoilers.

The tool needs no dependencies and no API key. An optional LLM judge (free Gemini tier, or any OpenAI-compatible or Anthropic model) can re-score candidates.

Two ways to use it: a **web app** (`app.py`, Streamlit: upload episode `.txt` files or the pack `.zip`, or try the invented demo show) and a **command line** (`run.py`). Both run the same pipeline and give the same promos.

## Web app

```powershell
python -m pip install -r requirements.txt
streamlit run app.py          # opens http://localhost:8501
```

Pick the demo show or upload screenplays, press **Find promos**, and you get: the first posting batch, every promo ranked by quality with its spoiler tier, a clip-by-clip view of the script lines inside each cut (to check it before editing), and downloads (`promos.csv`, `promos.json`, `top_promos.md`). It runs in code-only mode; uploads are processed in a temporary folder that is deleted after the run.

**Deploy it free on Streamlit Community Cloud** (about 5 minutes, after pushing this repo to GitHub): sign in at share.streamlit.io with GitHub, create an app from this repository, branch `main`, main file `app.py` (Python 3.12 is a safe choice under advanced settings), and deploy. The hosted app contains only the invented demo show. For a demo on Kuku's own scripts, run it locally so they stay on your machine.

## Quick start on Windows (PowerShell, about 10 minutes)

Runs on an ordinary laptop CPU: no GPU, CUDA, WSL or API key. Tested with Python 3.10, 3.11, 3.12, 3.13 and 3.14; the commands work in Windows PowerShell 5.1 and PowerShell 7.

```powershell
# 1. Python 3.10 or newer. If 'py' is not recognised, install Python from python.org and tick "Add python.exe to PATH".
py --version

# 2. Virtual environment, inside the extracted repo folder
cd "$HOME\Downloads\kuku-promo-finder"
py -m venv .venv
Set-ExecutionPolicy -Scope Process -ExecutionPolicy Bypass   # only if activation is blocked; affects this window only
.\.venv\Scripts\Activate.ps1                                  # the prompt now starts with (.venv)

# 3. Dependencies: the tool itself is standard-library only; Streamlit runs the web app, pytest the tests
python -m pip install -r requirements.txt

# 4. Tests (no data, no network)                 -> run the 71 collected tests
python -m pytest -q

# 5. Synthetic example                           -> episodes_parsed 2, promos 3, validation_failures 0
python run.py find --input examples\synthetic_show --out outputs\demo

# 5b. Web app: a browser tab opens; press Find promos on the demo show (Ctrl+C in PowerShell to stop)
streamlit run app.py
```

**Your own screenplays.** Keep the pack and every output outside this folder: outputs quote the scripts, and this repo is public.

```powershell
$Private = "$HOME\kuku_private"
Expand-Archive -Path "$HOME\Downloads\<screenplay-pack>.zip" -DestinationPath "$Private\pack"   # once; pass the folder, never the .zip

# 6. All episodes (about 3 s for 50)             -> episodes_parsed 50, load_errors [], promos 89, validation_failures 0
python run.py find --input "$Private\pack" --out "$Private\runs\local"
$LASTEXITCODE                                    # 0 = success; 2 = error, with the reason printed above

# 7. Structural checks against simple baselines (episode end, episode start, random)
python run.py eval --input "$Private\pack" --run "$Private\runs\local"

# 8. Compare with the supplied baseline (from the private raw-outputs zip)
Expand-Archive -Path "$HOME\Downloads\raw_outputs_v5.2.zip" -DestinationPath "$Private\baseline"
python run.py compare --input "$Private\pack" --old "$Private\baseline\raw_outputs_final\final_run" --new "$Private\runs\local"
#    -> "same_recommendations": true and "first_batch_overlap": 7
(Get-FileHash "$Private\runs\local\promos.csv").Hash -eq (Get-FileHash "$Private\baseline\raw_outputs_final\final_run\promos.csv").Hash
#    -> True on Python 3.12/3.13. On 3.10/3.11 it can be False because one score differs in the 4th decimal
#       (Python 3.12 changed how floats are summed); the compare line above is the real test.

# 9. Blind review kit: 15 clips = 5 episodes x (this version, previous version, episode ending), shuffled
python run.py review --input "$Private\pack" --run "$Private\runs\local" --previous "$Private\baseline\raw_outputs_final\previous_version_v4" --episodes 5
#    Read review\review_packet.md, fill review\review_sheet.csv (Excel is fine; keep it as CSV), do not open review_key.json

# 10. Score your ratings                          -> "rated_clips": 15, "skipped_rows": 0
python run.py score-review --run "$Private\runs\local"
```

| Step | Looks like a failure | What to do |
|---|---|---|
| 1 | `'py' is not recognized` | Install Python 3.12 or 3.13 from python.org, or try `python --version` |
| 2 | `running scripts is disabled on this system` | Run the `Set-ExecutionPolicy` line, then activate again. Or skip activation and use `.\.venv\Scripts\python.exe` in place of `python` |
| 4 | any `failed` or `error` | Send the last 30 lines of output |
| 6 | `error: no screenplay episodes loaded (... not a .txt screenplay ...)` | You passed the .zip: extract it and pass the folder |
| 6 | `(... path not found)` or `(no .txt screenplay files ...)` | Check the path; the folder may be one level deeper |
| 6 | `promos` is not 89, or `validation_failures` > 0 (exit code 2) | Different code or input version; send `run_meta.json` |
| 8 | `"same_recommendations": false` | Different code or input version; send `compare_vs_previous.json` |
| 10 | `error: no row could be scored` | Save the sheet as CSV (not .xlsx) and keep its header row |

**macOS / Linux:** the same commands with `python3 -m venv .venv`, `source .venv/bin/activate`, forward slashes, and `unzip` for extracting.

`--input` takes a folder (searched recursively for `.txt`) or a single `.txt` file. A `.zip`, a missing path or a folder without `.txt` files stops with exit code 2 instead of producing an empty or misleading run.

## Commands

| Command | What it does |
|---|---|
| `find --input D --out O [--provider P] [--model M] [--episodes 1,4,10-12] [--launch-window N]` | Run the pipeline. Exit code 2 if a promo fails validation, a provider is misconfigured or the input holds no episodes |
| `eval --input D --run O` | Structural checks of the picks against simple baselines (episode end, episode start, random) |
| `compare --input D --old O1 --new O2` | `same_recommendations` (same promos, timestamps and ranks), plus a like-for-like comparison of the two runs' first posting batches |
| `review --input D --run O [--previous O_old] [--episodes 5]` | Blind review kit, about 20 min: for 5 episodes spread across the series, the current pick, the previous version's pick and an episode-ending clip, shuffled with the method hidden. Rate hook, cliffhanger and standalone clarity 1–5, and would-post yes/no |
| `score-review --run O` | Score the filled sheet (plain CSV or either Excel CSV format): averages per method plus per-episode head-to-heads, with a verdict that says "no clear difference" unless one side wins at least 4 of 5 episodes. A blank would-post answer counts as unanswered |
| `export-prompts --input D --out prompts.jsonl [--episodes ...]` | Write the exact LLM prompts for a batch API or offline model; answers come back via `--provider replay:FILE` |

## Model providers (swap with one flag; keys come only from environment variables)

| `--provider` | Key env var | Default model | Notes |
|---|---|---|---|
| `heuristic` (default) | none | none | Offline, deterministic, free |
| `gemini` | `GEMINI_API_KEY` | gemini-2.5-flash | Free tier, about 10 requests/min. The tool waits 6.5 s between calls, so 50 episodes take about 10–20 min. Free-tier prompts may be used by Google for training |
| `groq` / `openrouter` | `GROQ_API_KEY` / `OPENROUTER_API_KEY` | Llama 3.3 70B | OpenAI-compatible |
| `ollama` | none | qwen2.5:7b-instruct | Local model at `http://localhost:11434` |
| `openai` | `OPENAI_API_KEY` | gpt-4o-mini | Any OpenAI-compatible gateway via `LLM_BASE_URL` and `LLM_API_KEY` |
| `anthropic` | `ANTHROPIC_API_KEY` | claude-haiku-5-5 | Override with `--model` |
| `replay:FILE` | none | none | Recorded answers keyed by prompt hash (batch or offline use) |
| `mock:good\|malformed\|null\|chaos\|bad_ids\|fail` | none | none | Test doubles only. Their outputs are not model judgements |

Setting a key: bash `export GEMINI_API_KEY=...` · PowerShell `$env:GEMINI_API_KEY="..."`. Never put keys in files in this repo.

**Live API test (8 Nov 2026):** Gemini `gemini-3.5-flash-lite` was tested successfully on the two-episode invented demo show: 2 calls, 0 failures, 1 selected promo, and 0 validation failures; both episodes had `status=llm`. The complete confidential 50-episode evaluation used the local heuristic provider, not Gemini. The adapter regression tests also use a local fake server. To reproduce an optional live test, set your own `GEMINI_API_KEY` in the environment and run `python run.py find --input examples/synthetic_show --out outputs/gemini_demo --provider gemini --model gemini-3.5-flash-lite`. Never commit or share your API key.

## How it works

```
.txt screenplays ─► parse (code) ─► legal cut points: never inside a spoken line (code)
  ─► every 20–60 s window, scored on hook, cliffhanger, context, escalation, intensity (code)
  ─► 6 distinct candidates per episode ─► optional LLM judge: one call per episode, answers by event ID
  ─► per-criterion floors (code) ─► validator: timestamps, durations, quotes (code)
  ─► quality ranking + launch batch with spoiler tier and angle (code) ─► editor approves (human)
```

- **The LLM never writes a timestamp or a line.** It scores code-generated candidates and cites event IDs. Unknown IDs, out-of-range scores and malformed JSON are rejected, not repaired. After 3 attempts the episode falls back to the heuristic, and the reason is logged.
- **Floors, not averages:** a clip must clear the hook, cliffhanger and context floors separately.
- **Editing effort** (clean cuts, length) is reported but only breaks near-ties: an editor can fix a jump cut, not a weak story.
- **Launch batch:**
  - never Red, at most 2 Amber, one clip per episode, top half by quality only
  - a small nudge toward different story angles
  - Green means the clip comes from episodes 1–`--launch-window` (default 10). That window is an editorial assumption to set per show, not an industry rule.

## Output schema (`promos.json` → `promos[]`; the same fields appear flattened in `promos.csv`)

| Field | Meaning |
|---|---|
| `rank` / `launch_rank` | Position in the quality ranking / in the launch batch (`null` if not in the batch) |
| `segments[]` | `episode`, `start`/`end` (MM:SS.cc), `start_s`/`end_s`, `first_event_id`, `last_event_id`, `source_file` |
| `hook`, `cliffhanger` | `event_id`, `at`, `offset_s`, `speaker`, `delivery`, exact `line` from the script, `description` |
| `why` | One line built from the clip's own lines (or the LLM's reason) |
| `score`, `features` | Quality score and its parts: hook, cliff, context, escalation, intensity, plus `edit_effort` and flags |
| `spoiler` | `tier` (green/amber/red), `confidence`, `evidence[]` |
| `angle`, `angle_evidence` | betrayal / humiliation / revenge / romance / secret_identity / danger / conflict, with the matched words |
| `method`, `llm` | `heuristic` or `llm:<model>` and the raw judge scores |
| `validation_errors` | Must be empty; the run exits with code 2 otherwise |

Other files: `top_promos.md` (editor view), `episodes.csv` (one row per input file, with status), `rejected_candidates.json` (with reasons), `parse_report.json`, `run_meta.json` (provider, model, token usage, config), `llm_calls.jsonl` (raw model replies).

## Configuration

All weights, floors, word lists and launch-policy values are in `promo_finder/config.py`; none are show-specific. The most useful knobs are `FLOORS_HEUR`, `LAUNCH_WINDOW`, `LAUNCH_MAX_AMBER`, `LAUNCH_MIN_PCTL`, `CANDIDATES_PER_EPISODE` and `LLM_MIN_INTERVAL`.

## Troubleshooting

| Symptom | Fix |
|---|---|
| `error: No API key: set GEMINI_API_KEY` | Export the key in the same shell, or use `--provider heuristic` |
| Status `llm_failed_used_heuristic` in `episodes.csv` | Read `llm_calls.jsonl` for the HTTP error or unusable reply. That episode still has valid heuristic picks |
| HTTP 429 on a free tier | The tool backs off and retries. Raise `LLM_MIN_INTERVAL` if it persists. Rerunning is free for cached episodes |
| An episode has 0 promos | Expected when nothing clears the floors. See `rejected_candidates.json` for the reasons |
| Paths with spaces | Quote them: `--input "C:\My Files\episodes"` |
| Episodes renumbered | Unnumbered or duplicate episodes get the next free number, reported in `load_errors` and `run_meta.json` |
| `promos.csv` shows `â€”` in Excel | The file is UTF-8. In Excel use Data > From Text/CSV with File origin "65001: Unicode (UTF-8)", or open it in Google Sheets or VS Code |

## Limitations (observed)

- The scores are judgements made from text alone. A slap's visual punch, the acting and the music are not visible in a script.
- The heuristic favours open conflict. It misses some romance-tension scenes and still over-rates loud, context-heavy late scenes in the quality list. The LLM judge is meant for those calls.
- Spoiler tiers come from position plus wording, so they are deliberately conservative (some Red labels are false alarms) and an editor should check them. Angle labels are word matches too and can be wrong (a possessive "mine" can read as romance).
- On Python 3.10/3.11 a score can differ from 3.12+ in the 4th decimal (float summation changed in 3.12). Picks, timestamps and ranks are the same.
- Scores are editorial estimates, not audience results. Validate with editor acceptance first, then hook rate and hold rate in A/B tests.
- Multi-segment (stitched) promos fit the output format but are not generated.

## Layout

`promo_finder/`:
- `parser.py`
- `candidates.py`: cuts, features, quality score
- `llm_judge.py`: providers, prompt, validation, cache, replay
- `editorial.py`: angle, spoiler tier, rankings
- `pipeline.py`, `validate.py`, `evaluate.py`, `config.py`

Plus `app.py` (web app), `run.py` (CLI), `tests/` (71 tests), `examples/synthetic_show/` (original, messy-format test show) and `docs/AI_USAGE.md`.
