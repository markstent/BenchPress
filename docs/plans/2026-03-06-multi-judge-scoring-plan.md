# Multi-Judge Scoring Implementation Plan

> **For Claude:** REQUIRED SUB-SKILL: Use superpowers:executing-plans to implement this plan task-by-task.

**Goal:** Replace single-judge scoring with multi-judge evaluation using GPT-4.1, Claude Sonnet 4, and Gemini 2.5 Pro, averaging their scores and handling self-exclusion.

**Architecture:** Config changes from `judge:` (single) to `judges:` (array). Each run entry stores per-judge scores in a `judge_scores` dict with timestamps. A migration command converts existing results. Dashboard shows averaged scores with expandable per-judge breakdown.

**Tech Stack:** Python 3, PyYAML, existing provider abstraction, Chart.js dashboard

**Design doc:** `docs/plans/2026-03-06-multi-judge-scoring-design.md`

---

### Task 1: Migration Command - `migrate-judges`

Start here because all downstream work depends on the new data format. Existing results need to be migrated before any new code can read them.

**Files:**
- Modify: `run.py` (add `cmd_migrate_judges` function and CLI subparser)

**Step 1: Write the migration function**

Add to `run.py` after the `cmd_deepeval` function (around line 736):

```python
def cmd_migrate_judges(args):
    """Migrate results from old single-judge format to new multi-judge format."""
    import shutil

    model_names = list_evaluated_models()
    if not model_names:
        print("No models to migrate.")
        return

    # Backup
    backup_dir = os.path.join(RESULTS_DIR, "backup")
    os.makedirs(backup_dir, exist_ok=True)
    for name in model_names:
        src = model_path(name)
        dst = os.path.join(backup_dir, f"{name}.json")
        if not os.path.exists(dst):
            shutil.copy2(src, dst)
    print(f"  Backed up {len(model_names)} files to {backup_dir}/")

    total_migrated = 0
    total_skipped = 0

    for name in model_names:
        model_data = load_model_results(name)
        changed = False

        for pid, runs in model_data.get("runs", {}).items():
            for run in runs:
                # Skip if already migrated
                if "judge_scores" in run:
                    total_skipped += 1
                    continue

                old_score = run.get("judge_score")
                old_rationale = run.get("judge_rationale", "")
                old_model = run.get("judge_model")

                judge_scores = {}
                if old_model and old_score is not None:
                    judge_scores[old_model] = {
                        "score": old_score,
                        "rationale": old_rationale,
                        "judged_at": None,
                    }

                # Compute average
                valid_scores = [v["score"] for v in judge_scores.values() if v["score"] is not None]
                run["judge_scores"] = judge_scores
                run["judge_score_avg"] = round(sum(valid_scores) / len(valid_scores), 2) if valid_scores else None
                run["judge_count"] = len(valid_scores)

                # Remove old fields
                run.pop("judge_score", None)
                run.pop("judge_rationale", None)
                run.pop("judge_model", None)

                changed = True
                total_migrated += 1

        if changed:
            save_model_results(name, model_data)
            print(f"  {name}: migrated")
        else:
            print(f"  {name}: already migrated")

    print(f"\n  Done: {total_migrated} runs migrated, {total_skipped} already migrated")
```

**Step 2: Add CLI subparser**

In the `main()` function, after the `dashboard` subparser (around line 788), add:

```python
p = sub.add_parser("migrate-judges", help="Migrate results from single-judge to multi-judge format")
```

And add to the `cmds` dict:

```python
cmds = {..., "migrate-judges": cmd_migrate_judges, ...}
```

**Step 3: Run migration on existing results**

Run: `python run.py migrate-judges`

Expected: All 44 result files backed up to `results/backup/` and migrated to new format. Each run entry now has `judge_scores`, `judge_score_avg`, `judge_count` fields. Old `judge_score`, `judge_rationale`, `judge_model` fields removed.

**Step 4: Verify migration**

Run: `python3 -c "import json; d=json.load(open('results/claude-opus-4.6.json')); r=list(d['runs'].values())[0][-1]; print(json.dumps({k:v for k,v in r.items() if 'judge' in k}, indent=2))"`

Expected: Output shows `judge_scores` dict, `judge_score_avg`, `judge_count`. No `judge_score`, `judge_rationale`, or `judge_model` fields.

**Step 5: Commit**

```bash
git add run.py
git commit -m "Add migrate-judges command for multi-judge data format"
```

---

### Task 2: Update Config Format

**Files:**
- Modify: `config.yaml` (replace `judge:` with `judges:`)
- Modify: `config.example.yaml` (same change if it exists, or `config.yaml` only)

**Step 1: Update config.yaml**

Replace the `judge:` block (around line 922-927):

```yaml
# Old:
judge:
  model: gpt-4.1
  params:
    max_tokens: 1024
    temperature: 0

# New:
judges:
  - model: gpt-4.1
    params:
      max_tokens: 1024
      temperature: 0
  - model: claude-sonnet-4.6
    params:
      max_tokens: 1024
      temperature: 0
  - model: gemini-2.5-pro
    params:
      max_tokens: 1024
      temperature: 0
```

Note: Use `claude-sonnet-4.6` since that's the model name in the existing config (not `claude-sonnet-4`). Verify the exact model names exist in the `models:` section. For Gemini, check if `gemini-2.5-pro` exists in config - if not, it needs to be added to the `models:` section too.

**Step 2: Verify all judge models exist in models section**

Run: `python3 -c "import yaml; c=yaml.safe_load(open('config.yaml')); judges=[j['model'] for j in c['judges']]; models=list(c['models'].keys()); missing=[j for j in judges if j not in models]; print(f'Missing: {missing}' if missing else 'All judge models found in config')"`

Expected: "All judge models found in config" (or fix any missing models)

**Step 3: Commit**

```bash
git add config.yaml
git commit -m "Update config format from single judge to judges array"
```

---

### Task 3: Update `run.py` - Config Reading and Eval Command

This is the core change. Update all code that reads `config["judge"]` to read `config["judges"]` (the array) and loop through multiple judges.

**Files:**
- Modify: `run.py` lines 166-223 (eval judge setup and scoring loop)

**Step 1: Update judge setup in `cmd_eval` (lines 166-183)**

Replace the judge setup block with:

```python
    # Set up LLM judges
    judges_cfg = config.get("judges", [])
    judge_providers = {}
    for jcfg in judges_cfg:
        jname = jcfg.get("model")
        if not jname or jname not in models_cfg:
            print(f"  Warning: judge model '{jname}' not found in config, skipping")
            continue
        if jname == model_name:
            print(f"  Skipping judge {jname} (cannot self-judge)")
            continue
        try:
            jprov = get_provider(models_cfg[jname])
            judge_providers[jname] = {
                "provider": jprov,
                "params": jcfg.get("params", {}),
            }
        except ValueError as e:
            print(f"  Warning: could not init judge provider '{jname}': {e}")
```

**Step 2: Update eval header print (lines 187-189)**

Replace:
```python
    if judge_provider:
        print(f"  Judge: {judge_model_name} ({models_cfg[judge_model_name]['model']})")
```

With:
```python
    if judge_providers:
        judge_names = ", ".join(judge_providers.keys())
        print(f"  Judges: {judge_names}")
```

**Step 3: Update entry creation (lines 202-213)**

Replace the entry dict's judge fields:
```python
            entry = {
                "timestamp": datetime.now().isoformat(),
                "api_model": model_cfg["model"],
                "content": content,
                "latency_s": round(latency, 2),
                "input_tokens": usage.get("input_tokens"),
                "output_tokens": usage.get("output_tokens"),
                "auto_checks": auto,
                "judge_scores": {},
                "judge_score_avg": None,
                "judge_count": 0,
            }
```

**Step 4: Update judge scoring loop (lines 218-223)**

Replace:
```python
            if judge_provider:
                jr = judge_response(judge_provider, judge_params, pmeta, content, auto)
                entry["judge_score"] = jr["judge_score"]
                entry["judge_rationale"] = jr["judge_rationale"]
                score_str = f"{jr['judge_score']}/5" if jr["judge_score"] else "failed"
                print(f"    Judge: {score_str}")
```

With:
```python
            if judge_providers:
                for jname, jinfo in judge_providers.items():
                    jr = judge_response(jinfo["provider"], jinfo["params"], pmeta, content, auto)
                    entry["judge_scores"][jname] = {
                        "score": jr["judge_score"],
                        "rationale": jr["judge_rationale"],
                        "judged_at": datetime.now().isoformat(),
                    }
                    score_str = f"{jr['judge_score']}/5" if jr["judge_score"] else "failed"
                    print(f"    Judge ({jname}): {score_str}")

                # Compute average
                valid = [v["score"] for v in entry["judge_scores"].values() if v["score"] is not None]
                entry["judge_score_avg"] = round(sum(valid) / len(valid), 2) if valid else None
                entry["judge_count"] = len(valid)
```

**Step 5: Update error entry (lines 242-252)**

Replace judge fields in the error entry:
```python
                "judge_scores": {},
                "judge_score_avg": None,
                "judge_count": 0,
```

**Step 6: Update summary stats (lines 266-274)**

Replace:
```python
    judged = sum(
        1 for p in prompts
        if latest_run(model_data, p["id"]).get("judge_score") is not None
    )
    print(f"\n  Done: {len(prompts)} prompts, {flagged} auto-flagged, {judged} judge-scored")
```

With:
```python
    judged = sum(
        1 for p in prompts
        if latest_run(model_data, p["id"]).get("judge_score_avg") is not None
    )
    print(f"\n  Done: {len(prompts)} prompts, {flagged} auto-flagged, {judged} judge-scored")
```

**Step 7: Test with a dry run (read-only verification)**

Run: `python run.py eval --help`

Expected: No import errors, help text displays correctly.

**Step 8: Commit**

```bash
git add run.py
git commit -m "Update eval command for multi-judge scoring loop"
```

---

### Task 4: Update `run.py` - Rejudge Command

**Files:**
- Modify: `run.py` lines 484-611 (`cmd_rejudge`)

**Step 1: Rewrite `cmd_rejudge`**

Replace the entire `cmd_rejudge` function with:

```python
def cmd_rejudge(args):
    config = load_config(args.config)
    models_cfg = config.get("models", {})
    judges_cfg = config.get("judges", [])

    if not judges_cfg:
        print("No judges configured in config.yaml")
        sys.exit(1)

    # Filter to specific judge if --judge flag provided
    if args.judge:
        judges_cfg = [j for j in judges_cfg if j["model"] == args.judge]
        if not judges_cfg:
            print(f"Judge '{args.judge}' not found in config.yaml judges list")
            sys.exit(1)

    # Initialize judge providers
    judge_providers = {}
    for jcfg in judges_cfg:
        jname = jcfg.get("model")
        if not jname or jname not in models_cfg:
            print(f"  Warning: judge model '{jname}' not found in config, skipping")
            continue
        try:
            jprov = get_provider(models_cfg[jname])
            judge_providers[jname] = {
                "provider": jprov,
                "params": jcfg.get("params", {}),
            }
        except ValueError as e:
            print(f"  Warning: could not init judge provider '{jname}': {e}")

    if not judge_providers:
        print("No valid judge providers could be initialized.")
        sys.exit(1)

    # Determine which models to rejudge
    if args.models:
        model_names = args.models
    else:
        model_names = list_evaluated_models()

    if not model_names:
        print("No models to rejudge.")
        return

    prompts = load_eval()
    prompts_by_id = {p["id"]: p for p in prompts}
    delay = config.get("eval", {}).get("delay_between_calls", 1.0)

    print(f"\n{'='*60}")
    print(f"  Rejudging with: {', '.join(judge_providers.keys())}")
    print(f"  Models: {len(model_names)}")
    print(f"  Force: {args.force}")
    print(f"{'='*60}\n")

    total_judged = 0
    total_skipped = 0
    total_errors = 0

    for model_name in model_names:
        model_data = load_model_results(model_name)
        if not model_data["runs"]:
            print(f"  Skipping {model_name} (no results)")
            continue

        # Determine which judges apply (exclude self)
        applicable_judges = {k: v for k, v in judge_providers.items() if k != model_name}
        if not applicable_judges:
            print(f"  Skipping {model_name} (all judges excluded - self-judge)")
            continue

        pids = list(model_data["runs"].keys())
        to_judge = []

        for pid in pids:
            run = latest_run(model_data, pid)
            if not run or run.get("error"):
                continue
            existing = run.get("judge_scores", {})
            # Check if all applicable judges already scored
            if not args.force:
                missing = [j for j in applicable_judges if j not in existing or existing[j].get("score") is None]
                if not missing:
                    total_skipped += 1
                    continue
            to_judge.append(pid)

        if not to_judge:
            print(f"  {model_name}: all {len(pids)} prompts fully judged")
            continue

        print(f"  {model_name}: rejudging {len(to_judge)}/{len(pids)} prompts with {len(applicable_judges)} judges...")

        for i, pid in enumerate(to_judge, 1):
            run = latest_run(model_data, pid)
            pmeta = prompts_by_id.get(pid)
            if not pmeta:
                print(f"    [{i}/{len(to_judge)}] {pid} - prompt not found, skipping")
                continue

            print(f"    [{i}/{len(to_judge)}] {pid}...", end=" ", flush=True)

            auto_checks = run.get("auto_checks", {"flags": [], "auto_scores": {}, "passed": True})
            judge_scores = run.get("judge_scores", {})

            for jname, jinfo in applicable_judges.items():
                # Skip if already scored and not forcing
                if not args.force and jname in judge_scores and judge_scores[jname].get("score") is not None:
                    continue
                try:
                    jr = judge_response(jinfo["provider"], jinfo["params"], pmeta, run["content"], auto_checks)
                    judge_scores[jname] = {
                        "score": jr["judge_score"],
                        "rationale": jr["judge_rationale"],
                        "judged_at": datetime.now().isoformat(),
                    }
                    score_str = f"{jr['judge_score']}/5" if jr["judge_score"] else "failed"
                    print(f"{jname}={score_str}", end=" ", flush=True)
                    if jr["judge_score"] is not None:
                        total_judged += 1
                    else:
                        total_errors += 1
                except Exception as e:
                    print(f"{jname}=error", end=" ", flush=True)
                    total_errors += 1
                time.sleep(delay)

            # Update the latest run entry in-place
            valid = [v["score"] for v in judge_scores.values() if v["score"] is not None]
            run["judge_scores"] = judge_scores
            run["judge_score_avg"] = round(sum(valid) / len(valid), 2) if valid else None
            run["judge_count"] = len(valid)

            print()  # newline after judge scores

            try:
                save_model_results(model_name, model_data)
            except Exception as e:
                print(f"    Save failed: {e}")

    print(f"\n  Done: {total_judged} judged, {total_skipped} skipped, {total_errors} errors")

    path = generate_dashboard()
    if path:
        print(f"  Dashboard updated: {path}")
```

**Step 2: Add `--judge` CLI argument**

Update the rejudge subparser (around line 777-779):

```python
p = sub.add_parser("rejudge", help="Re-score existing responses with configured judges")
p.add_argument("models", nargs="*", help="Models to rejudge (default: all)")
p.add_argument("--judge", help="Only rejudge with this specific judge model")
p.add_argument("--force", action="store_true", help="Rejudge even if already scored")
```

**Step 3: Test**

Run: `python run.py rejudge --help`

Expected: Help shows `--judge` flag.

**Step 4: Commit**

```bash
git add run.py
git commit -m "Update rejudge command for multi-judge support with --judge flag"
```

---

### Task 5: Update `run.py` - Compare Command

**Files:**
- Modify: `run.py` lines 286-420 (`cmd_compare`)

**Step 1: Update score reading in compare**

In `cmd_compare`, replace all references to `run.get("judge_score")` with `run.get("judge_score_avg")`. There are 3 occurrences in the function:

Line ~331:
```python
# Old:
if run.get("judge_score") is not None:
    scores.append(run["judge_score"])
# New:
if run.get("judge_score_avg") is not None:
    scores.append(run["judge_score_avg"])
```

Lines ~386-388 (category breakdown):
```python
# Old:
sc = [
    latest_run(data, pid).get("judge_score")
    for pid in cat_pids
    if latest_run(data, pid) and latest_run(data, pid).get("judge_score") is not None
]
# New:
sc = [
    latest_run(data, pid).get("judge_score_avg")
    for pid in cat_pids
    if latest_run(data, pid) and latest_run(data, pid).get("judge_score_avg") is not None
]
```

**Step 2: Update `cmd_models` to show judge count**

In `cmd_models` (line ~432):
```python
# Old:
scored = sum(1 for rs in data["runs"].values() if rs and rs[-1].get("judge_score") is not None)
# New:
scored = sum(1 for rs in data["runs"].values() if rs and rs[-1].get("judge_score_avg") is not None)
```

**Step 3: Update `cmd_deepeval` judge exclusion (line ~630-631)**

Replace:
```python
    judge_model = config.get("judge", {}).get("model")
    model_names = [m for m in model_names if m != judge_model]
```

With:
```python
    # Don't exclude judge models from deepeval scoring - multi-judge doesn't need this
```

(In multi-judge, we no longer exclude judge models from deepeval since they may only be judges for other models.)

**Step 4: Update `_save_comparison_md` (line ~473)**

Replace `run.get("judge_score", "-")` with `run.get("judge_score_avg", "-")`.

**Step 5: Test**

Run: `python run.py compare --help`

Expected: No errors.

**Step 6: Commit**

```bash
git add run.py
git commit -m "Update compare and models commands for multi-judge score format"
```

---

### Task 6: Update `scripts/dashboard.py` - Data Layer

The dashboard's `compute_stats` and `generate_dashboard` functions need to read the new `judge_score_avg` field and pass multi-judge data through to the frontend.

**Files:**
- Modify: `scripts/dashboard.py`

**Step 1: Update `compute_stats` (line 52 onwards)**

Replace all `run.get("judge_score")` with `run.get("judge_score_avg")`. There are ~6 occurrences in `compute_stats`:

- Line 77-78: `if run.get("judge_score") is not None: scores.append(run["judge_score"])` -> `if run.get("judge_score_avg") is not None: scores.append(run["judge_score_avg"])`
- Line 105-107: category scores - same replacement
- Line 135-137: difficulty scores - same replacement
- Line 162: divergence calc `js, da = run.get("judge_score"), ...` -> `js, da = run.get("judge_score_avg"), ...`
- Line 270: prompt results `"judge_score": run.get("judge_score")` -> `"judge_score": run.get("judge_score_avg")`
- Line 278: error prompt results - stays `None`, no change needed

**Step 2: Update `generate_dashboard` function (line ~3514-3523)**

Replace:
```python
    judge_model = config.get("judge", {}).get("model")
    if judge_model and judge_model in models:
        del models[judge_model]
```

With:
```python
    judges_cfg = config.get("judges", [])
    judge_models = [j["model"] for j in judges_cfg]
    # Don't exclude judge models from leaderboard - in multi-judge mode
    # models only self-exclude, they can still appear on the board
```

**Step 3: Update stats dict**

In the `compute_stats` return dict (line ~293), replace:
```python
"judge_model": judge_model,
```

With:
```python
"judge_models": judge_models,
```

Also add per-judge data to each leaderboard entry. After the main stats computation, add judge breakdown per model. In `compute_stats`, inside the model loop, after computing `avg_divergence` (around line 165), add:

```python
        # Per-judge score breakdown
        judge_breakdown = {}
        for pid in pids:
            run = runs_cache[pid]
            if not run or run.get("error"):
                continue
            for jname, jdata in run.get("judge_scores", {}).items():
                if jname not in judge_breakdown:
                    judge_breakdown[jname] = []
                if jdata.get("score") is not None:
                    judge_breakdown[jname].append(jdata["score"])

        judge_averages = {}
        for jname, jscores in judge_breakdown.items():
            judge_averages[jname] = round(sum(jscores) / len(jscores), 2) if jscores else None

        # Judge agreement (std dev of per-judge averages)
        ja_values = [v for v in judge_averages.values() if v is not None]
        if len(ja_values) >= 2:
            mean_ja = sum(ja_values) / len(ja_values)
            judge_std_dev = round((sum((x - mean_ja) ** 2 for x in ja_values) / len(ja_values)) ** 0.5, 2)
        else:
            judge_std_dev = None
```

And add to the leaderboard entry dict:
```python
            "judge_averages": judge_averages,
            "judge_std_dev": judge_std_dev,
```

Also add per-judge data to prompt_results. In the prompt results section (line ~269), expand the model entry:
```python
                pr["models"][name] = {
                    "judge_score": run.get("judge_score_avg"),
                    "judge_scores": run.get("judge_scores", {}),
                    "judge_count": run.get("judge_count", 0),
                    "deepeval_avg": run.get("deepeval_avg"),
                    "latency_s": round(run.get("latency_s", 0), 1),
                    "error": False,
                    "flags": run.get("auto_checks", {}).get("flags", []),
                }
```

**Step 4: Update the header meta line in all HTML templates**

There are 5 occurrences of the judge model display in meta lines (lines 727, 1691, 2104, 2855, 3350). Replace each:

```python
# Old:
{f' &middot; Judged by {stats["judge_model"]}' if stats.get("judge_model") else ''}

# New:
{f' &middot; Judges: {", ".join(stats["judge_models"])}' if stats.get("judge_models") else ''}
```

**Step 5: Commit**

```bash
git add scripts/dashboard.py
git commit -m "Update dashboard data layer for multi-judge scores"
```

---

### Task 7: Update Dashboard HTML/JS - Leaderboard Display

Update the leaderboard table to show averaged judge scores with per-judge breakdown.

**Files:**
- Modify: `scripts/dashboard.py` (the HTML template sections)

**Step 1: Find and read the leaderboard table HTML**

Search for the table header row in the dashboard HTML to find where judge score is displayed. Look for column headers containing "Judge" or "Score".

**Step 2: Add judge count tooltip to score display**

In the JavaScript that renders leaderboard rows, update the judge score cell to include:
- The averaged score
- A tooltip showing judge count (e.g., "2 of 3 judges")
- A colored agreement indicator dot:
  - Green dot: std_dev < 0.3
  - Yellow dot: std_dev 0.3-0.7
  - Red dot: std_dev > 0.7

**Step 3: Add expandable row for per-judge breakdown**

After each leaderboard row, add a hidden detail row that shows:
- A small table with columns: Judge Model | Score | Judged At
- Populated from `judge_averages` data
- Toggle visibility on row click

**Step 4: Update the per-prompt detail view**

In the prompts page, update the model score display to show per-judge scores. For each model's result on a prompt, show each judge's score and rationale in a stacked layout.

**Step 5: Update the JavaScript sorting**

Find the JS that sorts by `judge_score` (lines ~3421-3426) and update to use `judge_score` which now represents the average (from `judge_score_avg` in Python data).

**Step 6: Visual verification**

Run: `python run.py dashboard --open`

Verify:
- Leaderboard shows averaged scores
- Judge count visible on hover
- Expandable rows show per-judge breakdown
- No layout breakage

**Step 7: Commit**

```bash
git add scripts/dashboard.py
git commit -m "Update dashboard UI for multi-judge display with agreement indicators"
```

---

### Task 8: Run Migration and Verify End-to-End

**Files:**
- No new files - verification only

**Step 1: Run the migration**

Run: `python run.py migrate-judges`

Expected: All result files migrated, backups created.

**Step 2: Verify a migrated file**

Run: `python3 -c "import json; d=json.load(open('results/claude-opus-4.6.json')); r=list(d['runs'].values())[0][-1]; print(json.dumps({k:v for k,v in r.items() if 'judge' in k}, indent=2))"`

Expected: New format with `judge_scores`, `judge_score_avg`, `judge_count`.

**Step 3: Verify compare still works**

Run: `python run.py compare`

Expected: Leaderboard displays with averaged scores from migrated data (single judge initially).

**Step 4: Verify dashboard generates**

Run: `python run.py dashboard --open`

Expected: Dashboard renders without errors, shows judges in header.

**Step 5: Commit migrated results**

```bash
git add results/
git commit -m "Migrate result files to multi-judge format"
```

---

### Task 9: Update `config.example.yaml`

**Files:**
- Modify: `config.example.yaml` (if it exists)

**Step 1: Check if config.example.yaml exists and update**

If it exists, update it to match the new `judges:` format. Remove `judge:` block, add `judges:` array with the three default judges.

**Step 2: Commit**

```bash
git add config.example.yaml
git commit -m "Update config.example.yaml for multi-judge format"
```

---

### Task 10: Final Integration Test

**Step 1: Run eval with a small test (1 prompt, cheap model)**

Run: `python run.py eval claude-haiku-4.5 --ids C01 --rerun`

Expected: Model is evaluated, all configured judges that aren't claude-haiku-4.5 score the response. Output shows per-judge scores and computed average.

**Step 2: Run rejudge with single judge**

Run: `python run.py rejudge claude-haiku-4.5 --judge gpt-4.1 --force --ids C01`

(Note: if rejudge doesn't support --ids yet, just test with the model. The --force flag ensures it re-runs.)

**Step 3: Verify dashboard**

Run: `python run.py dashboard --open`

Expected: Dashboard shows multi-judge data for claude-haiku-4.5 with agreement indicators.

**Step 4: Final commit**

```bash
git add -A
git commit -m "Multi-judge scoring feature complete"
```
