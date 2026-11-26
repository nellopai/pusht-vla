"""Collect all finished evaluations into results/ (committed, so every number in the README is traceable).

Usage: python collect_results.py
Reads outputs/eval/*_n100/ and writes:
  results/summary.csv            one row per evaluation run
  results/runs/<name>.json       overall metrics, per-episode max coverage and success, instruction used
"""

import csv
import json
from pathlib import Path

EVAL_DIR = Path("outputs/eval")
OUT = Path("results")
(OUT / "runs").mkdir(parents=True, exist_ok=True)

rows = []
for run in sorted(EVAL_DIR.glob("*_n100")):
    info_path = run / "eval_info.json"
    if not info_path.exists():
        continue
    info = json.loads(info_path.read_text())
    overall = info["overall"]
    metrics = info["per_task"][0]["metrics"]
    task_file = run / "task_descriptions.json"
    task = next(iter(json.loads(task_file.read_text()).values())) if task_file.exists() else None
    if run.name.startswith("dp_"):
        task = "n/a (no language input)"
    lo, hi = overall["pc_success_ci95"]
    rows.append({
        "run": run.name,
        "n_episodes": overall["n_episodes"],
        "success_pct": round(overall["pc_success"], 1),
        "ci95_low": round(lo, 1),
        "ci95_high": round(hi, 1),
        "avg_max_coverage": round(overall["avg_max_reward"], 3),
        "instruction": task,
    })
    (OUT / "runs" / f"{run.name}.json").write_text(json.dumps({
        "overall": {k: v for k, v in overall.items() if "video" not in k},
        "max_rewards": metrics["max_rewards"],
        "successes": metrics["successes"],
        "instruction": task,
    }, indent=2))

with open(OUT / "summary.csv", "w", newline="") as f:
    writer = csv.DictWriter(f, fieldnames=list(rows[0]))
    writer.writeheader()
    writer.writerows(rows)
for r in rows:
    print(f"{r['run']:50s} {r['success_pct']:5.1f}% [{r['ci95_low']}, {r['ci95_high']}]  cov {r['avg_max_coverage']}")
