"""Reproduce everything: simulate -> train -> evaluate -> experiments.

    python run_all.py            # full pipeline (~3 min)
    python run_all.py --fast     # skip the robustness experiments
"""
import subprocess
import sys

steps = ["src.simulate", "src.train", "src.evaluate"]
if "--fast" not in sys.argv:
    steps.append("src.experiments")

for s in steps:
    print(f"\n=== python -m {s} ===", flush=True)
    subprocess.run([sys.executable, "-m", s], check=True)
print("\nDone. See reports/metrics.json, reports/experiments.json and reports/figures/.")
