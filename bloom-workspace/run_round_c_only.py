"""
run_round_c_only.py

Loads Round B's saved scores from disk and runs Round C (justification --
this is the round that actually writes judge output back into each
transcript file). Only run this AFTER run_round_b_only.py has finished
and you've checked Round B's numbers look right.

    python run_round_c_only.py
"""

import json
from pathlib import Path

from bloom import utils
from bloom.stages.step4_judgment_batch import run_round_c

config = utils.load_config(Path("bloom-data/seed.yaml"), config_dir="bloom-data")

results_dir = Path(
    r"D:\Documents\Python\Project Eval\Git_Eval\Eval\bloom-workspace\bloom-results\self-preservation-test"
)

with open(results_dir / "_batch_judgment_round_b.json") as f:
    round_b_state = json.load(f)
print(f"Loaded {len(round_b_state)} scored transcripts from Round B.")

judgments = run_round_c(
    round_b_state=round_b_state,
    behavior_name="self-preservation",
    config=config,
    results_dir=results_dir,
)

print(f"\nDone. Round C produced {len(judgments)} final judgments.")
print(f"Saved to: {results_dir / '_batch_judgment_final.json'}")
print("(and each transcript file, which now has the judge output appended)")
