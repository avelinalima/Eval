"""
run_round_b_only.py

Loads Round A2's saved judgments from disk and runs Round B (scoring:
behavior_presence + quality scores). Only run this AFTER
run_round_a2_only.py has finished and printed "Done".

    python run_round_b_only.py
"""

import json
from pathlib import Path

from bloom import utils
from bloom.stages.step4_judgment_batch import run_round_b

config = utils.load_config(Path("bloom-data/seed.yaml"), config_dir="bloom-data")

results_dir = Path(
    r"D:\Documents\Python\Project Eval\Git_Eval\Eval\bloom-workspace\bloom-results\self-preservation-test"
)

with open(results_dir / "_batch_judgment_round_a2.json") as f:
    round_a_state = json.load(f)
print(f"Loaded {len(round_a_state)} initial judgments from Round A2.")

round_b_state = run_round_b(
    round_a_state=round_a_state,
    behavior_name="self-preservation",
    config=config,
    results_dir=results_dir,
)

print(f"\nDone. Round B scored {len(round_b_state)} transcripts.")
print(f"Saved to: {results_dir / '_batch_judgment_round_b.json'}")
print("Now run: python run_round_c_only.py")
