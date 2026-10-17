"""
run_round_d_only.py

Loads Round C's saved final judgments from disk and runs Round D
(meta-judgment -- one call across the whole suite). Only run this AFTER
run_round_c_only.py has finished and you've checked Round C's numbers
look right.

    python run_round_d_only.py
"""

import json
from pathlib import Path

from bloom import utils
from bloom.stages.step4_judgment_batch import run_round_d

config = utils.load_config(Path("bloom-data/seed.yaml"), config_dir="bloom-data")
prompts = utils.load_configurable_prompts(config)

results_dir = Path(
    r"D:\Documents\Python\Project Eval\Git_Eval\Eval\bloom-workspace\bloom-results\self-preservation-test"
)

with open(results_dir / "_batch_judgment_final.json") as f:
    round_c_judgments = json.load(f)
print(f"Loaded {len(round_c_judgments)} final judgments from Round C.")

metajudgment_result = run_round_d(
    round_c_judgments=round_c_judgments,
    behavior_name="self-preservation",
    config=config,
    prompts=prompts,
    results_dir=results_dir,
)

if metajudgment_result is None:
    print("\nRound D skipped or failed -- see the warning above for why.")
else:
    print("\nDone. Round D meta-judgment scores:")
    for key, value in metajudgment_result["metajudgment_scores"].items():
        print(f"  {key}: {value}")
    print(f"Saved to: {results_dir / '_batch_judgment_round_d.json'}")
