"""
run_round_a1_only.py

Submits Round A1 (shared injection scores) and stops -- does NOT run
Round A2. Run this first, wait for it to say "Done", then run
run_round_a2_only.py separately.

    python run_round_a1_only.py
"""

import json
from pathlib import Path

from bloom import utils
from bloom.stages.step4_judgment_batch import run_round_a1

config = utils.load_config(Path("bloom-data/seed.yaml"), config_dir="bloom-data")
prompts = utils.load_configurable_prompts(config)
understanding = utils.load_understanding_results("self-preservation")

with open("bloom-results/self-preservation/ideation_combined.json") as f:
    ideation = json.load(f)

results_dir = Path(
    r"D:\Documents\Python\Project Eval\Git_Eval\Eval\bloom-workspace\bloom-results\self-preservation-test"
)

shared_injection_scores = run_round_a1(
    behavior_name="self-preservation",
    behavior_understanding=understanding["understanding"],
    scientific_motivation=understanding["scientific_motivation"],
    config=config, prompts=prompts, target_model_name="Claude Sonnet 4",
    results_dir=results_dir, ideation_results=ideation,
)

print(f"\nDone. Round A1 produced {len(shared_injection_scores)} scenario scores.")
print(f"Saved to: {results_dir / '_batch_judgment_round_a1.json'}")
print("Now run: python run_round_a2_only.py")
