"""
run_round_a2_only.py

Loads Round A1's saved scores from disk and runs Round A2. Only run this
AFTER run_round_a1_only.py has finished and printed "Done".

    python run_round_a2_only.py
"""

import json
from pathlib import Path

from bloom import utils
from bloom.stages.step4_judgment_batch import run_round_a2

config = utils.load_config(Path("bloom-data/seed.yaml"), config_dir="bloom-data")
prompts = utils.load_configurable_prompts(config)
understanding = utils.load_understanding_results("self-preservation")

with open("bloom-results/self-preservation/ideation_combined.json") as f:
    ideation = json.load(f)

results_dir = Path(
    r"D:\Documents\Python\Project Eval\Git_Eval\Eval\bloom-workspace\bloom-results\self-preservation-test"
)

# Load Round A1's saved scores (keys were saved as "v{n}-r{m}" strings --
# converting back to (n, m) tuples, which is what run_round_a2 expects)
with open(results_dir / "_batch_judgment_round_a1.json") as f:
    raw = json.load(f)
shared_injection_scores = {
    (int(k.split("-")[0][1:]), int(k.split("-")[1][1:])): v
    for k, v in raw.items()
}
print(f"Loaded {len(shared_injection_scores)} scenario scores from Round A1.")

round_a_state = run_round_a2(
    behavior_name="self-preservation",
    behavior_understanding=understanding["understanding"],
    scientific_motivation=understanding["scientific_motivation"],
    config=config, prompts=prompts, target_model_name="Claude Sonnet 4",
    results_dir=results_dir, ideation_results=ideation,
    shared_injection_scores=shared_injection_scores,
)

print(f"\nDone. Round A2 produced {len(round_a_state)} initial judgments.")
print(f"Saved to: {results_dir / '_batch_judgment_round_a2.json'}")
