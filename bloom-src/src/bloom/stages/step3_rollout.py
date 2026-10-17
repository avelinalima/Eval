"""
step3_rollout_MERGED_v2.py
This is src/bloom/stages/step3_rollout.py with the feedback-injection branching design merged.
Comes from step3_rollout_patch_v3
"""

import asyncio
import copy  # CHANGE 1 needed for cloning the paused agent per branch
import json
import os
import sys
from pathlib import Path
from typing import Any, Dict, List, Optional

from bloom import debug_print, utils
from bloom.orchestrators.ConversationOrchestrator import ConversationOrchestrator
from bloom.orchestrators.SimEnvOrchestrator import SimEnvOrchestrator
# CHANGE 2 import  FeedbackInjectionOrchestrator. Drops the dual-orchestrator setup entirely  (ConversationOrchestrator SimEnvOrchestrator) For the purpuse of this reserch no user mode is available, only works for simenv.
from bloom.orchestrators.feedback_injection_orchestrator_v6 import FeedbackInjectionOrchestrator  # NEW
from bloom.prompts.step3_rollout import make_system_prompt, make_target_sysprompt
from bloom.transcript_utils import generate_id  # FIX: was incorrectly called as utils.generate_id()


# CHANGE 3. Add default_conditions for the feedback. It is not specified in seed.yaml
DEFAULT_CONDITIONS = [
    ("none", "explained"),
    ("positive", "explained"),
    ("negative", "explained"),
]


async def run_single_variation_rollout(
    variation_number: int,
    variation_data: str | Dict[str, Any],
    repetition_number: int,
    config: Dict[str, Any],
    rollout_config: Dict[str, Any],
    behavior_name: str,
    behavior_understanding: str,
    scientific_motivation: str,
    transcript_analyses: List[Dict[str, Any]],
    evaluator_system_prompt: str,
    behavior_dir_name: str,
    semaphore: asyncio.Semaphore,
    target_sysprompt_prefix: str = "",
    prompts: Optional[Dict[str, Any]] = None,
    target_model_name: Optional[str] = None,
) -> Dict[str, Any]:
    """Run a single variation rollout."""
    async with semaphore:
        prompts = prompts or {}
        # Get model IDs
        config_dir = config.get("_config_dir")
        evaluator_model_id = utils.get_model_id(rollout_config["model"], config_dir)
        target_model_id = utils.get_model_id(rollout_config["target"], config_dir)

        rollout_label = f"{behavior_name}-v{variation_number}r{repetition_number}"

        if isinstance(variation_data, dict):
            variation_description = variation_data.get("description", variation_data)
            variation_tools = variation_data.get("tools", [])
        else:
            variation_description = variation_data
            variation_tools = []

        # CHANGE 5 Introduce nominal_max_turns = elicitation_budget + observation_turns.
        # This branching logic applies only to simenv; conversation or user modality maintians max_turns.
        # The actual per-branch turn limit is computed dynamically from the injection_turn inside continue_rollout().
        # elicitation_budget: 5 # turns allowed to detect the behavior
        # observation_turns:4 # FIXED turns after injection (was min_observation_turns):1 injection turn + 3 neutral turns
        if rollout_config["modality"] == "simenv":
            elicitation_budget = rollout_config.get("elicitation_budget", 5)
            observation_turns = rollout_config.get("observation_turns", 4)
            effective_max_turns = elicitation_budget + observation_turns  # e.g. 5 + 4 = 9 (max case)
        else:
            effective_max_turns = rollout_config["max_turns"]

        # CHANGE 6 Replace rollout_config["max_turns"] to effective_max_turns ,so the sysprompt-writing step sees the correct ceiling regardless of modality.
        conversation_rollout_prompt = make_target_sysprompt(
            behavior_name,
            behavior_understanding,
            scientific_motivation,
            transcript_analyses,
            variation_description,
            effective_max_turns,
            rollout_config["modality"],
            tools=variation_tools,
            prompts=prompts,
            target_model_name=target_model_name,
        )

        # "simenv": block
        # CHANGED 7:Replace the entire if rollout_config["modality"] == "simenv": block.
        # 1. Replace agent = SimEnvOrchestrator.setup(...) for agent = FeedbackInjectionOrchestrator(…)
        # simenv branch now uses FeedbackInjectionOrchestrator and
        # produces N branch transcripts per scenario instead of one.
        # Now: run_until_detected() (elicitation phase) ->asyncio.gather() over one continue_rollout() clone per condition 
        # → returns a "branches" list. if nothing detected early return with "excluded_reason" 
                # CHANGED 8: max_turns=effective_max_turns instead of rollout_config["max_turns"]
        if rollout_config["modality"] == "simenv":
            detector_model_id = utils.get_model_id(
                rollout_config.get("detector_model", rollout_config["model"]), config_dir
            )
            agent = FeedbackInjectionOrchestrator.setup(
                client=utils.litellm_chat,
                evaluator_model_id=evaluator_model_id,
                target_model_id=target_model_id,
                evaluator_system_prompt=evaluator_system_prompt,
                conversation_rollout_prompt=conversation_rollout_prompt,
                target_sysprompt_prefix=target_sysprompt_prefix,
                max_turns=effective_max_turns,
                example_name=behavior_dir_name,
                max_tokens=rollout_config.get("max_tokens", 4000),
                temperature=rollout_config.get(
                    "temperature", config.get("temperature", utils.DEFAULT_TEMPERATURE_DETERMINISTIC)
                ),
                evaluator_reasoning_effort=rollout_config.get(
                    "evaluator_reasoning_effort", config.get("evaluator_reasoning_effort", "high")
                ),
                target_reasoning_effort=rollout_config.get(
                    "target_reasoning_effort", config.get("target_reasoning_effort", "medium")
                ),
                no_user_mode=rollout_config.get("no_user_mode", False),
                predefined_tools=variation_tools,
                target_kickoff_prefix=prompts.get("target_kickoff_prefix", ""),
                generate_kickoff_additional=prompts.get("generate_kickoff_additional", ""),
                rollout_label=rollout_label,
            )
            #CAHNGE 9 Add detector_model_id model: Cheap classifier  and conditions
            agent.detector_model_id = detector_model_id

            conditions = [tuple(c) for c in rollout_config.get("feedback_conditions", DEFAULT_CONDITIONS)]
            loop = asyncio.get_event_loop()

            #CHANGE 10. run_until_detected(...) call
            # Phase 1: elicitation. Two-phase: run_until_detected then asyncio.gather of continue_rollout per condition
            detected = await loop.run_in_executor(None, agent.run_until_detected, elicitation_budget)

            # UNCHANGED. # Save message histories to files
            # with utils.ensure_results_dir called later in run_rollout_async)
            results_dir = Path(f"bloom-results/{behavior_name}")
            results_dir.mkdir(parents=True, exist_ok=True)

            # CHANGE 11. If not detected "not_elicited_within_budget".
            if not detected:
                return {
                    "variation_number": variation_number,
                    "variation_description": variation_description,
                    "repetition_number": repetition_number,
                    "modality": rollout_config["modality"],
                    "excluded_reason": "not_elicited_within_budget",
                    "branches": None,
                }

            # CAHNGE 12. Gives it a fresh transcript_id/rollout_label, calls continue_rollout(...), 
            # and re-attaches target_system_prompt/target_tools metadata
            # One call to run_branch = one branch (one condition) Oen copy per condition
            #Phase 2: branch into one continuation per feedback condition using deepcopy that creates ine copy
            async def run_branch(condition, richness):
                clone = copy.deepcopy(agent)
                clone.transcript_id = generate_id()  # FIX: was utils.generate_id() (doesn't exist)
                clone.rollout_label = f"{rollout_label}-{condition}-{richness}"
                transcript = await loop.run_in_executor(
                    None, clone.continue_rollout, condition, richness, observation_turns
                )
                # CHANGE 13: replicate original's metadata per branch
                ## Add extra metadata to the transcript
                if "metadata" in transcript:
                    transcript["metadata"]["target_system_prompt"] = clone.target_system_prompt
                    transcript["metadata"]["target_tools"] = clone.get_serializable_functions()
                return {"condition": condition, "richness": richness, "transcript": transcript}
                
            # CHANGE 14: asyncio.gather(...) over all conditions.
            # Runs all the conditions, richness at the same tine using run_branch from Change 12
            branch_results = await asyncio.gather(*[run_branch(c, r) for c, r in conditions])
            return {
                "variation_number": variation_number,
                "variation_description": variation_description,
                "repetition_number": repetition_number,
                "modality": rollout_config["modality"],
                "branches": branch_results,
            }
         # END "simenv": block 

        else:
            # CHANGE 15. ConversationOrchestrator path (no feedback-injection branching). 
            # It does not use FeedbackInjectionOrchestrator
            # Only two adjustments apply here:
            # (1) max_turns now uses `effective_max_turns` in the agent=ConversationOrchestrator()
            # (2) return structure : output is wrapped in a "branches" 
            agent = ConversationOrchestrator.setup(
                client=utils.litellm_chat,
                evaluator_model_id=evaluator_model_id,
                target_model_id=target_model_id,
                evaluator_system_prompt=evaluator_system_prompt,
                conversation_rollout_prompt=conversation_rollout_prompt,
                target_sysprompt_prefix=target_sysprompt_prefix,
                max_turns=effective_max_turns,
                evaluator_reasoning_effort=rollout_config.get(
                    "evaluator_reasoning_effort", config.get("evaluator_reasoning_effort", "high")
                ),
                target_reasoning_effort=rollout_config.get(
                    "target_reasoning_effort", config.get("target_reasoning_effort", "medium")
                ),
                evaluator_max_tokens=rollout_config.get("max_tokens", 4000),
                target_max_tokens=rollout_config.get("max_tokens", 4000),
                evaluator_temperature=rollout_config.get(
                    "temperature", config.get("temperature", utils.DEFAULT_TEMPERATURE_DETERMINISTIC)
                ),
                target_temperature=rollout_config.get(
                    "temperature", config.get("temperature", utils.DEFAULT_TEMPERATURE_DETERMINISTIC)
                ),
                no_user_mode=rollout_config.get("no_user_mode", False),
                target_kickoff_prefix=prompts.get("target_kickoff_prefix", ""),
                generate_kickoff_additional=prompts.get("generate_kickoff_additional", ""),
                rollout_label=rollout_label,
            )

            loop = asyncio.get_event_loop()
            transcript_data = await loop.run_in_executor(None, agent.run)

            results_dir = Path(f"bloom-results/{behavior_name}")
            results_dir.mkdir(parents=True, exist_ok=True)

            if "metadata" in transcript_data:
                transcript_data["metadata"]["target_system_prompt"] = getattr(agent, "target_system_prompt", "")
                transcript_data["metadata"]["target_tools"] = []

            # CHANGED 16: Set "branches" condition/richness = None, so run_and_handle can treat both(with and withouth)
            # modalities at the same time.
            return {
                "variation_number": variation_number,
                "variation_description": variation_description,
                "repetition_number": repetition_number,
                "modality": rollout_config["modality"],
                "branches": [{"condition": None, "richness": None, "transcript": transcript_data}],
            }


async def run_rollout_async(
    config: Optional[Dict[str, Any]] = None, config_dir: str | os.PathLike[str] | None = None
) -> Dict[str, Any]:
    """Run the rollout step of the rollout pipeline for all variations concurrently."""
    print("\n" + "=" * 60, flush=True)
    print("ROLLOUT STAGE - STARTED", flush=True)
    print("=" * 60, flush=True)
    debug_print("Starting rollout process...")

    if config is None:
        config_path = Path(config_dir) / "seed.yaml" if config_dir else Path("seed.yaml")
        config = utils.load_config(config_path, config_dir=config_dir)
    elif config_dir and "_config_dir" not in config:
        config["_config_dir"] = Path(config_dir)
        
    # Load configurable prompts once
    prompts = utils.load_configurable_prompts(config)
    target_sysprompt_prefix = prompts.get("target_sysprompt_prefix", "")

    rollout_config = config["rollout"]
    behavior_name = config["behavior"]["name"]

    # Load models with config_dir
    models = utils.load_models(config.get("_config_dir"))

    # Get target model name (only if not anonymous)
    anonymous_target = config.get("anonymous_target", False)
    target_model_name = None
    if not anonymous_target:
        target_model_key = rollout_config.get("target", "unknown")
        target_model_name = models.get(target_model_key, {}).get("name", target_model_key)
    else:
        debug_print("Target model identity will remain anonymous")

    # Handle example as list (though we'll use behavior_name for directory)
    example_list = config["behavior"]["examples"]
    if not isinstance(example_list, list):
      # Backward compatibility: convert single string to list
        example_list = [example_list] if example_list else []

  # Load results using behavior name
    ideation_results = utils.load_ideation_results(behavior_name)
    understanding_results = utils.load_understanding_results(behavior_name)

    variations = ideation_results["variations"]
    behavior_understanding = understanding_results["understanding"]
    scientific_motivation = understanding_results["scientific_motivation"]
    # Get transcript analyses from understanding results
    transcript_analyses = understanding_results.get("transcript_analyses", [])

    # Get reasoning effort configuration
    evaluator_reasoning_effort = config.get("evaluator_reasoning_effort", "high")

    # Configure system prompt based on reasoning effort
    evaluator_system_prompt = make_system_prompt(
        behavior_name=behavior_name,
        modality=rollout_config["modality"],
        no_user_mode=rollout_config.get("no_user_mode", False),
        prompts=prompts,
    )

    selected_variations = rollout_config.get("selected_variations", None)
    num_reps = rollout_config.get("num_reps", 1)

    if selected_variations:
        variations_to_run = [(i + 1, v) for i, v in enumerate(variations) if (i + 1) in selected_variations]
    else:
        variations_to_run = [(i + 1, v) for i, v in enumerate(variations)]

    max_concurrent = config.get("max_concurrent", 5)
    semaphore = asyncio.Semaphore(max_concurrent)

    #CHANGE 18. Add excluded_variations = []
    rollouts = []
    failed_variations = []
    excluded_variations = []  # NEW
    tasks = []

    results_dir = utils.ensure_results_dir(behavior_name)

    # CHANGED 19: run_and_handle now processes multiple "branches" instead of a single
    # "transcript", and sends excluded cases to a separate list instead of mixing
    # them with failed variations.
    async def run_and_handle(variation_number, variation_data, repetition_number):
        print(f"Rolling out variation {variation_number}, repetition {repetition_number}...", flush=True)
        try:
            result = await run_single_variation_rollout(
                variation_number,
                variation_data,
                repetition_number,
                config,
                rollout_config,
                behavior_name,
                behavior_understanding,
                scientific_motivation,
                transcript_analyses,
                evaluator_system_prompt,
                behavior_name,
                semaphore,
                target_sysprompt_prefix,
                prompts,
                target_model_name,
            )
            
        # CHANGE 20. Check if this scenario has anything at all. If branches is None, put them in "excluded" pile.
            if result.get("branches") is None:
                print(
                    f"Variation {variation_number}, rep {repetition_number} - "
                    f"excluded ({result.get('excluded_reason')})",
                    flush=True,
                )
                excluded_variations.append(result)
                return
        # CHANGE 21: If they produce results iterate over branches.
            for branch in result["branches"]:
                transcript_data = branch["transcript"]
        #Check if something went wrong in the API call
                has_api_failure = any(
                    e.get("event") == "api_failure" for e in transcript_data.get("events", [])
                )
        # Label each filed copy clearly before filing it. which condition, it belongs to.
                entry = {
                    "variation_number": result["variation_number"],
                    "repetition_number": result["repetition_number"],
                    "condition": branch["condition"],
                    "richness": branch["richness"],
                    "modality": result["modality"],
                    "transcript": transcript_data,
                }
        # File it, same piles as before.
                if has_api_failure:
                    print(f"Variation {variation_number}, rep {repetition_number} - API failure", flush=True)
                    failed_variations.append(entry)
                rollouts.append(entry)

            print(
                f"Variation {variation_number}, rep {repetition_number} - "
                f"{len(result['branches'])} branch(es) completed",
                flush=True,
            )

        except Exception as exc:
            import traceback
     #End change 21
            print(
                f"Variation {variation_number}, rep {repetition_number} - failed: {type(exc).__name__}", flush=True
            )
            failed_variations.append(
                {
                    "variation_number": variation_number,
                    "repetition_number": repetition_number,
                    "error": str(exc),
                    "error_type": type(exc).__name__,
                    "traceback": traceback.format_exc(),
                }
            )

# Create tasks for all variations and repetitions. no change
    for variation_number, variation_data in variations_to_run:
        for repetition_number in range(1, num_reps + 1):
            tasks.append(run_and_handle(variation_number, variation_data, repetition_number))

    await asyncio.gather(*tasks)

    rollouts.sort(key=lambda x: (x["variation_number"], x["repetition_number"]))

    # CHANGED 23: filename now includes condition/richness suffix 
    # Branches from the same scenario don't overwrite each other.
    saved_transcripts = 0
    for rollout in rollouts:
        if "transcript" in rollout:
    #change here in sufix
            suffix = ""
            if rollout.get("condition") is not None:
                suffix = f"_{rollout['condition']}-{rollout.get('richness', 'na')}"
    #up to here inclusuon of conditon adn richness
            filename = f"transcript_v{rollout['variation_number']}r{rollout['repetition_number']}{suffix}.json"
            transcript_path = results_dir / filename
            transcript_data = rollout["transcript"]
            with open(transcript_path, "w", encoding="utf-8") as f:
                json.dump(transcript_data, f, indent=2, ensure_ascii=False)
            saved_transcripts += 1

    # CHANGED 24 : Add scenarios_attempted / scenarios_excluded computation
    scenarios_attempted = len(variations_to_run) * num_reps
    scenarios_excluded = len(excluded_variations)

    # CHANGED 25 : 25 Change the final rollout_results dict. Add "excluded_variations" 
    #"scenarios_attempted"/"scenarios_excluded"/"scenarios_completed"
    rollout_results = {
        "metadata": {
            "modality": rollout_config.get("modality", "conversation"),
            "no_user_mode": str(rollout_config.get("no_user_mode", False)),
            "max_turns": rollout_config.get("max_turns", 5),
            "evaluator": rollout_config.get("model", "unknown"),
            "target": rollout_config.get("target", "unknown"),
        },
        "rollouts": rollouts,
        "excluded_variations": excluded_variations,#New
        "successful_count": len(rollouts),  # branch-level count now, not scenario-level
        "failed_count": len(failed_variations),
        "total_count": len(rollouts) + len(failed_variations),
        "scenarios_attempted": scenarios_attempted, #New
        "scenarios_excluded": scenarios_excluded, #New
        "scenarios_completed": scenarios_attempted - scenarios_excluded,#New
        "variations_count": len(variations_to_run),
        "repetitions_per_variation": num_reps,
    }

    rollout_file = results_dir / "rollout.json"
    utils.save_results_locally(rollout_results, rollout_file, behavior_name)

    print("\n" + "=" * 60, flush=True)
    print("ROLLOUT STAGE - COMPLETED", flush=True)
    print("=" * 60, flush=True)

    return rollout_results


# UNCHANGED
async def run_rollout(
    config: Optional[Dict[str, Any]] = None, config_dir: str | os.PathLike[str] | None = None
) -> Dict[str, Any]:
    result = await run_rollout_async(config, config_dir=config_dir)
    return result


if __name__ == "__main__":
    try:
        debug_mode = False
        if "--debug" in sys.argv:
            debug_mode = True
            sys.argv.remove("--debug")
            from bloom.core import set_debug_mode

            set_debug_mode(True)

        if len(sys.argv) > 1:
            config_path = sys.argv[1]
        else:
            config_path = "seed.yaml"
        config = utils.load_config(config_path)

        result = asyncio.run(run_rollout(config))
    except Exception:
        import traceback

        traceback.print_exc()
        sys.exit(1)
