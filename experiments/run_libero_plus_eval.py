"""
run_libero_eval.py

Evaluates a trained policy in a LIBERO simulation benchmark task suite.
"""
import sys
sys.path.append("./")
import json
import logging
import os
import sys
from collections import deque
from dataclasses import dataclass
from enum import Enum
from pathlib import Path
from typing import Optional, Union

import draccus
import numpy as np
import tqdm
from libero.libero import benchmark

import swanlab
from transformers import AutoProcessor
import torch
import random
import time

from experiments.libero_utils import (
    get_libero_dummy_action,
    get_libero_env,
    get_libero_image,
    get_libero_wrist_image,
    quat2axisangle,
    save_rollout_video,
)
from experiments.deepthinkvla_utils import (
    resize_image_for_policy,
    get_vla,
    get_vla_action,
    get_vla_action_mask_cot,
    get_vla_action_mask_cot_random,
    compose_with_sidepanel,
    binarize_gripper_action
)
from sft.constants import NUM_ACTIONS_CHUNK


DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")

# Define task suite constants
class TaskSuite(str, Enum):
    LIBERO_SPATIAL = "libero_spatial"
    LIBERO_OBJECT = "libero_object"
    LIBERO_GOAL = "libero_goal"
    LIBERO_10 = "libero_10"
    LIBERO_90 = "libero_90"
    LIBERO_MIX = 'libero_mix'


# Define max steps for each task suite
TASK_MAX_STEPS = {
    TaskSuite.LIBERO_SPATIAL: 220,  # longest training demo has 193 steps
    TaskSuite.LIBERO_OBJECT: 280,  # longest training demo has 254 steps
    TaskSuite.LIBERO_GOAL: 300,  # longest training demo has 270 steps
    TaskSuite.LIBERO_10: 620,  # longest training demo has 620 steps
    TaskSuite.LIBERO_90: 400,  # longest training demo has 373 steps
}


# Set up logging
logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(message)s",
    handlers=[logging.StreamHandler()],
)
logger = logging.getLogger(__name__)


@dataclass
class GenerateConfig:
    # fmt: off

    #################################################################################################################
    # Model-specific parameters
    #################################################################################################################
    pretrained_checkpoint: Union[str, Path] = "yinchenghust/sft_cot"     # Pretrained checkpoint path (or HF repo id)

    num_images_in_input: int = 2                # Number of images in input context

    max_new_tokens: int = 2048                       # Maximum number of cot tokens to generate (COT only)

    compute_dtype: str = "bfloat16"                    # Model compute dtype (float32, float16, bfloat16)

    img_resize_size: int = 224                     # Input image resolution for the model

    #################################################################################################################
    # LIBERO environment-specific parameters
    #################################################################################################################
    task_suite_name: str = TaskSuite.LIBERO_OBJECT     # Task suite
    num_steps_wait: int = 10                         # Number of steps to wait for objects to stabilize in sim
    initial_states_path: str = "DEFAULT"             # "DEFAULT", or path to initial states JSON file
    env_img_res: int = 256                           # Resolution for environment images (not policy input resolution)

    #################################################################################################################
    # Utils
    #################################################################################################################
    run_id_note: Optional[str] = None                # Extra note to add to end of run ID for logging
    local_log_dir: str = "./experiments/logs"        # logs_mask_cot, logs_mask_cot_random, logs

    project_name: str = "deepthinkvla"                 # Name of project to log to
    swanlab_api_key: Optional[str] = None              # Prefer env var; keep None for open-source safety
    swanlab_mode: str = 'disabled'                      # cloud-only, local, disabled

    seed: int = 429                                    # Random Seed (for reproducibility)

    panel_width_px: int = 812                         # Width of side panel for displaying CoT text

    # fmt: on

def set_seed_everywhere(seed: int) -> None:
    """
    Set random seed for all random number generators for reproducibility.

    Args:
        seed: The random seed to use
    """
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    np.random.seed(seed)
    random.seed(seed)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
    os.environ["PYTHONHASHSEED"] = str(seed)

def validate_config(cfg: GenerateConfig) -> None:
    """Validate configuration parameters."""
    assert cfg.pretrained_checkpoint is not None, "pretrained_checkpoint must not be None!"

    # Validate task suite
    assert cfg.task_suite_name in [suite.value for suite in TaskSuite], f"Invalid task suite: {cfg.task_suite_name}"




def setup_logging(cfg: GenerateConfig):
    """Set up logging to file and optionally to wandb."""
    # Create run ID
    run_id = f"EVAL-{cfg.task_suite_name}-deepthinkvla-{DATE_TIME}"
    if cfg.run_id_note is not None:
        run_id += f"--{cfg.run_id_note}"

    # Set up local logging
    os.makedirs(cfg.local_log_dir, exist_ok=True)
    local_log_filepath = os.path.join(cfg.local_log_dir, run_id + ".txt")
    log_file = open(local_log_filepath, "w")
    logger.info(f"Logging to local log file: {local_log_filepath}")

    # Initialize SwanLab logging if enabled
    if cfg.swanlab_mode != "disabled":
        # Prefer environment variable; fall back to cfg
        api_key = os.environ.get("SWANLAB_API_KEY", None) or cfg.swanlab_api_key
        if api_key:
            swanlab.login(api_key)  # NOTE: previous login information will be overwritten
        swanlab.init(
            project=cfg.project_name,
            experiment_name=run_id,
            logdir=cfg.local_log_dir,
            mode=cfg.swanlab_mode,
        )

    return log_file, local_log_filepath, run_id


def log_message(message: str, log_file=None):
    """Log a message to console and optionally to a log file."""
    logger.info(message)
    if log_file:
        log_file.write(message + "\n")
        log_file.flush()


def load_initial_states(cfg: GenerateConfig, task_suite, task_id: int, log_file=None):
    """Load initial states for the given task."""
    # Get default initial states
    initial_states = task_suite.get_task_init_states(task_id)

    # If using custom initial states, load them from file
    if cfg.initial_states_path != "DEFAULT":
        with open(cfg.initial_states_path, "r") as f:
            all_initial_states = json.load(f)
        log_message(f"Using initial states from {cfg.initial_states_path}", log_file)
        return initial_states, all_initial_states
    else:
        log_message("Using default initial states", log_file)
        return initial_states, None


def prepare_observation(obs, resize_size):
    """Prepare observation for policy input."""
    # Get preprocessed images
    img = get_libero_image(obs)
    wrist_img = get_libero_wrist_image(obs)

    # Resize images to size expected by model
    img_resized = resize_image_for_policy(img, resize_size)
    wrist_img_resized = resize_image_for_policy(wrist_img, resize_size)

    # Prepare observations dict
    observation = {
        "full_image": img_resized,
        "wrist_image": wrist_img_resized,
        "state": np.concatenate(
            (obs["robot0_eef_pos"], quat2axisangle(obs["robot0_eef_quat"]), obs["robot0_gripper_qpos"])
        ),
    }

    return observation, img  # Return both processed observation and original image for replay


def run_episode(
    cfg: GenerateConfig,
    env,
    task_description: str,
    model,
    unomrmalize_action,
    resize_size,
    processor=None,
    initial_state=None,
    log_file=None,
):
    """Run a single episode in the environment."""
    # Reset environment
    env.reset()

    # Set initial state if provided
    if initial_state is not None:
        obs = env.set_init_state(initial_state)
    else:
        obs = env.get_observation()

    action_queue = deque(maxlen=NUM_ACTIONS_CHUNK)

    # Setup
    t = 0
    replay_images = []
    cot_replay = []
    max_steps = TASK_MAX_STEPS[cfg.task_suite_name]

    # Run episode
    success = False
    try:
        while t < max_steps + cfg.num_steps_wait:
            # Do nothing for the first few timesteps to let objects stabilize
            if t < cfg.num_steps_wait:
                obs, reward, done, info = env.step(get_libero_dummy_action())
                t += 1
                continue

            # Prepare observation
            observation, img = prepare_observation(obs, resize_size)
            replay_images.append(img)

            # If action queue is empty, requery model
            if len(action_queue) == 0:
                # Query model to get action
                with torch.no_grad():
                    # mask_cot: get_vla_action_mask_cot
                    # mask_cot_random: get_vla_action_mask_cot_random
                    actions, cot_text = get_vla_action(
                        cfg=cfg,
                        vla=model,
                        unomrmalize_action = unomrmalize_action,
                        processor=processor,
                        obs=observation,
                        task_label=task_description,
                    )
                action_queue.extend(actions)
                cot_replay.append(cot_text)

            # Get action from queue
            action = action_queue.popleft()

            # Process action
            action = binarize_gripper_action(action)
            replay_images[-1] = compose_with_sidepanel(replay_images[-1], cot_replay[-1] if len(cot_replay) > 0 else None, panel_width_px=cfg.panel_width_px)

            # Execute action in environment
            obs, reward, done, info = env.step(action.tolist())
            if done:
                success = True
                break
            t += 1

    except Exception as e:
        log_message(f"Episode error: {e}", log_file)

    return success, replay_images


def run_task(
    cfg: GenerateConfig,
    task_suite,
    task_id: int,
    model,
    unomrmalize_action,
    resize_size,
    processor=None,
    log_file=None,
):
    """Run evaluation for a single task."""
    # Get task
    task = task_suite.get_task(task_id)

    # Get initial states
    initial_states, all_initial_states = load_initial_states(cfg, task_suite, task_id, log_file)

    # Initialize environment and get task description
    env, task_description = get_libero_env(task, resolution=cfg.env_img_res)

    log_message(f"\nTask: {task_description}", log_file)

    # Handle initial state
    if cfg.initial_states_path == "DEFAULT":
        # Use default initial state
        initial_state = initial_states[0]
    else:
        raise('now is not supported')

    # Run episode
    success, replay_images = run_episode(
        cfg,
        env,
        task_description,
        model,
        unomrmalize_action,
        resize_size,
        processor,
        initial_state,
        log_file,
    )

    # Save replay video
    save_rollout_video(
        replay_images, success=success, task_description=task_description, log_file=log_file
    )

    # Log results
    log_message(f"Success: {success}", log_file)

    return success


@draccus.wrap()
def eval_libero(cfg: GenerateConfig) -> float:
    """Main function to evaluate a trained policy on LIBERO benchmark tasks."""
    # Validate configuration
    validate_config(cfg)

    # Set random seed
    set_seed_everywhere(cfg.seed)

    with open("name_to_category.json", "r") as f:
        name_to_category =  json.load(f)

    ##########################################################################################
    # Initialize model and components
    model, unomrmalize_action = get_vla(cfg)

    processor = AutoProcessor.from_pretrained(cfg.pretrained_checkpoint)
    ##########################################################################################

    ##########################################################################################
    # Setup logging
    log_file, local_log_filepath, run_id = setup_logging(cfg)

    # Initialize LIBERO task suite
    benchmark_dict = benchmark.get_benchmark_dict()
    task_suite = benchmark_dict[cfg.task_suite_name]()
    num_tasks = task_suite.n_tasks

    log_message(f"Task suite: {cfg.task_suite_name}", log_file)

    # Start evaluation
    result_success_dict = {
        'Objects Layout': 0,
        'Language Instructions': 0,
        'Light Conditions': 0,
        'Camera Viewpoints': 0,
        'Robot Initial States' : 0,
        'Background Textures': 0,
        'Sensor Noise': 0,
    }
    result_fail_dict = {
        'Objects Layout': 0,
        'Language Instructions': 0,
        'Light Conditions': 0,
        'Camera Viewpoints': 0,
        'Robot Initial States' : 0,
        'Background Textures': 0,
        'Sensor Noise': 0,
    }
    total_successes = 0
    for task_id in tqdm.tqdm(range(num_tasks)):
        # task_id = 0
        success = run_task(
            cfg,
            task_suite,
            task_id,
            model,
            unomrmalize_action,
            cfg.img_resize_size,
            processor,
            log_file,
        )
        result_success_dict[name_to_category[task_suite.get_task_names()[task_id]]] += 1 if success else 0
        result_fail_dict[name_to_category[task_suite.get_task_names()[task_id]]] += 1 if not success else 0
        if success:
            total_successes += 1

    # Calculate final success rate
    final_success_rate = float(total_successes) / float(num_tasks)

    # Log final results
    log_message("Final results:", log_file)
    log_message(f"Total episodes: {num_tasks}", log_file)
    log_message(f"Total successes: {total_successes}", log_file)
    log_message(f"Overall success rate: {final_success_rate:.4f} ({final_success_rate * 100:.1f}%)", log_file)

    # Close log file
    with open(cfg.task_suite_name.lower()+'_success_outcome.json', "w") as f:
        json.dump(result_success_dict, f, indent=4)
    with open(cfg.task_suite_name.lower()+'_fail_outcome.json', "w") as f:
        json.dump(result_fail_dict, f, indent=4)
    if log_file:
        log_file.close()

    return final_success_rate


if __name__ == "__main__":
    eval_libero()
