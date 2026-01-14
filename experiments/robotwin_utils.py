import os
import yaml
import importlib
from typing import Any, Dict, List, Optional, Tuple, Union
import json

import numpy as np
import torch

from PIL import Image, ImageDraw, ImageFont
from torchvision import transforms
from transformers import GenerationConfig
from sft.modeling_openpi_fast_oft import OpenpiFastOft
from data.normalize import Unnormalize_Action
from sft.constants import ACTION_PROPRIO_NORMALIZATION_TYPE, ACTION_MASK, NUM_ACTIONS_CHUNK, ACTION_DIM

current_file_path = os.path.abspath(__file__)
parent_directory = os.path.dirname(current_file_path)

resize_trans = transforms.Resize(size=(480, 640))
THINK_PREFIX = "First output the thinking process in <think></think> tags and then output the final action in <action></action>."
DEVICE = torch.device("cuda") if torch.cuda.is_available() else torch.device("cpu")

def get_embodiment_config(robot_file):
    robot_config_file = os.path.join(robot_file, "config.yml")
    with open(robot_config_file, "r", encoding="utf-8") as f:
        embodiment_args = yaml.load(f.read(), Loader=yaml.FullLoader)
    return embodiment_args

def get_camera_config(camera_type):
    camera_config_path = os.path.join(parent_directory, "../RoboTwin/task_config/_camera_config.yml")

    assert os.path.isfile(camera_config_path), "task config file is missing"

    with open(camera_config_path, "r", encoding="utf-8") as f:
        args = yaml.load(f.read(), Loader=yaml.FullLoader)

    assert camera_type in args, f"camera {camera_type} is not defined"
    return args[camera_type]


def class_decorator(task_name):
    envs_module = importlib.import_module(f"RoboTwin.envs.{task_name}")
    try:
        env_class = getattr(envs_module, task_name)
        env_instance = env_class()
    except:
        raise SystemExit("No Task")
    return env_instance

def binarize_gripper_action(action: np.ndarray) -> np.ndarray:
    # Create a copy to avoid modifying the original
    normalized_action = action.copy()

    # Binarize to -1 or +1
    normalized_action[..., 6] = np.sign(normalized_action[..., 6])
    normalized_action[..., -1] = np.sign(normalized_action[..., -1])

    return normalized_action

def get_vla(cfg) -> torch.nn.Module:
    """
    Load and initialize the VLA model from checkpoint.

    Args:
        cfg: Configuration object

    Returns:
        torch.nn.Module: The initialized VLA model
    """

    # Load the model
    vla = OpenpiFastOft.from_pretrained(
        cfg.pretrained_checkpoint,
        torch_dtype=getattr(torch, cfg.compute_dtype),
        attn_implementation = 'sdpa',
    )

    vla.eval()

    vla = vla.to(DEVICE)

    unomrmalize_action = _get_unomrmalize_action(cfg.pretrained_checkpoint)

    return vla, unomrmalize_action


def _get_unomrmalize_action(checkpoint_path: str) -> None:

    dataset_statistics_path = os.path.join(checkpoint_path, "norm_stats.json")
    if os.path.isfile(dataset_statistics_path):
        with open(dataset_statistics_path, "r") as f:
            norm_stats = json.load(f)
        for key in norm_stats["action"].keys():
            norm_stats["action"][key] = np.array(norm_stats["action"][key], dtype=np.float64)
        unomrmalize_action = Unnormalize_Action(
                normalization_type=ACTION_PROPRIO_NORMALIZATION_TYPE,
                stats=norm_stats["action"],
                action_mask=ACTION_MASK,
            )
        return unomrmalize_action
    else:
        print(
            "WARNING: No local dataset_statistics.json file found for current checkpoint.\n"
            "You can ignore this if you are loading the base VLA (i.e. not fine-tuned) checkpoint."
            "Otherwise, you may run into errors when trying to call `predict_action()` due to an absent `unnorm_key`."
        )
        raise NotImplementedError("No norm stats found!")


def encode_obs(obs: dict) -> dict:
    return {
        "full_image": obs["observation"]["head_camera"]["rgb"],
        "left_wrist_image": obs["observation"]["left_camera"]["rgb"],
        "right_wrist_image": obs["observation"]["right_camera"]["rgb"],
        "state": obs["joint_action"]["vector"],
        "instruction": obs["language"],
    }


def get_vla_action(
    cfg: Any,
    vla: torch.nn.Module,
    unomrmalize_action,
    processor: Any,
    obs: Dict[str, Any],
) -> List[np.ndarray]:
    with torch.inference_mode():
        # Process images
        obs = encode_obs(obs)
        image = (
            [
                resize_trans(Image.fromarray(obs["full_image"]).convert("RGB")),
                resize_trans(Image.fromarray(obs["left_wrist_image"]).convert("RGB")),
                resize_trans(Image.fromarray(obs["right_wrist_image"]).convert("RGB")),
            ]
        )

        # Build VLA prompt
        if "cot" in cfg.pretrained_checkpoint:
            prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + THINK_PREFIX + f"Task: {obs['instruction'].lower().replace('.', '')};"
        else:
            prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + f"Task: {obs['instruction'].lower().replace('.', '')};"

        # Process primary image
        inputs = processor(text = [prompt], images = image, return_tensors="pt").to(DEVICE, dtype=torch.bfloat16)

        # Generate action
        # Standard VLA output (single-image inputs, discrete actions)
        if 'cot' in cfg.pretrained_checkpoint:
            kwargs = {
                "max_new_tokens": cfg.max_new_tokens,
                "do_sample": False,
                "pad_token_id": processor.tokenizer.pad_token_id,
                "bos_token_id" : processor.tokenizer.bos_token_id,
                "eos_token_id" : None,
                "use_cache" : True,
                "num_beams": 1,
                "temperature" : None,
                "top_p" : None,
                "top_k" : None,
            }
            generation_config = GenerationConfig(**kwargs)
            normalized_actions, input_cot_ids = vla.predict_cot_action(
                input_ids = inputs["input_ids"],
                pixel_values = inputs["pixel_values"],
                attention_mask = inputs["attention_mask"],
                generation_config = generation_config,
            )
            actions = unomrmalize_action(torch.from_numpy(normalized_actions)).numpy()
            cot_text = processor.tokenizer.decode(input_cot_ids[0, inputs["input_ids"].shape[-1]:-1])
        else:
            actions, _ = vla.predict_action(**inputs, unnorm_key=cfg.unnorm_key, do_sample=False)

    # Return action chunk as list of actions
    return [actions[i] for i in range(len(actions))], cot_text

def get_vla_action_mask_cot(
    cfg: Any,
    vla: torch.nn.Module,
    unomrmalize_action,
    processor: Any,
    obs: Dict[str, Any],
) -> List[np.ndarray]:
    with torch.inference_mode():
        # Process images
        obs = encode_obs(obs)
        image = (
            [
                resize_trans(Image.fromarray(obs["full_image"]).convert("RGB")),
                resize_trans(Image.fromarray(obs["left_wrist_image"]).convert("RGB")),
                resize_trans(Image.fromarray(obs["right_wrist_image"]).convert("RGB")),
            ]
        )

        # Build VLA prompt
        if "cot" in cfg.pretrained_checkpoint:
            prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + THINK_PREFIX + f"Task: {obs['instruction'].lower().replace('.', '')};"
        else:
            prompt = processor.tokenizer.additional_special_tokens[0] * len(image) + f"Task: {obs['instruction'].lower().replace('.', '')};"

        # Process primary image
        inputs = processor(text = [prompt], images = image, return_tensors="pt").to(DEVICE, dtype=torch.bfloat16)

        # Generate action
        # Standard VLA output (single-image inputs, discrete actions)
        input_ids = torch.cat([inputs["input_ids"], torch.tensor([[257153, 257154, 257155]], device = inputs["input_ids"].device)], dim=-1)
        attention_mask = torch.cat([inputs["attention_mask"], torch.tensor([[1, 1, 1]], device = inputs["attention_mask"].device)], dim=-1)
        logits, action_start_idx = vla.prompt_cot_predict_action(
            input_cot_ids = input_ids,
            pixel_values = inputs["pixel_values"],
            attention_mask = attention_mask,
        )
        start_indices = action_start_idx.unsqueeze(1)  # [batch_size, 1]
        position_offsets = torch.arange(ACTION_DIM * NUM_ACTIONS_CHUNK, device=logits.device).unsqueeze(0)  # [1, seq_length]
        seq_indices = start_indices + position_offsets  # [batch_size, ACTION_DIM*NUM_ACTIONS_CHUNK]

        # Discrete token-based prediction
        predicted_action_token_ids = (vla.config.action_token_end_idx - vla.config.action_token_begin_idx) - (
            logits[
                torch.arange(logits.shape[0], device=logits.device).unsqueeze(-1),
                seq_indices,
                vla.config.action_token_begin_idx:vla.config.action_token_end_idx + 1
            ]
            .argmax(dim=-1)
            .cpu()
            .numpy()
        )
        discretized_actions = discretized_actions = np.clip(predicted_action_token_ids, a_min=0, a_max=vla.bin_centers.shape[0] - 1)
        normalized_actions = vla.bin_centers[discretized_actions]
        normalized_actions = normalized_actions.reshape(NUM_ACTIONS_CHUNK, ACTION_DIM)

        actions = unomrmalize_action(torch.from_numpy(normalized_actions)).numpy()
        cot_text = '<think></think>'


    # Return action chunk as list of actions
    return [actions[i] for i in range(len(actions))], cot_text