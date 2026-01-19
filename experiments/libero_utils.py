"""Utils for evaluating policies in LIBERO simulation environments."""

import math
import os
import time

import imageio
import numpy as np
from libero.libero import get_libero_path
from libero.libero.envs import OffScreenRenderEnv

DATE = time.strftime("%Y_%m_%d")
DATE_TIME = time.strftime("%Y_%m_%d-%H_%M_%S")


def _ceil_to_multiple(x: int, m: int) -> int:
    if m <= 0:
        return x
    return ((x + m - 1) // m) * m


def _pad_frame_to_shape(img: np.ndarray, target_h: int, target_w: int, pad_value: int = 255) -> np.ndarray:
    """
    Pad an HxWxC uint8 image to (target_h, target_w, C) using a constant background.
    Assumes img is HxWx3 or HxWxC.
    """
    if not isinstance(img, np.ndarray):
        img = np.asarray(img)
    if img.ndim != 3:
        raise ValueError(f"Expected image with 3 dims (H,W,C), got shape={getattr(img, 'shape', None)}")
    h, w, c = img.shape
    if img.dtype != np.uint8:
        img = img.astype(np.uint8)

    if h == target_h and w == target_w:
        return img

    out = np.full((target_h, target_w, c), pad_value, dtype=np.uint8)
    out[: min(h, target_h), : min(w, target_w)] = img[: min(h, target_h), : min(w, target_w)]
    return out


def get_libero_env(task, resolution=256):
    """Initializes and returns the LIBERO environment, along with the task description."""
    task_description = task.language
    task_bddl_file = os.path.join(get_libero_path("bddl_files"), task.problem_folder, task.bddl_file)
    env_args = {
        "bddl_file_name": task_bddl_file,
        "camera_heights": resolution,
        "camera_widths": resolution,
    }
    env = OffScreenRenderEnv(**env_args)
    env.seed(0)  # IMPORTANT: seed seems to affect object positions even when using fixed initial state
    return env, task_description


def get_libero_dummy_action():
    """Get dummy/no-op action, used to roll out the simulation while the robot does nothing."""
    return [0, 0, 0, 0, 0, 0, -1]


def get_libero_image(obs):
    """Extracts third-person image from observations and preprocesses it."""
    img = obs["agentview_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def get_libero_wrist_image(obs):
    """Extracts wrist camera image from observations and preprocesses it."""
    img = obs["robot0_eye_in_hand_image"]
    img = img[::-1, ::-1]  # IMPORTANT: rotate 180 degrees to match train preprocessing
    return img


def save_rollout_video(rollout_images, success, task_description, log_file=None):
    """Saves an MP4 replay of an episode."""
    rollout_dir = f"./rollouts/{DATE}"
    os.makedirs(rollout_dir, exist_ok=True)
    processed_task_description = task_description.lower().replace(" ", "_").replace("\n", "_").replace(".", "_")[:50]
    mp4_path = f"{rollout_dir}/{DATE_TIME}--success={success}--task={processed_task_description}.mp4"

    # Ensure all frames have the same size.
    # This avoids: ValueError("All images in a movie should have same size")
    # Additionally, we pad to a multiple of 16 to avoid ffmpeg auto-resizing warnings.
    if rollout_images is None or len(rollout_images) == 0:
        print(f"Warning: empty rollout_images, skip saving video.")
        return None

    hs, ws = [], []
    for img in rollout_images:
        arr = np.asarray(img)
        if arr.ndim != 3:
            continue
        hs.append(arr.shape[0])
        ws.append(arr.shape[1])
    if len(hs) == 0 or len(ws) == 0:
        print(f"Warning: no valid frames, skip saving video.")
        return None

    target_h = max(hs)
    target_w = max(ws)
    target_h = _ceil_to_multiple(target_h, 16)
    target_w = _ceil_to_multiple(target_w, 16)

    video_writer = imageio.get_writer(mp4_path, fps=30)
    try:
        for img in rollout_images:
            frame = _pad_frame_to_shape(np.asarray(img), target_h=target_h, target_w=target_w, pad_value=255)
            video_writer.append_data(frame)
    finally:
        video_writer.close()

    print(f"Saved rollout MP4 at path {mp4_path}")
    if log_file is not None:
        log_file.write(f"Saved rollout MP4 at path {mp4_path}\n")
    return mp4_path


def quat2axisangle(quat):
    """
    Copied from robosuite: https://github.com/ARISE-Initiative/robosuite/blob/eafb81f54ffc104f905ee48a16bb15f059176ad3/robosuite/utils/transform_utils.py#L490C1-L512C55

    Converts quaternion to axis-angle format.
    Returns a unit vector direction scaled by its angle in radians.

    Args:
        quat (np.array): (x,y,z,w) vec4 float angles

    Returns:
        np.array: (ax,ay,az) axis-angle exponential coordinates
    """
    # clip quaternion
    if quat[3] > 1.0:
        quat[3] = 1.0
    elif quat[3] < -1.0:
        quat[3] = -1.0

    den = np.sqrt(1.0 - quat[3] * quat[3])
    if math.isclose(den, 0.0):
        # This is (close to) a zero degree rotation, immediately return
        return np.zeros(3)

    return (quat[:3] * 2.0 * math.acos(quat[3])) / den
