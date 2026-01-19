#!/usr/bin/env python
"""
Pi0.5 + SO-100/SO-101 ロボット制御スクリプト

Usage:
    python examples/tutorial/pi0/run_pi05_robot.py \
        --robot.type=so101_follower \
        --robot.port=/dev/ttyACM1 \
        --robot.id=follower1 \
        --dataset.repo_id=daikw/thumbturn \
        --task="pick the object in front of you" \
        --steps=100 \
        --robot.cameras='{
            base_0_rgb: {type: opencv, index_or_path: /dev/video_top, width: 640, height: 480, fps: 30},
            left_wrist_0_rgb: {type: opencv, index_or_path: /dev/video_wrist, width: 640, height: 480, fps: 30}
        }'

Options:
    --dataset.repo_id        Required. Dataset for action denormalization stats.
    --use_action_chunking    Use action chunking for faster control (default: true)
    --no-use_action_chunking Disable action chunking (inference every step)
    --verbose_timing         Show detailed timing breakdown for each inference
    --freq                   Control frequency in Hz (default: 5.0)

Note:
    --dataset.repo_id is required for proper action denormalization.
    The Pi0.5 base model outputs normalized actions in [-1, 1] range,
    which must be mapped to your robot's action range using dataset stats.

    Action chunking reuses actions from a single inference across multiple steps,
    reducing the number of expensive model inferences needed.

"""

import logging
import time
from dataclasses import asdict, dataclass, field
from pathlib import Path
from pprint import pformat

import torch

from lerobot.cameras import CameraConfig  # noqa: F401
from lerobot.cameras.opencv.configuration_opencv import OpenCVCameraConfig  # noqa: F401
from lerobot.configs import parser
from lerobot.datasets.lerobot_dataset import LeRobotDataset
from lerobot.datasets.utils import hw_to_dataset_features
from lerobot.policies.factory import make_pre_post_processors
from lerobot.policies.pi05.modeling_pi05 import PI05Policy
from lerobot.policies.utils import build_inference_frame, make_robot_action
from lerobot.robots import (  # noqa: F401
    RobotConfig,
    make_robot_from_config,
    so100_follower,
    so101_follower,
)
from lerobot.utils.import_utils import register_third_party_plugins
from lerobot.utils.utils import init_logging


@dataclass
class DatasetConfig:
    # Dataset repo_id for normalization stats (e.g. "daikw/thumbturn")
    repo_id: str | None = None
    # Local root path for dataset (optional)
    root: str | Path | None = None


@dataclass
class Pi05ControlConfig:
    robot: RobotConfig
    # Dataset config for normalization stats
    dataset: DatasetConfig = field(default_factory=DatasetConfig)
    # Language instruction for the robot
    task: str = "pick up the object"
    # Number of control steps
    steps: int = 100
    # Control frequency in Hz
    freq: float = 5.0
    # Model to use
    model_id: str = "lerobot/pi05_base"
    # Device for inference
    device: str = "cuda"
    # Use action chunking for faster inference (reuse actions from single inference)
    use_action_chunking: bool = True
    # Disable gradient checkpointing for faster inference (not needed at inference time)
    disable_gradient_checkpointing: bool = True
    # Show detailed timing info
    verbose_timing: bool = False


@parser.wrap()
def run(cfg: Pi05ControlConfig):
    init_logging()
    logging.info(pformat(asdict(cfg)))

    device = torch.device(cfg.device)

    print(f"\nTask: {cfg.task}")
    print(f"Steps: {cfg.steps}")
    print(f"Freq: {cfg.freq} Hz")
    print(f"Action chunking: {'enabled' if cfg.use_action_chunking else 'disabled'}")
    print()

    # 1. Load dataset stats (if provided)
    dataset_stats = None
    action_dim = None
    if cfg.dataset.repo_id:
        print(f"Loading dataset stats from: {cfg.dataset.repo_id}")
        dataset = LeRobotDataset(cfg.dataset.repo_id, root=cfg.dataset.root)
        dataset_stats = dataset.meta.stats
        # Get actual action dimension from dataset
        action_dim = dataset_stats["action"]["mean"].shape[0]
        print(f"✓ Dataset stats loaded (action_dim={action_dim})")
    else:
        print("⚠ No dataset specified - actions will NOT be denormalized!")
        print("  Use --dataset.repo_id=<your_dataset> for proper action scaling")

    # 2. Load model
    print("Loading Pi0.5 model...")
    model = PI05Policy.from_pretrained(cfg.model_id)
    model.eval()

    # Disable gradient checkpointing for faster inference
    if cfg.disable_gradient_checkpointing and model.config.gradient_checkpointing:
        model.config.gradient_checkpointing = False
        print("  → Disabled gradient checkpointing for inference")

    # Override model config with actual action dimension from dataset
    # This is necessary because Pi0.5 base model uses max_action_dim=32 by default,
    # but we need to match the actual robot's action dimension for proper denormalization
    if action_dim is not None:
        from lerobot.configs.types import FeatureType, PolicyFeature

        model.config.output_features["action"] = PolicyFeature(
            type=FeatureType.ACTION,
            shape=(action_dim,),
        )
        print(f"  → Updated model action dim: 32 → {action_dim}")

    # Build processor overrides
    preprocessor_overrides = {"device_processor": {"device": str(device)}}
    postprocessor_overrides = {}

    if dataset_stats:
        preprocessor_overrides["normalizer_processor"] = {"stats": dataset_stats}
        postprocessor_overrides["unnormalizer_processor"] = {"stats": dataset_stats}

    preprocess, postprocess = make_pre_post_processors(
        model.config,
        cfg.model_id,
        preprocessor_overrides=preprocessor_overrides,
        postprocessor_overrides=postprocessor_overrides,
    )
    print("✓ Model loaded")

    # 3. Connect robot
    print("Connecting to robot...")
    robot = make_robot_from_config(cfg.robot)
    robot.connect()
    print(f"✓ Robot connected: {robot.name} (id={cfg.robot.id})")

    # 4. Build feature mapping
    action_features = hw_to_dataset_features(robot.action_features, "action")
    obs_features = hw_to_dataset_features(robot.observation_features, "observation")
    dataset_features = {**action_features, **obs_features}

    # Warmup: run one inference to trigger JIT compilation
    print("Warming up model (JIT compile)...")
    warmup_obs = robot.get_observation()
    warmup_frame = build_inference_frame(
        observation=warmup_obs,
        ds_features=dataset_features,
        device=device,
        task=cfg.task,
        robot_type=robot.name,
    )
    warmup_processed = preprocess(warmup_frame)
    with torch.no_grad():
        _ = model.select_action(warmup_processed)
    print("✓ Warmup done")

    # Get n_action_steps from model config for action chunking
    n_action_steps = model.config.n_action_steps
    print(f"\nStarting control loop ({cfg.steps} steps at {cfg.freq} Hz)")
    if cfg.use_action_chunking:
        print(f"  Action chunk size: {n_action_steps} (model internally caches actions)")
    else:
        print("  Action chunking disabled (clearing model's internal queue each step)")
    print("Press Ctrl+C to stop\n")

    input("Press Enter to start robot control...")
    print("Starting!\n")

    dt = 1.0 / cfg.freq
    inference_count = 0
    total_inference_time = 0.0

    try:
        for step in range(cfg.steps):
            t0 = time.time()

            # Get observation
            obs = robot.get_observation()
            t_obs = time.time()

            # Build inference frame
            obs_frame = build_inference_frame(
                observation=obs,
                ds_features=dataset_features,
                device=device,
                task=cfg.task,
                robot_type=robot.name,
            )
            t_frame = time.time()

            # Preprocess
            obs_processed = preprocess(obs_frame)
            t_preprocess = time.time()

            # If action chunking is disabled, clear the model's internal action queue
            # This forces a new inference every step
            if not cfg.use_action_chunking:
                model._action_queue.clear()

            # Check if model will do inference (queue is empty)
            will_infer = len(model._action_queue) == 0

            # select_action internally manages action chunking:
            # - If queue is empty: runs inference, fills queue with n_action_steps actions
            # - Returns one action from queue
            with torch.no_grad():
                action = model.select_action(obs_processed)
            t_inference = time.time()

            # Track inference timing (only when actual inference happened)
            if will_infer:
                inference_time = t_inference - t_preprocess
                total_inference_time += inference_time
                inference_count += 1

            # Postprocess (action is torch.Tensor with shape: batch_size x action_dim)
            action = postprocess(action)
            t_postprocess = time.time()

            # Convert to robot action format (postprocess returns {"action": tensor})
            action = make_robot_action(action, dataset_features)

            if cfg.verbose_timing:
                infer_str = f"infer:{(t_inference-t_preprocess)*1000:.0f}ms" if will_infer else "cached"
                print(f"  [Timing] obs:{(t_obs-t0)*1000:.0f}ms frame:{(t_frame-t_obs)*1000:.0f}ms "
                      f"pre:{(t_preprocess-t_frame)*1000:.0f}ms {infer_str} "
                      f"post:{(t_postprocess-t_inference)*1000:.0f}ms")

            # Debug: show actions
            print(f"  action: {action}")

            # Send action to robot
            robot.send_action(action)

            # Timing
            elapsed = time.time() - t0
            if elapsed < dt:
                time.sleep(dt - elapsed)

            if step % 10 == 0:
                avg_inference = (total_inference_time / inference_count * 1000) if inference_count > 0 else 0
                queue_len = len(model._action_queue)
                chunk_info = f" queue={queue_len}" if cfg.use_action_chunking else ""
                print(f"Step {step}/{cfg.steps} ({elapsed*1000:.0f}ms, avg_infer={avg_inference:.0f}ms{chunk_info})")

    except KeyboardInterrupt:
        print("\nStopped by user")

    finally:
        print("Disconnecting robot...")
        robot.disconnect()
        if inference_count > 0:
            print(f"Stats: {inference_count} inferences, avg {total_inference_time/inference_count*1000:.0f}ms/inference")
        print("Done!")


def main():
    register_third_party_plugins()
    run()


if __name__ == "__main__":
    main()
