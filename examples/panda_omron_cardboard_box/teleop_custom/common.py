from __future__ import annotations

import argparse
import contextlib
from dataclasses import dataclass
import datetime
import io
import json
import os
from pathlib import Path
import re
import subprocess
import sys
import threading
import time
from typing import Any
import warnings

import mujoco
import mujoco.viewer
import numpy as np


MUJOCO_ROOT = Path(__file__).resolve().parents[1]
REPO_ROOT = MUJOCO_ROOT.parent
ENTRYPOINT_PATH = MUJOCO_ROOT / "teleop_custom_scene_t265.py"
MODEL_PATH = (
    MUJOCO_ROOT
    / "third_party"
    / "robosuite_assets"
    / "teleop_scenes"
    / "panda_omron_robocasa_native_fragile_wall_lift_scene.xml"
)
DEFAULT_ROBOCASA_ROOT = Path(os.environ.get("ROBOCASA_ROOT", REPO_ROOT / "robocasa"))
DEFAULT_ROBOSUITE_ROOT = Path(
    os.environ.get("ROBOSUITE_ROOT", REPO_ROOT / "robosuite")
)
DEFAULT_RANDOM_SCENE_OUTPUT_DIR = Path("/tmp/mujoco_robocasa_random_scenes")
DEFAULT_RECORDING_DIR = MUJOCO_ROOT / "data"
VIDEO_WIDTH = 640
VIDEO_HEIGHT = 480
VIDEO_FPS = 30
VIDEO_RENDER_EVERY = 16
WRIST_CAMERA_NAME = "ur10e_gripper_wrist_camera"
DEFAULT_VIDEO_VIEWS = (
    "agentview_right",
    "agentview_left",
    "robot_side_view",
    "robot0_eye_in_hand",
)
SIDE_VIEW_WIDTH = 480
SIDE_VIEW_HEIGHT = 360
SIDE_VIEW_FPS = 30.0
SIDE_VIEW_WINDOW = "robot_side_view"
SIDE_VIEW_LOOKAT = np.array([0.0, 0.25, 0.65], dtype=np.float64)
SIDE_VIEW_DISTANCE = 1.2155
SIDE_VIEW_AZIMUTH = 2.9512
LEFT_SIDE_VIEW_AZIMUTH = 182.9512
SIDE_VIEW_ELEVATION = 2.75
WORLD_VIEW_DISTANCE = 1.0478
WORLD_VIEW_AZIMUTH = -130.2488
WORLD_VIEW_ELEVATION = -27.1750
WORLD_VIEW_GLASS_WALL_Z_OFFSET = 0.16
WORKSPACE_VIEW_DISTANCE = 2.8
WORKSPACE_VIEW_AZIMUTH = -45.0
WORKSPACE_VIEW_REAR_SIDE_OFFSET = 35.0
WORKSPACE_VIEW_ELEVATION = -22.0
WORLD_VIEW_MIN_DISTANCE = 2.2
WORLD_VIEW_MAX_DISTANCE = 5.0
DEBUG_VIEW_INTERVAL = 1.0
DEBUG_CONTROL_INTERVAL = 0.25
VISUAL_GEOM_GROUP = 1
LOCAL_TELEOP_REALSENSE_SRC = (
    MUJOCO_ROOT / "third_party" / "teleop_realsense" / "source" / "TeleopRealSense"
)
LEGACY_TELEOP_REALSENSE_SRC = (
    REPO_ROOT
    / "extrinsic-dexterity"
    / "controller"
    / "teleop_isaac_realsense"
    / "source"
    / "TeleopRealSense"
)
TELEOP_REALSENSE_SRC = (
    LOCAL_TELEOP_REALSENSE_SRC
    if LOCAL_TELEOP_REALSENSE_SRC.exists()
    else LEGACY_TELEOP_REALSENSE_SRC
)
LOCAL_T265_SERVER_PY = (
    MUJOCO_ROOT / "third_party" / "teleop_realsense" / "realsense_client" / "server.py"
)

# Xbox-style pygame button indices. These match teleop_ur10e_mujoco.py.
DEFAULT_BTN_TOGGLE_CONTROL = 5  # RB
DEFAULT_BTN_RESET = 4  # LB
DEFAULT_BTN_GRIPPER = 1  # B
DEFAULT_BTN_RECORD = 2  # X
DEFAULT_BTN_ROTATION = 7  # RT/R2 on common pygame mappings
DEFAULT_AXIS_ROTATION = 5  # RT/R2 analog trigger on common pygame mappings
DS4_BTN_TOGGLE_CONTROL = 5  # R1
DS4_BTN_RESET = 4  # L1
DS4_BTN_GRIPPER = 1  # Circle
DS4_BTN_RECORD = 2  # Square
DS4_BTN_ROTATION = 7  # R2
ARM_ACTUATORS = (
    "ur10e_shoulder_pan",
    "ur10e_shoulder_lift",
    "ur10e_elbow",
    "ur10e_wrist_1",
    "ur10e_wrist_2",
    "ur10e_wrist_3",
)
ARM_JOINTS = (
    "ur10e_shoulder_pan_joint",
    "ur10e_shoulder_lift_joint",
    "ur10e_elbow_joint",
    "ur10e_wrist_1_joint",
    "ur10e_wrist_2_joint",
    "ur10e_wrist_3_joint",
)
GRIPPER_ACTUATOR = "ur10e_gripper_fingers_actuator"
GRIPPER_TOGGLE_THRESHOLD = 127.5
TRANSLATION_SPEED = 8.1
ROTATION_SPEED = 25.0
T265_TRANSLATION_DEADBAND = 5e-4
T265_ROTATION_DEADBAND = np.deg2rad(0.25)
MAX_EE_TARGET_POSITION_ERROR = 0.05
DEFAULT_EE_COMMAND_FRAME = "base"
VIEW_STEP = 0.05
ZOOM_STEP = 0.1
KEY_RIGHT = 262
KEY_LEFT = 263
KEY_DOWN = 264
KEY_UP = 265
KEY_Z_UP = ord("i")
KEY_Z_DOWN = ord("k")
KEY_LEFT_SHIFT = 340
KEY_RIGHT_SHIFT = 344
CARDBOARD_BOX_BODY = "cardboard_box"
CARDBOARD_BOX_FREEJOINT = "cardboard_box_freejoint"
WOODEN_TRAY_BODY = "wooden_tray"
TACTILE_BOX_BODY = CARDBOARD_BOX_BODY
TACTILE_GRIPPER_BODIES = ("ur10e_gripper_left_pad", "ur10e_gripper_right_pad")
MAX_RECORDED_CONTACTS = 16
RANDOMIZE_BOX_WALL_MARGIN = 0.020
RANDOMIZE_BOX_MAX_ATTEMPTS = 100
RANDOMIZE_BOX_DEFAULT_REGION = "front_right"
ROBOCASA_ISLAND_FIXTURE_MARKER = "island_island_group"
DEFAULT_RANDOM_SCENE_STYLE_IDS = tuple(range(1, 61))
RANDOM_SCENE_GENERATION_MAX_ATTEMPTS = 20
OBJECT_SUPPORT_CLEARANCE = 0.002
FRAGILE_WALL_BREAK_FORCE_THRESHOLD = 4000.0
FRAGILE_WALL_SHARD_STASH_POS = np.array([0.0, -1.45, -1.0], dtype=np.float64)
FRAGILE_WALL_SHARD_OFFSETS = np.array(
    [
        [0.05, 0.00, 0.02],
        [-0.05, 0.01, 0.01],
        [0.02, 0.05, 0.00],
        [-0.01, -0.05, 0.025],
        [0.04, -0.03, -0.01],
        [-0.035, 0.035, 0.02],
        [0.00, 0.00, 0.055],
        [0.055, 0.025, 0.01],
    ],
    dtype=np.float64,
)
FRAGILE_WALL_SHARD_QUATS = np.array(
    [
        [1.0, 0.0, 0.0, 0.0],
        [0.9239, 0.3827, 0.0, 0.0],
        [0.9239, 0.0, 0.3827, 0.0],
        [0.9239, 0.0, 0.0, 0.3827],
        [0.8660, 0.2887, 0.2887, 0.2887],
        [0.8660, -0.2887, 0.2887, -0.2887],
        [0.7071, 0.7071, 0.0, 0.0],
        [0.7071, 0.0, 0.7071, 0.0],
    ],
    dtype=np.float64,
)


__all__ = (
    "ARM_ACTUATORS",
    "ARM_JOINTS",
    "Any",
    "CARDBOARD_BOX_BODY",
    "CARDBOARD_BOX_FREEJOINT",
    "DEBUG_CONTROL_INTERVAL",
    "DEBUG_VIEW_INTERVAL",
    "DEFAULT_AXIS_ROTATION",
    "DEFAULT_BTN_GRIPPER",
    "DEFAULT_BTN_RECORD",
    "DEFAULT_BTN_RESET",
    "DEFAULT_BTN_ROTATION",
    "DEFAULT_BTN_TOGGLE_CONTROL",
    "DEFAULT_EE_COMMAND_FRAME",
    "DEFAULT_RANDOM_SCENE_OUTPUT_DIR",
    "DEFAULT_RANDOM_SCENE_STYLE_IDS",
    "DEFAULT_RECORDING_DIR",
    "DEFAULT_ROBOCASA_ROOT",
    "DEFAULT_ROBOSUITE_ROOT",
    "DEFAULT_VIDEO_VIEWS",
    "DS4_BTN_GRIPPER",
    "DS4_BTN_RECORD",
    "DS4_BTN_RESET",
    "DS4_BTN_ROTATION",
    "DS4_BTN_TOGGLE_CONTROL",
    "ENTRYPOINT_PATH",
    "FRAGILE_WALL_BREAK_FORCE_THRESHOLD",
    "FRAGILE_WALL_SHARD_OFFSETS",
    "FRAGILE_WALL_SHARD_QUATS",
    "FRAGILE_WALL_SHARD_STASH_POS",
    "GRIPPER_ACTUATOR",
    "GRIPPER_TOGGLE_THRESHOLD",
    "KEY_DOWN",
    "KEY_LEFT",
    "KEY_LEFT_SHIFT",
    "KEY_RIGHT",
    "KEY_RIGHT_SHIFT",
    "KEY_UP",
    "KEY_Z_DOWN",
    "KEY_Z_UP",
    "LEFT_SIDE_VIEW_AZIMUTH",
    "LEGACY_TELEOP_REALSENSE_SRC",
    "LOCAL_T265_SERVER_PY",
    "LOCAL_TELEOP_REALSENSE_SRC",
    "MAX_EE_TARGET_POSITION_ERROR",
    "MAX_RECORDED_CONTACTS",
    "MODEL_PATH",
    "MUJOCO_ROOT",
    "OBJECT_SUPPORT_CLEARANCE",
    "Path",
    "RANDOMIZE_BOX_DEFAULT_REGION",
    "RANDOMIZE_BOX_MAX_ATTEMPTS",
    "RANDOMIZE_BOX_WALL_MARGIN",
    "RANDOM_SCENE_GENERATION_MAX_ATTEMPTS",
    "REPO_ROOT",
    "ROBOCASA_ISLAND_FIXTURE_MARKER",
    "ROTATION_SPEED",
    "SIDE_VIEW_AZIMUTH",
    "SIDE_VIEW_DISTANCE",
    "SIDE_VIEW_ELEVATION",
    "SIDE_VIEW_FPS",
    "SIDE_VIEW_HEIGHT",
    "SIDE_VIEW_LOOKAT",
    "SIDE_VIEW_WIDTH",
    "SIDE_VIEW_WINDOW",
    "T265_ROTATION_DEADBAND",
    "T265_TRANSLATION_DEADBAND",
    "TACTILE_BOX_BODY",
    "TACTILE_GRIPPER_BODIES",
    "TELEOP_REALSENSE_SRC",
    "TRANSLATION_SPEED",
    "VIDEO_FPS",
    "VIDEO_HEIGHT",
    "VIDEO_RENDER_EVERY",
    "VIDEO_WIDTH",
    "VIEW_STEP",
    "VISUAL_GEOM_GROUP",
    "WOODEN_TRAY_BODY",
    "WORKSPACE_VIEW_AZIMUTH",
    "WORKSPACE_VIEW_DISTANCE",
    "WORKSPACE_VIEW_ELEVATION",
    "WORKSPACE_VIEW_REAR_SIDE_OFFSET",
    "WORLD_VIEW_AZIMUTH",
    "WORLD_VIEW_DISTANCE",
    "WORLD_VIEW_ELEVATION",
    "WORLD_VIEW_GLASS_WALL_Z_OFFSET",
    "WORLD_VIEW_MAX_DISTANCE",
    "WORLD_VIEW_MIN_DISTANCE",
    "WRIST_CAMERA_NAME",
    "ZOOM_STEP",
    "argparse",
    "contextlib",
    "dataclass",
    "datetime",
    "io",
    "json",
    "mujoco",
    "np",
    "os",
    "re",
    "subprocess",
    "sys",
    "threading",
    "time",
    "warnings",
)
