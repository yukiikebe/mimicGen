# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
robomimic-style wrapper for the custom MuJoCo cardboard-box scene used by local
teleoperation recordings.
"""
import os
import re
import subprocess
import sys
import hashlib
from pathlib import Path

import mujoco
import numpy as np
from robomimic.envs.env_base import EnvBase


def _import_teleop_helpers(model_xml=None):
    candidates = []
    env_root = os.environ.get("CUSTOM_MUJOCO_REPO")
    if env_root:
        candidates.append(Path(env_root).expanduser())
    repo_root = Path(__file__).resolve().parents[2]
    candidates.append(repo_root / "examples" / "panda_omron_cardboard_box")
    if model_xml:
        model_path = Path(model_xml).expanduser()
        parts = model_path.parts
        if "third_party" in parts:
            candidates.append(Path(*parts[: parts.index("third_party")]))
        candidates.append(model_path.parent)
    candidates.append(Path.cwd())

    for root in candidates:
        if (root / "teleop_custom_scene_t265.py").exists():
            root_str = str(root)
            if root_str not in sys.path:
                sys.path.insert(0, root_str)
            import teleop_custom_scene_t265 as teleop_helpers

            return teleop_helpers
    raise ImportError(
        "Could not find teleop_custom_scene_t265.py. Set CUSTOM_MUJOCO_REPO to "
        "the custom MuJoCo repo root."
    )


def _compat_xml_path(model_xml):
    source_path = Path(model_xml).expanduser().resolve()
    cache_dir = Path("/tmp/mimicgen_mujoco_compat_xml")
    cache_dir.mkdir(parents=True, exist_ok=True)
    digest = hashlib.sha1(str(source_path).encode("utf-8")).hexdigest()[:12]
    expanded_path = cache_dir / f"{source_path.stem}_{digest}_expanded.xml"
    compat_path = cache_dir / f"{source_path.stem}_{digest}_compat.xml"
    if compat_path.exists():
        return str(compat_path)

    mujoco3_python = os.environ.get(
        "MUJOCO3_PYTHON",
        sys.executable,
    )
    subprocess.run(
        [
            mujoco3_python,
            "-c",
            (
                "import mujoco, sys; "
                "model = mujoco.MjModel.from_xml_path(sys.argv[1]); "
                "mujoco.mj_saveLastXML(sys.argv[2], model)"
            ),
            str(source_path),
            str(expanded_path),
        ],
        check=True,
    )

    asset_files = {}
    for root in [source_path.parent, source_path.parents[2]]:
        for path in root.rglob("*"):
            if path.is_file():
                asset_files.setdefault(path.name, path)

    xml = expanded_path.read_text()
    for attr in ["colorspace", "content_type"]:
        xml = re.sub(rf'\s{attr}="[^"]*"', "", xml)
    xml = re.sub(r'(<light\b[^>]*)\stype="[^"]*"', r"\1", xml)
    xml = xml.replace('integrator="implicitfast"', 'integrator="implicit"')
    xml = xml.replace(
        '<compiler angle="radian"/>',
        '<compiler angle="radian" autolimits="true"/>',
    )

    def resolve_asset(match):
        value = match.group(1)
        if value.startswith("/"):
            path = Path(value)
        elif "/" in value:
            path = (source_path.parent / value).resolve()
        else:
            path = asset_files.get(value, source_path.parent / value)
        return f'file="{path}"'

    xml = re.sub(r'file="([^"]+)"', resolve_asset, xml)
    compat_path.write_text(xml)
    return str(compat_path)


def _load_model(model_xml):
    try:
        return mujoco.MjModel.from_xml_path(model_xml), model_xml
    except ValueError:
        compat_path = _compat_xml_path(model_xml)
        return mujoco.MjModel.from_xml_path(compat_path), compat_path


class EnvCustomCardboardBox(EnvBase):
    """
    Minimal robomimic EnvBase implementation for MimicGen data generation on the
    custom cardboard-box MuJoCo scene.
    """

    def __init__(
        self,
        env_name,
        model_xml,
        render=False,
        render_offscreen=False,
        use_image_obs=False,
        postprocess_visual_obs=True,
        task_name=None,
        source_format=None,
        goal_definition=None,
        randomize_on_reset=True,
        fragile_wall_break_force=None,
        seed=1,
        camera_names=None,
        camera_height=84,
        camera_width=84,
        **kwargs,
    ):
        del postprocess_visual_obs, source_format, kwargs
        self._env_name = env_name
        self.model_xml = str(Path(model_xml).expanduser())
        self.task_name = task_name or env_name
        self.render_enabled = render
        self.render_offscreen = render_offscreen or use_image_obs
        self.use_image_obs = use_image_obs
        self.camera_names = list(camera_names or [])
        self.camera_height = camera_height
        self.camera_width = camera_width
        self.randomize_on_reset = bool(randomize_on_reset)
        self.rng = np.random.default_rng(seed)

        self.helpers = _import_teleop_helpers(self.model_xml)
        self.model, self.loaded_model_xml = _load_model(self.model_xml)
        self.data = mujoco.MjData(self.model)
        self.runtime_state = {"fragile_wall_broken": False}
        break_force = (
            self.helpers.FRAGILE_WALL_BREAK_FORCE_THRESHOLD
            if fragile_wall_break_force is None
            else fragile_wall_break_force
        )
        self.fragile_wall = self.helpers.FragileWall(
            self.model,
            runtime_state=self.runtime_state,
            break_force_threshold=break_force,
        )
        self.goal_definition = goal_definition or {
            "type": "fragile_wall_lift",
            "stable_steps": 15,
            "margin": 0.0,
        }
        self.goal_evaluator = self.helpers.GoalEvaluator(
            self.model,
            self.goal_definition,
            runtime_state=self.runtime_state,
        )
        self.controller = self.helpers.CartesianController(self.model, self.data)
        self.arm_actuator_ids = np.asarray(
            [self.helpers.get_actuator_id(self.model, name) for name in self.helpers.ARM_ACTUATORS],
            dtype=np.int32,
        )
        self.gripper_actuator_id = self.helpers.get_actuator_id(
            self.model,
            self.helpers.GRIPPER_ACTUATOR,
        )
        self.object_names = ("cardboard_box", "wooden_tray")
        self.renderer = None

        self.reset()

    @property
    def base_env(self):
        return self

    def _set_state_vector(self, state):
        state = np.asarray(state, dtype=np.float64)
        nq = self.model.nq
        nv = self.model.nv
        if state.shape[-1] != nq + nv:
            raise ValueError(
                "Expected state vector with nq + nv = {} entries, got {}".format(
                    nq + nv,
                    state.shape[-1],
                )
            )
        self.data.qpos[:] = state[:nq]
        self.data.qvel[:] = state[nq : nq + nv]
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset_target_to_current()

    def reset(self):
        self.helpers.set_initial_pose(self.model, self.data)
        self.fragile_wall.reset(self.data)
        if self.randomize_on_reset:
            self.helpers.randomize_red_box_xy_in_tray(self.model, self.data, self.rng)
        mujoco.mj_forward(self.model, self.data)
        self.controller.reset_target_to_current()
        return self.get_observation()

    def reset_to(self, state):
        if "states" in state:
            self._set_state_vector(state["states"])
        elif "qpos" in state and "qvel" in state:
            self._set_state_vector(np.concatenate([state["qpos"], state["qvel"]]))
        else:
            raise ValueError("reset_to expects a 'states' vector or qpos/qvel.")
        return self.get_observation()

    def step(self, action):
        action = np.asarray(action, dtype=np.float64)
        if action.shape[-1] != self.action_dimension:
            raise ValueError(
                "Expected action dimension {}, got {}".format(
                    self.action_dimension,
                    action.shape[-1],
                )
            )
        self.data.ctrl[self.arm_actuator_ids] = action[:6]
        if self.gripper_actuator_id >= 0:
            self.data.ctrl[self.gripper_actuator_id] = action[6]
        mujoco.mj_step(self.model, self.data)
        self.fragile_wall.update(self.data)
        return self.get_observation(), self.get_reward(), self.is_done(), {}

    def target_pose_to_action(self, target_pose):
        target_pose = np.asarray(target_pose, dtype=np.float64)
        self.controller.target_pos = target_pose[:3, 3].copy()
        self.controller.target_rot = target_pose[:3, :3].copy()
        self.controller.update_control()
        return self.data.ctrl[self.arm_actuator_ids].copy()

    def action_to_target_pose(self, action):
        del action
        pose = np.eye(4, dtype=np.float64)
        pose[:3, :3] = self.controller.target_rot
        pose[:3, 3] = self.controller.target_pos
        return pose

    def get_body_pose(self, body_name):
        return self.helpers.pose_matrix_from_body(self.model, self.data, body_name)

    def get_object_poses(self):
        return {name: self.get_body_pose(name) for name in self.object_names}

    def get_subtask_term_signals(self):
        return {
            self.goal_evaluator.name: int(self.goal_evaluator.evaluate(self.data)),
        }

    def render(self, mode="human", height=None, width=None, camera_name=None):
        if mode == "human":
            return None
        if mode != "rgb_array":
            raise ValueError("Unsupported render mode: {}".format(mode))
        height = height or self.camera_height
        width = width or self.camera_width
        if self.renderer is None:
            self.renderer = mujoco.Renderer(self.model, height=height, width=width)
        camera = camera_name
        self.renderer.update_scene(self.data, camera=camera)
        return self.renderer.render()

    def get_observation(self):
        obs = {
            "qpos": self.data.qpos.copy(),
            "qvel": self.data.qvel.copy(),
            "eef_pos": self.data.xpos[self.controller.ee_body_id].copy(),
            "object": np.concatenate(
                [pose.reshape(-1) for pose in self.get_object_poses().values()],
                axis=0,
            ),
        }
        if self.use_image_obs:
            for camera_name in self.camera_names:
                obs[f"{camera_name}_image"] = self.render(
                    mode="rgb_array",
                    height=self.camera_height,
                    width=self.camera_width,
                    camera_name=camera_name,
                )
        return obs

    def get_state(self):
        return {
            "states": np.concatenate([self.data.qpos.copy(), self.data.qvel.copy()]),
        }

    def get_reward(self):
        return float(self.is_success()["task"])

    def get_goal(self):
        return {}

    def set_goal(self, **kwargs):
        if kwargs:
            self.goal_definition = kwargs
            self.goal_evaluator = self.helpers.GoalEvaluator(
                self.model,
                self.goal_definition,
                runtime_state=self.runtime_state,
            )

    def is_done(self):
        return False

    def is_success(self):
        task_success = self.goal_evaluator.evaluate(self.data)
        return {
            "task": bool(task_success)
            and not bool(self.runtime_state.get("fragile_wall_broken", False)),
        }

    @property
    def action_dimension(self):
        return 7

    @property
    def name(self):
        return self._env_name

    @property
    def type(self):
        return "mujoco"

    @property
    def rollout_exceptions(self):
        return (mujoco.FatalError, RuntimeError, ValueError)

    def serialize(self):
        return {
            "env_name": self._env_name,
            "type": "mujoco",
            "env_kwargs": {
                "model_xml": self.model_xml,
                "loaded_model_xml": self.loaded_model_xml,
                "task_name": self.task_name,
                "source_format": "mujoco_custom_scene_npz",
                "goal_definition": self.goal_definition,
                "randomize_on_reset": self.randomize_on_reset,
            },
        }

    @classmethod
    def create_for_data_processing(
        cls,
        env_name,
        camera_names,
        camera_height,
        camera_width,
        reward_shaping,
        **kwargs,
    ):
        del reward_shaping
        return cls(
            env_name=env_name,
            camera_names=camera_names,
            camera_height=camera_height,
            camera_width=camera_width,
            **kwargs,
        )
