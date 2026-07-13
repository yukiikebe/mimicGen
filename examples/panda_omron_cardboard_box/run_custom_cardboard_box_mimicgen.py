from __future__ import annotations

import argparse
import json
import os
import sys
from pathlib import Path
import tempfile

import mujoco
import numpy as np


EXAMPLE_ROOT = Path(__file__).resolve().parent
MUJOCO_ROOT = Path(
    os.environ.get("CUSTOM_MUJOCO_REPO", EXAMPLE_ROOT)
).expanduser().resolve()
MIMICGEN_ROOT = Path(
    os.environ.get("MIMICGEN_ROOT", EXAMPLE_ROOT.parents[1])
).expanduser().resolve()
DEFAULT_CONFIG = (
    MIMICGEN_ROOT
    / "mimicgen"
    / "exps"
    / "templates"
    / "mujoco"
    / "custom_cardboard_box.json"
)


def install_mimicgen_patches() -> None:
    if str(MIMICGEN_ROOT) not in sys.path:
        sys.path.insert(0, str(MIMICGEN_ROOT))
    if str(MUJOCO_ROOT) not in sys.path:
        sys.path.insert(0, str(MUJOCO_ROOT))

    import mimicgen.env_interfaces.mujoco as mujoco_interfaces
    import mimicgen.utils.robomimic_utils as robomimic_utils
    import h5py
    from mimicgen.envs.mujoco_custom import (
        EnvCustomCardboardBox,
        _import_teleop_helpers,
        _load_model,
    )
    from teleop_custom.cameras import make_visual_only_scene_option
    from teleop_custom.common import CARDBOARD_BOX_FREEJOINT
    from teleop_custom.contacts import (
        extract_tactile_contacts,
        make_tactile_contact_filter,
    )
    from teleop_custom.control import CartesianController
    from teleop_custom.fragile_wall import FragileWall
    from teleop_custom.goals import GoalEvaluator
    from teleop_custom.placement import set_freejoint
    from teleop_custom.robots import (
        get_actuator_id,
        lock_mobile_base,
        resolve_robot_control_profile,
    )

    class PandaOmronCardboardBoxEnv(EnvCustomCardboardBox):
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
            action_repeat=2,
            reset_source_dataset=None,
            filter_source_wall_aligned_placements=True,
            source_placement_min_abs_box_y_world_y=0.85,
            freeze_eef_rotation_from_start=False,
            delay_gripper_close_until_stood=False,
            freeze_eef_rotation_after_stood=False,
            freeze_eef_rotation_after_close=False,
            script_stood_point_local_axis_down=None,
            script_stood_rotation_tilt_deg=0.0,
            script_stood_rotation_tilt_axis="y",
            stand_axis_threshold=0.85,
            script_lift_after_stood=False,
            script_retreat_regrasp_after_stood=False,
            script_contact_regrasp_after_stood=False,
            script_position_regrasp_after_stood=False,
            script_grasp_steps=80,
            script_retreat_steps=60,
            script_retreat_height=0.08,
            script_regrasp_steps=80,
            script_contact_regrasp_max_steps=120,
            script_contact_regrasp_step_size=0.006,
            script_contact_regrasp_side_offset=0.02,
            script_contact_regrasp_z_offset=0.015,
            script_contact_regrasp_min_contact_count=1,
            script_position_regrasp_steps=120,
            script_position_regrasp_side_offset=0.055,
            script_position_regrasp_z_offset=0.035,
            script_position_regrasp_xy_tolerance=0.015,
            script_position_regrasp_z_tolerance=0.015,
            script_grasp_local_offset=None,
            script_grasp_world_frame=False,
            script_grasp_world_side_offset=0.035,
            script_grasp_world_z_offset=0.02,
            script_lift_hold_steps=20,
            script_lift_steps=160,
            script_lift_height=0.22,
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
            self.action_repeat = max(1, int(action_repeat))
            self.freeze_eef_rotation_from_start = bool(freeze_eef_rotation_from_start)
            self.delay_gripper_close_until_stood = bool(delay_gripper_close_until_stood)
            self.freeze_eef_rotation_after_stood = bool(
                freeze_eef_rotation_after_stood
            )
            self.freeze_eef_rotation_after_close = bool(
                freeze_eef_rotation_after_close
            )
            self.script_stood_point_local_axis_down = (
                None
                if script_stood_point_local_axis_down is None
                else str(script_stood_point_local_axis_down)
            )
            valid_down_axes = {None, "x", "y", "z", "-x", "-y", "-z"}
            if self.script_stood_point_local_axis_down not in valid_down_axes:
                raise ValueError(
                    "script_stood_point_local_axis_down must be x, y, z, -x, -y, or -z."
                )
            self.script_stood_rotation_tilt_deg = float(
                script_stood_rotation_tilt_deg
            )
            self.script_stood_rotation_tilt_axis = str(script_stood_rotation_tilt_axis)
            if self.script_stood_rotation_tilt_axis not in {"x", "y", "z"}:
                raise ValueError("script_stood_rotation_tilt_axis must be x, y, or z.")
            self.stand_axis_threshold = float(stand_axis_threshold)
            self.script_lift_after_stood = bool(script_lift_after_stood)
            self.script_retreat_regrasp_after_stood = bool(
                script_retreat_regrasp_after_stood
            )
            self.script_contact_regrasp_after_stood = bool(
                script_contact_regrasp_after_stood
            )
            self.script_position_regrasp_after_stood = bool(
                script_position_regrasp_after_stood
            )
            self.script_grasp_steps = max(0, int(script_grasp_steps))
            self.script_retreat_steps = max(1, int(script_retreat_steps))
            self.script_retreat_height = float(script_retreat_height)
            self.script_regrasp_steps = max(1, int(script_regrasp_steps))
            self.script_contact_regrasp_max_steps = max(
                1,
                int(script_contact_regrasp_max_steps),
            )
            self.script_contact_regrasp_step_size = float(
                script_contact_regrasp_step_size
            )
            self.script_contact_regrasp_side_offset = float(
                script_contact_regrasp_side_offset
            )
            self.script_contact_regrasp_z_offset = float(script_contact_regrasp_z_offset)
            self.script_contact_regrasp_min_contact_count = max(
                1,
                int(script_contact_regrasp_min_contact_count),
            )
            self.script_position_regrasp_steps = max(
                1,
                int(script_position_regrasp_steps),
            )
            self.script_position_regrasp_side_offset = float(
                script_position_regrasp_side_offset
            )
            self.script_position_regrasp_z_offset = float(
                script_position_regrasp_z_offset
            )
            self.script_position_regrasp_xy_tolerance = float(
                script_position_regrasp_xy_tolerance
            )
            self.script_position_regrasp_z_tolerance = float(
                script_position_regrasp_z_tolerance
            )
            if script_grasp_local_offset is None:
                script_grasp_local_offset = (0.0, 0.052, 0.045)
            self.script_grasp_local_offset = np.asarray(
                script_grasp_local_offset,
                dtype=np.float64,
            )
            if self.script_grasp_local_offset.shape != (3,):
                raise ValueError("script_grasp_local_offset must have 3 values.")
            self.script_grasp_world_frame = bool(script_grasp_world_frame)
            self.script_grasp_world_side_offset = float(script_grasp_world_side_offset)
            self.script_grasp_world_z_offset = float(script_grasp_world_z_offset)
            self.script_lift_hold_steps = max(0, int(script_lift_hold_steps))
            self.script_lift_steps = max(1, int(script_lift_steps))
            self.script_lift_height = float(script_lift_height)
            self.script_lift_active = False
            self.script_lift_step = 0
            self.script_lift_close_step = None
            self.script_lift_start_pose = None
            self.script_retreat_start_pose = None
            self.script_retreat_end_pose = None
            self.script_regrasp_target_pose = None
            self.script_contact_regrasp_contacted = False
            self.script_contact_regrasp_timed_out = False
            self.script_position_regrasp_done = False
            self.freeze_eef_rotation_active = False
            self.frozen_eef_rotation = None
            self.rng = np.random.default_rng(seed)
            self.filter_source_wall_aligned_placements = bool(
                filter_source_wall_aligned_placements
            )
            self.source_placement_min_abs_box_y_world_y = float(
                source_placement_min_abs_box_y_world_y
            )
            self.source_object_qpos = self._load_source_object_qpos(
                reset_source_dataset
            )

            self.helpers = _import_teleop_helpers(self.model_xml)
            self.model, self.loaded_model_xml = _load_model(self.model_xml)
            self.data = mujoco.MjData(self.model)
            self.robot_profile = resolve_robot_control_profile(self.model)
            self.runtime_state = {"fragile_wall_broken": False}
            break_force = (
                self.helpers.FRAGILE_WALL_BREAK_FORCE_THRESHOLD
                if fragile_wall_break_force is None
                else fragile_wall_break_force
            )
            self.fragile_wall = FragileWall(
                self.model,
                runtime_state=self.runtime_state,
                robot_profile=self.robot_profile,
                break_force_threshold=break_force,
            )
            self.goal_definition = goal_definition or {
                "type": "fragile_wall_lift",
                "stable_steps": 15,
                "margin": 0.0,
            }
            self.goal_evaluator = GoalEvaluator(
                self.model,
                self.goal_definition,
                runtime_state=self.runtime_state,
            )
            self.controller = CartesianController(
                self.model,
                self.data,
                self.robot_profile,
            )
            self.tactile_contact_filter = make_tactile_contact_filter(
                self.model,
                self.robot_profile.tactile_gripper_bodies,
            )
            self.tactile_gripper_body_ids = tuple(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
                for name in self.robot_profile.tactile_gripper_bodies
            )
            self.gripper_pad_collision_geom_ids = tuple(
                mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_GEOM, name)
                for name in (
                    "gripper0_right_finger1_pad_collision",
                    "gripper0_right_finger2_pad_collision",
                )
            )
            self.cardboard_collision_geom_id = mujoco.mj_name2id(
                self.model,
                mujoco.mjtObj.mjOBJ_GEOM,
                "cardboard_collision",
            )
            self.arm_actuator_ids = np.asarray(
                [
                    get_actuator_id(self.model, name)
                    for name in self.robot_profile.arm_actuators
                ],
                dtype=np.int32,
            )
            self.gripper_actuator_ids = np.asarray(
                [
                    get_actuator_id(self.model, name)
                    for name in self.robot_profile.gripper_actuators
                ],
                dtype=np.int32,
            )
            if np.any(self.arm_actuator_ids < 0) or np.any(
                self.gripper_actuator_ids < 0
            ):
                raise ValueError(f"Missing actuator for {self.robot_profile.name}")
            self.object_names = ("cardboard_box", "wooden_tray")
            self.gripper_open_ctrl = np.asarray(
                self.robot_profile.gripper_open_ctrl,
                dtype=np.float64,
            )
            self.gripper_closed_ctrl = np.asarray(
                self.robot_profile.gripper_closed_ctrl,
                dtype=np.float64,
            )
            self.renderer = None
            self.scene_option = make_visual_only_scene_option()

            self.reset()

        @staticmethod
        def _pose_matrix_to_freejoint_qpos(pose):
            pose = np.asarray(pose, dtype=np.float64)
            quat = np.zeros(4, dtype=np.float64)
            mujoco.mju_mat2Quat(quat, pose[:3, :3].reshape(9))
            return np.concatenate([pose[:3, 3], quat], axis=0)

        def _load_source_object_qpos(self, dataset_path):
            if dataset_path is None:
                return None
            path = Path(dataset_path).expanduser()
            if not path.is_absolute():
                path = (MUJOCO_ROOT / path).resolve()
            qpos = []
            with h5py.File(path, "r") as h5:
                for demo_name in sorted(h5["data"].keys()):
                    pose = h5[
                        f"data/{demo_name}/datagen_info/object_poses/cardboard_box"
                    ][0]
                    if self.filter_source_wall_aligned_placements:
                        box_y_world_y = abs(float(pose[1, 1]))
                        if (
                            box_y_world_y
                            < self.source_placement_min_abs_box_y_world_y
                        ):
                            continue
                    qpos.append(self._pose_matrix_to_freejoint_qpos(pose))
            if not qpos:
                raise ValueError(
                    f"No source placements found in {path} after filtering."
                )
            return np.asarray(qpos, dtype=np.float64)

        def reset(self):
            self.helpers.set_initial_pose(self.model, self.data, self.robot_profile)
            lock_mobile_base(self.model, self.data, self.robot_profile)
            self.fragile_wall.reset(self.data)
            if self.randomize_on_reset:
                if self.source_object_qpos is None:
                    self.helpers.randomize_red_box_xy_in_tray(
                        self.model,
                        self.data,
                        self.rng,
                    )
                else:
                    index = int(self.rng.integers(len(self.source_object_qpos)))
                    set_freejoint(
                        self.model,
                        self.data,
                        CARDBOARD_BOX_FREEJOINT,
                        self.source_object_qpos[index],
                    )
            mujoco.mj_forward(self.model, self.data)
            self.controller.reset_target_to_current()
            self.script_lift_active = False
            self.script_lift_step = 0
            self.script_lift_close_step = None
            self.script_lift_start_pose = None
            self.script_retreat_start_pose = None
            self.script_retreat_end_pose = None
            self.script_regrasp_target_pose = None
            self.script_contact_regrasp_contacted = False
            self.script_contact_regrasp_timed_out = False
            self.script_position_regrasp_done = False
            self.freeze_eef_rotation_active = False
            self.frozen_eef_rotation = None
            if self.freeze_eef_rotation_from_start:
                self.freeze_current_eef_rotation()
            return self.get_observation()

        def step(self, action):
            action = self.filter_action_for_stand_gate(action)
            if action.shape[-1] != self.action_dimension:
                raise ValueError(
                    "Expected action dimension {}, got {}".format(
                        self.action_dimension,
                        action.shape[-1],
                    )
                )
            arm_dim = len(self.arm_actuator_ids)
            gripper_dim = len(self.gripper_actuator_ids)
            self.update_frozen_rotation_from_close_action(action, arm_dim, gripper_dim)
            self.filter_action_for_scripted_lift(action, arm_dim, gripper_dim)
            self.data.ctrl[self.arm_actuator_ids] = action[:arm_dim]
            self.data.ctrl[self.gripper_actuator_ids] = action[
                arm_dim : arm_dim + gripper_dim
            ]
            for _ in range(self.action_repeat):
                lock_mobile_base(self.model, self.data, self.robot_profile)
                mujoco.mj_step(self.model, self.data)
                lock_mobile_base(self.model, self.data, self.robot_profile)
                mujoco.mj_forward(self.model, self.data)
                self.fragile_wall.update(self.data)
            return self.get_observation(), self.get_reward(), self.is_done(), {}

        def target_pose_to_action(self, target_pose):
            target_pose = np.asarray(target_pose, dtype=np.float64)
            target_pose = self.filter_target_pose_for_frozen_rotation(target_pose)
            self.controller.set_target_pose(
                pos=target_pose[:3, 3],
                rot=target_pose[:3, :3],
            )
            self.controller.update_control()
            return self.data.ctrl[self.arm_actuator_ids].copy()

        def action_to_target_pose(self, action):
            del action
            pose = np.eye(4, dtype=np.float64)
            pose[:3, :3] = self.controller.target_rot
            pose[:3, 3] = self.controller.target_pos
            return pose

        def filter_target_pose_for_frozen_rotation(self, target_pose):
            if not (
                self.freeze_eef_rotation_from_start
                or self.freeze_eef_rotation_after_stood
                or self.freeze_eef_rotation_after_close
            ):
                return target_pose

            target_pose = target_pose.copy()
            if self.freeze_eef_rotation_active:
                target_pose[:3, :3] = self.frozen_eef_rotation
                return target_pose

            if not self.is_cardboard_box_stood():
                return target_pose

            self.freeze_current_eef_rotation(apply_stood_tilt=True)
            target_pose[:3, :3] = self.frozen_eef_rotation
            return target_pose

        def freeze_current_eef_rotation(self, apply_stood_tilt=False):
            if self.freeze_eef_rotation_active:
                return
            self.freeze_eef_rotation_active = True
            self.frozen_eef_rotation = self.get_body_pose(self.robot_profile.eef_body)[
                :3, :3
            ].copy()
            if apply_stood_tilt:
                self.frozen_eef_rotation = self.apply_stood_rotation_adjustment(
                    self.frozen_eef_rotation
                )

        def apply_stood_rotation_adjustment(self, rotation):
            rotation = self.point_stood_local_axis_down(rotation)
            if abs(self.script_stood_rotation_tilt_deg) < 1e-9:
                return rotation

            axis_index = {"x": 0, "y": 1, "z": 2}[
                self.script_stood_rotation_tilt_axis
            ]
            axis = np.zeros(3, dtype=np.float64)
            axis[axis_index] = 1.0
            local_delta = self.axis_angle_rotation(
                axis,
                np.deg2rad(self.script_stood_rotation_tilt_deg),
            )
            return rotation @ local_delta

        def point_stood_local_axis_down(self, rotation):
            axis_name = self.script_stood_point_local_axis_down
            if axis_name is None:
                return rotation

            sign = -1.0 if axis_name.startswith("-") else 1.0
            axis_key = axis_name[-1]
            axis_index = {"x": 0, "y": 1, "z": 2}[axis_key]
            current_axis = sign * rotation[:, axis_index]
            target_axis = np.asarray([0.0, 0.0, -1.0], dtype=np.float64)
            world_delta = self.rotation_between_vectors(current_axis, target_axis)
            return world_delta @ rotation

        @staticmethod
        def axis_angle_rotation(axis, angle):
            axis = np.asarray(axis, dtype=np.float64)
            norm = float(np.linalg.norm(axis))
            if norm < 1e-12:
                return np.eye(3, dtype=np.float64)
            axis = axis / norm
            x, y, z = axis
            c = float(np.cos(angle))
            s = float(np.sin(angle))
            one_c = 1.0 - c
            return np.asarray(
                [
                    [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
                    [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
                    [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
                ],
                dtype=np.float64,
            )

        @classmethod
        def rotation_between_vectors(cls, source, target):
            source = np.asarray(source, dtype=np.float64)
            target = np.asarray(target, dtype=np.float64)
            source_norm = float(np.linalg.norm(source))
            target_norm = float(np.linalg.norm(target))
            if source_norm < 1e-12 or target_norm < 1e-12:
                return np.eye(3, dtype=np.float64)

            source = source / source_norm
            target = target / target_norm
            dot = float(np.clip(np.dot(source, target), -1.0, 1.0))
            if dot > 1.0 - 1e-9:
                return np.eye(3, dtype=np.float64)

            if dot < -1.0 + 1e-9:
                fallback = np.asarray([1.0, 0.0, 0.0], dtype=np.float64)
                if abs(float(np.dot(source, fallback))) > 0.9:
                    fallback = np.asarray([0.0, 1.0, 0.0], dtype=np.float64)
                axis = np.cross(source, fallback)
                return cls.axis_angle_rotation(axis, np.pi)

            axis = np.cross(source, target)
            angle = float(np.arccos(dot))
            return cls.axis_angle_rotation(axis, angle)

        def update_frozen_rotation_from_close_action(self, action, arm_dim, gripper_dim):
            if not self.freeze_eef_rotation_after_close or gripper_dim == 0:
                return
            gripper_action = np.asarray(
                action[arm_dim : arm_dim + gripper_dim],
                dtype=np.float64,
            )
            open_dist = float(np.linalg.norm(gripper_action - self.gripper_open_ctrl))
            closed_dist = float(
                np.linalg.norm(gripper_action - self.gripper_closed_ctrl)
            )
            if open_dist > 1e-4 and closed_dist <= open_dist:
                self.freeze_current_eef_rotation()

        def render(self, mode="human", height=None, width=None, camera_name=None):
            if mode == "human":
                return None
            if mode != "rgb_array":
                raise ValueError("Unsupported render mode: {}".format(mode))
            height = height or self.camera_height
            width = width or self.camera_width
            if self.renderer is None:
                self.renderer = mujoco.Renderer(
                    self.model,
                    height=height,
                    width=width,
                )
            self.renderer.update_scene(
                self.data,
                camera=camera_name,
                scene_option=self.scene_option,
            )
            return self.renderer.render()

        def filter_action_for_stand_gate(self, action):
            action = np.asarray(action, dtype=np.float64)
            arm_dim = len(self.arm_actuator_ids)
            gripper_dim = len(self.gripper_actuator_ids)
            if (
                self.delay_gripper_close_until_stood
                and gripper_dim > 0
                and not self.is_cardboard_box_stood()
            ):
                action[arm_dim : arm_dim + gripper_dim] = self.gripper_open_ctrl
            return action

        def filter_action_for_scripted_lift(self, action, arm_dim, gripper_dim):
            if (
                not (
                    self.script_lift_after_stood
                    or self.script_retreat_regrasp_after_stood
                    or self.script_contact_regrasp_after_stood
                    or self.script_position_regrasp_after_stood
                )
                or gripper_dim == 0
            ):
                return action

            if not self.script_lift_active and not self.is_cardboard_box_stood():
                return action

            if not self.script_lift_active:
                self.script_lift_active = True
                self.script_lift_step = 0
                self.script_lift_close_step = None
                self.script_lift_start_pose = None
                self.script_retreat_start_pose = None
                self.script_retreat_end_pose = None
                self.script_regrasp_target_pose = None
                self.script_contact_regrasp_contacted = False
                self.script_contact_regrasp_timed_out = False
                self.script_position_regrasp_done = False

            if (
                self.script_retreat_regrasp_after_stood
                or self.script_contact_regrasp_after_stood
                or self.script_position_regrasp_after_stood
            ):
                target_pose = self.scripted_retreat_regrasp_pose()
                if target_pose is not None:
                    action[arm_dim : arm_dim + gripper_dim] = self.gripper_open_ctrl
                elif (
                    self.script_contact_regrasp_timed_out
                    and not self.script_contact_regrasp_contacted
                ):
                    action[arm_dim : arm_dim + gripper_dim] = self.gripper_open_ctrl
                    target_pose = self.get_body_pose(self.robot_profile.eef_body)
                else:
                    action[arm_dim : arm_dim + gripper_dim] = np.asarray(
                        self.robot_profile.gripper_closed_ctrl,
                        dtype=np.float64,
                    )
                    if self.script_lift_close_step is None:
                        self.script_lift_close_step = self.script_lift_step
                    if self.script_lift_start_pose is None:
                        self.script_lift_start_pose = self.get_body_pose(
                            self.robot_profile.eef_body
                        )

                    lift_step = max(
                        0,
                        self.script_lift_step
                        - self.script_lift_close_step
                        - self.script_lift_hold_steps,
                    )
                    lift_fraction = min(1.0, lift_step / self.script_lift_steps)
                    target_pose = self.script_lift_start_pose.copy()
                    target_pose[2, 3] += self.script_lift_height * lift_fraction
            elif self.script_lift_step < self.script_grasp_steps:
                action[arm_dim : arm_dim + gripper_dim] = self.gripper_open_ctrl
                target_pose = self.scripted_grasp_pose()
            else:
                action[arm_dim : arm_dim + gripper_dim] = np.asarray(
                    self.robot_profile.gripper_closed_ctrl,
                    dtype=np.float64,
                )
                if self.script_lift_start_pose is None:
                    self.script_lift_start_pose = self.get_body_pose(
                        self.robot_profile.eef_body
                    )

                lift_step = max(
                    0,
                    self.script_lift_step
                    - self.script_grasp_steps
                    - self.script_lift_hold_steps,
                )
                lift_fraction = min(1.0, lift_step / self.script_lift_steps)
                target_pose = self.script_lift_start_pose.copy()
                target_pose[2, 3] += self.script_lift_height * lift_fraction

            target_pose = self.filter_target_pose_for_frozen_rotation(target_pose)
            self.controller.set_target_pose(
                pos=target_pose[:3, 3],
                rot=target_pose[:3, :3],
            )
            self.controller.update_control()
            action[:arm_dim] = self.data.ctrl[self.arm_actuator_ids]
            self.script_lift_step += 1
            return action

        def scripted_retreat_regrasp_pose(self):
            if self.script_lift_step < self.script_retreat_steps:
                if self.script_retreat_start_pose is None:
                    self.script_retreat_start_pose = self.get_body_pose(
                        self.robot_profile.eef_body
                    )
                    self.script_retreat_end_pose = self.script_retreat_start_pose.copy()
                    self.script_retreat_end_pose[2, 3] += self.script_retreat_height

                fraction = (self.script_lift_step + 1) / self.script_retreat_steps
                return self.interpolate_pose_position(
                    self.script_retreat_start_pose,
                    self.script_retreat_end_pose,
                    fraction,
                )

            regrasp_step = self.script_lift_step - self.script_retreat_steps
            if self.script_position_regrasp_after_stood:
                return self.scripted_position_regrasp_pose(regrasp_step)

            if self.script_contact_regrasp_after_stood:
                return self.scripted_contact_regrasp_pose(regrasp_step)

            if regrasp_step < self.script_regrasp_steps:
                if self.script_regrasp_target_pose is None:
                    if self.script_retreat_end_pose is None:
                        self.script_retreat_end_pose = self.get_body_pose(
                            self.robot_profile.eef_body
                        )
                    self.script_regrasp_target_pose = self.scripted_grasp_pose()

                fraction = (regrasp_step + 1) / self.script_regrasp_steps
                return self.interpolate_pose_position(
                    self.script_retreat_end_pose,
                    self.script_regrasp_target_pose,
                    fraction,
                )

            return None

        def scripted_contact_regrasp_pose(self, regrasp_step):
            if (
                self.gripper_box_contact_count()
                >= self.script_contact_regrasp_min_contact_count
            ):
                self.script_contact_regrasp_contacted = True
                self.script_contact_regrasp_timed_out = False
                return None
            if regrasp_step >= self.script_contact_regrasp_max_steps:
                self.script_contact_regrasp_timed_out = True
                return self.get_body_pose(self.robot_profile.eef_body)

            box_pose = self.get_body_pose("cardboard_box")
            eef_pose = self.get_body_pose(self.robot_profile.eef_body)
            pad_contact_point = self.gripper_pad_face_midpoint(box_pose[:3, 3])
            side_xy = pad_contact_point[:2] - box_pose[:2, 3]
            side_norm = float(np.linalg.norm(side_xy))
            if side_norm < 1e-6:
                side_xy = eef_pose[:2, 3] - box_pose[:2, 3]
                side_norm = float(np.linalg.norm(side_xy))
            if side_norm < 1e-6:
                side_xy = np.asarray([0.0, 1.0], dtype=np.float64)
                side_norm = 1.0

            desired_pad_contact_point = self.cardboard_box_side_contact_point(
                side_xy / side_norm,
                self.script_contact_regrasp_side_offset,
                self.script_contact_regrasp_z_offset,
            )

            delta = desired_pad_contact_point - pad_contact_point
            xy_norm = float(np.linalg.norm(delta[:2]))
            if xy_norm > self.script_contact_regrasp_step_size:
                delta[:2] *= self.script_contact_regrasp_step_size / xy_norm
            delta[2] = float(
                np.clip(
                    delta[2],
                    -self.script_contact_regrasp_step_size,
                    self.script_contact_regrasp_step_size,
                )
            )

            target_pose = eef_pose.copy()
            target_pose[:3, 3] += delta
            return target_pose

        def scripted_position_regrasp_pose(self, regrasp_step):
            if self.script_position_regrasp_done:
                return None

            if self.script_regrasp_target_pose is None:
                self.script_regrasp_target_pose = (
                    self.scripted_position_regrasp_target_pose()
                )

            eef_pose = self.get_body_pose(self.robot_profile.eef_body)
            target_pose = self.script_regrasp_target_pose.copy()
            target_pose[:3, :3] = eef_pose[:3, :3]
            target_pose = self.filter_target_pose_for_frozen_rotation(target_pose)

            position_error = target_pose[:3, 3] - eef_pose[:3, 3]
            xy_error = float(np.linalg.norm(position_error[:2]))
            z_error = abs(float(position_error[2]))
            if (
                xy_error <= self.script_position_regrasp_xy_tolerance
                and z_error <= self.script_position_regrasp_z_tolerance
            ):
                self.script_position_regrasp_done = True
                return None

            if regrasp_step >= self.script_position_regrasp_steps:
                self.script_position_regrasp_done = True
                return None

            fraction = min(1.0, (regrasp_step + 1) / self.script_position_regrasp_steps)
            if self.script_retreat_end_pose is None:
                self.script_retreat_end_pose = self.get_body_pose(
                    self.robot_profile.eef_body
                )
            return self.interpolate_pose_position(
                self.script_retreat_end_pose,
                target_pose,
                fraction,
            )

        def scripted_position_regrasp_target_pose(self):
            box_pose = self.get_body_pose("cardboard_box")
            eef_pose = self.get_body_pose(self.robot_profile.eef_body)
            side_xy = eef_pose[:2, 3] - box_pose[:2, 3]
            side_norm = float(np.linalg.norm(side_xy))
            if side_norm < 1e-6:
                side_xy = np.asarray([0.0, 1.0], dtype=np.float64)
                side_norm = 1.0

            target_pose = eef_pose.copy()
            target_pose[:2, 3] = (
                box_pose[:2, 3]
                + side_xy / side_norm * self.script_position_regrasp_side_offset
            )
            target_pose[2, 3] = (
                box_pose[2, 3] + self.script_position_regrasp_z_offset
            )
            return target_pose

        def gripper_pad_face_midpoint(self, target_position):
            positions = []
            for geom_id in self.gripper_pad_collision_geom_ids:
                if geom_id < 0:
                    continue
                positions.append(
                    self.geom_face_point_toward_position(geom_id, target_position)
                )
            if positions:
                return np.mean(np.asarray(positions, dtype=np.float64), axis=0)
            return self.gripper_tip_midpoint()

        def geom_face_point_toward_position(self, geom_id, target_position):
            center = self.data.geom_xpos[geom_id].copy()
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            size = self.model.geom_size[geom_id]
            direction = np.asarray(target_position, dtype=np.float64) - center
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                return center

            direction /= norm
            axis_index = int(
                np.argmax(
                    [
                        abs(float(np.dot(rotation[:, axis], direction)))
                        for axis in range(3)
                    ]
                )
            )
            axis = rotation[:, axis_index]
            sign = 1.0 if float(np.dot(axis, direction)) >= 0.0 else -1.0
            return center + sign * axis * float(size[axis_index])

        def cardboard_box_side_contact_point(
            self,
            side_direction_xy,
            side_clearance,
            z_offset,
        ):
            if self.cardboard_collision_geom_id < 0:
                center = self.get_body_pose("cardboard_box")[:3, 3].copy()
                point = center.copy()
                point[:2] += np.asarray(side_direction_xy, dtype=np.float64) * float(
                    side_clearance
                )
                point[2] = center[2] + float(z_offset)
                return point

            geom_id = self.cardboard_collision_geom_id
            center = self.data.geom_xpos[geom_id].copy()
            rotation = self.data.geom_xmat[geom_id].reshape(3, 3)
            size = self.model.geom_size[geom_id]
            direction = np.asarray(
                [side_direction_xy[0], side_direction_xy[1], 0.0],
                dtype=np.float64,
            )
            norm = float(np.linalg.norm(direction))
            if norm < 1e-9:
                direction[:] = (0.0, 1.0, 0.0)
            else:
                direction /= norm

            half_extent = sum(
                abs(float(np.dot(rotation[:, axis], direction))) * float(size[axis])
                for axis in range(3)
            )
            point = center + direction * (half_extent + float(side_clearance))
            point[2] = center[2] + float(z_offset)
            return point

        def gripper_tip_midpoint(self):
            positions = [
                self.data.xpos[body_id]
                for body_id in self.tactile_gripper_body_ids
                if body_id >= 0
            ]
            if not positions:
                return self.get_body_pose(self.robot_profile.eef_body)[:3, 3]
            return np.mean(np.asarray(positions, dtype=np.float64), axis=0)

        def gripper_box_contact_count(self):
            if (
                self.cardboard_collision_geom_id >= 0
                and all(geom_id >= 0 for geom_id in self.gripper_pad_collision_geom_ids)
            ):
                count = 0
                pad_geom_ids = set(self.gripper_pad_collision_geom_ids)
                for contact_id in range(self.data.ncon):
                    contact = self.data.contact[contact_id]
                    geom1 = int(contact.geom1)
                    geom2 = int(contact.geom2)
                    if (
                        geom1 == self.cardboard_collision_geom_id
                        and geom2 in pad_geom_ids
                    ) or (
                        geom2 == self.cardboard_collision_geom_id
                        and geom1 in pad_geom_ids
                    ):
                        count += 1
                return count

            contacts = extract_tactile_contacts(
                self.model,
                self.data,
                self.tactile_contact_filter,
            )
            return int(contacts["contact_count"])

        @staticmethod
        def interpolate_pose_position(start_pose, end_pose, fraction):
            fraction = float(np.clip(fraction, 0.0, 1.0))
            pose = np.asarray(start_pose, dtype=np.float64).copy()
            start_pos = np.asarray(start_pose, dtype=np.float64)[:3, 3]
            end_pos = np.asarray(end_pose, dtype=np.float64)[:3, 3]
            pose[:3, 3] = start_pos + fraction * (end_pos - start_pos)
            return pose

        def scripted_grasp_pose(self):
            box_pose = self.get_body_pose("cardboard_box")
            eef_pose = self.get_body_pose(self.robot_profile.eef_body)
            if self.script_grasp_world_frame:
                offset_xy = eef_pose[:2, 3] - box_pose[:2, 3]
                norm_xy = float(np.linalg.norm(offset_xy))
                if norm_xy < 1e-6:
                    offset_xy = np.asarray([0.0, 1.0], dtype=np.float64)
                    norm_xy = 1.0
                grasp_pose = eef_pose.copy()
                grasp_pose[:2, 3] = (
                    box_pose[:2, 3]
                    + offset_xy / norm_xy * self.script_grasp_world_side_offset
                )
                grasp_pose[2, 3] = (
                    box_pose[2, 3] + self.script_grasp_world_z_offset
                )
                return grasp_pose

            rel_eef = box_pose[:3, :3].T @ (eef_pose[:3, 3] - box_pose[:3, 3])
            side_sign = 1.0 if rel_eef[1] >= 0.0 else -1.0
            local_grasp_offset = self.script_grasp_local_offset.copy()
            local_grasp_offset[1] = side_sign * abs(local_grasp_offset[1])
            grasp_pose = eef_pose.copy()
            grasp_pose[:3, 3] = box_pose[:3, 3] + box_pose[:3, :3] @ local_grasp_offset
            return grasp_pose

        def is_cardboard_box_stood(self):
            pose = self.get_body_pose("cardboard_box")
            rot = pose[:3, :3]
            vertical_axis_score = max(abs(rot[2, 0]), abs(rot[2, 1]))
            return vertical_axis_score >= self.stand_axis_threshold

        @property
        def action_dimension(self):
            return len(self.arm_actuator_ids) + len(self.gripper_actuator_ids)

    def get_robot_eef_pose(self):
        return self.env.get_body_pose(self.env.robot_profile.eef_body)

    def action_to_gripper_action(self, action):
        if hasattr(self.env, "filter_action_for_stand_gate"):
            action = self.env.filter_action_for_stand_gate(action)
        arm_dim = len(self.env.arm_actuator_ids)
        gripper_dim = len(self.env.gripper_actuator_ids)
        return np.asarray(action[arm_dim : arm_dim + gripper_dim], dtype=np.float64)

    mujoco_interfaces.CustomMuJoCoInterface.get_robot_eef_pose = get_robot_eef_pose
    mujoco_interfaces.CustomMuJoCoInterface.action_to_gripper_action = (
        action_to_gripper_action
    )
    robomimic_utils.EnvCustomCardboardBox = PandaOmronCardboardBoxEnv


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Run MimicGen for the local PandaOmron cardboard-box scene."
    )
    parser.add_argument("--config", default=str(DEFAULT_CONFIG))
    parser.add_argument("--source", default=None)
    parser.add_argument("--folder", default=None)
    parser.add_argument("--num-demos", type=int, default=None)
    parser.add_argument(
        "--guarantee-success",
        action="store_true",
        help="Keep generating until --num-demos successful trajectories are saved.",
    )
    parser.add_argument("--seed", type=int, default=None)
    parser.add_argument("--debug", action="store_true")
    parser.add_argument("--auto-remove-exp", action="store_true")
    parser.add_argument("--video-path", default=None)
    parser.add_argument("--video-skip", type=int, default=5)
    parser.add_argument("--render", action="store_true")
    parser.add_argument("--no-randomize-on-reset", action="store_true")
    parser.add_argument("--transform-first-robot-pose", action="store_true")
    parser.add_argument("--interpolate-from-current-pose", action="store_true")
    parser.add_argument("--select-src-per-subtask", action="store_true")
    parser.add_argument("--num-interpolation-steps", type=int, default=None)
    parser.add_argument("--contact-lift-subtasks", action="store_true")
    parser.add_argument("--contact-stand-lift-subtasks", action="store_true")
    parser.add_argument("--contact-stand-grasp-lift-subtasks", action="store_true")
    parser.add_argument(
        "--freeze-eef-rotation-from-start",
        action="store_true",
        help=(
            "Keep the EEF target rotation fixed from reset. This prevents "
            "cardboard_box-relative SE(3) transforms from rotating the gripper "
            "before the box reaches the upright signal."
        ),
    )
    parser.add_argument(
        "--delay-gripper-close-until-stood",
        action="store_true",
        help="Keep the gripper open at runtime until cardboard_box is upright.",
    )
    parser.add_argument(
        "--freeze-eef-rotation-after-stood",
        action="store_true",
        help=(
            "After cardboard_box first becomes upright, keep the EEF target "
            "rotation fixed and only follow MimicGen target positions."
        ),
    )
    parser.add_argument(
        "--freeze-eef-rotation-after-close",
        action="store_true",
        help=(
            "After the gripper close command first appears, keep the EEF target "
            "rotation fixed and only follow MimicGen target positions."
        ),
    )
    parser.add_argument(
        "--script-stood-rotation-tilt-deg",
        type=float,
        default=0.0,
        help=(
            "Signed local-axis tilt, in degrees, applied once when freezing "
            "EEF rotation after cardboard_box first becomes upright."
        ),
    )
    parser.add_argument(
        "--script-stood-point-local-axis-down",
        choices=("x", "y", "z", "-x", "-y", "-z"),
        default=None,
        help=(
            "When freezing EEF rotation after upright, rotate the gripper so "
            "this local EEF axis points toward world down."
        ),
    )
    parser.add_argument(
        "--script-stood-rotation-tilt-axis",
        choices=("x", "y", "z"),
        default="y",
        help="Local EEF axis for --script-stood-rotation-tilt-deg.",
    )
    parser.add_argument("--stand-axis-threshold", type=float, default=0.85)
    parser.add_argument(
        "--script-lift-after-stood",
        action="store_true",
        help="After upright close, override arm targets with a vertical lift script.",
    )
    parser.add_argument(
        "--script-retreat-regrasp-after-stood",
        action="store_true",
        help=(
            "After upright, open the gripper, retreat upward, re-approach the "
            "upright box grasp pose, close, then lift."
        ),
    )
    parser.add_argument(
        "--script-contact-regrasp-after-stood",
        action="store_true",
        help=(
            "After upright, retreat, then move the gripper pad collision face "
            "toward the upright box until pad-box contact is detected before "
            "closing."
        ),
    )
    parser.add_argument(
        "--script-position-regrasp-after-stood",
        action="store_true",
        help=(
            "After upright, retreat, then move to a geometry-based open-gripper "
            "regrasp pose and close when the EEF is near the target, without "
            "using contact as the close condition."
        ),
    )
    parser.add_argument("--script-grasp-steps", type=int, default=80)
    parser.add_argument("--script-retreat-steps", type=int, default=60)
    parser.add_argument("--script-retreat-height", type=float, default=0.08)
    parser.add_argument("--script-regrasp-steps", type=int, default=80)
    parser.add_argument("--script-contact-regrasp-max-steps", type=int, default=120)
    parser.add_argument("--script-contact-regrasp-step-size", type=float, default=0.006)
    parser.add_argument("--script-contact-regrasp-side-offset", type=float, default=0.02)
    parser.add_argument("--script-contact-regrasp-z-offset", type=float, default=0.015)
    parser.add_argument("--script-contact-regrasp-min-contact-count", type=int, default=1)
    parser.add_argument("--script-position-regrasp-steps", type=int, default=120)
    parser.add_argument("--script-position-regrasp-side-offset", type=float, default=0.055)
    parser.add_argument("--script-position-regrasp-z-offset", type=float, default=0.035)
    parser.add_argument("--script-position-regrasp-xy-tolerance", type=float, default=0.015)
    parser.add_argument("--script-position-regrasp-z-tolerance", type=float, default=0.015)
    parser.add_argument(
        "--script-grasp-local-offset",
        type=float,
        nargs=3,
        default=(0.0, 0.052, 0.045),
        metavar=("X", "Y_ABS", "Z"),
        help=(
            "Local cardboard_box offset for scripted regrasp. The Y value is "
            "treated as an absolute side distance and signed at runtime."
        ),
    )
    parser.add_argument(
        "--script-grasp-world-frame",
        action="store_true",
        help=(
            "Use a world-frame upright-box regrasp target instead of a "
            "cardboard_box local offset."
        ),
    )
    parser.add_argument("--script-grasp-world-side-offset", type=float, default=0.035)
    parser.add_argument("--script-grasp-world-z-offset", type=float, default=0.02)
    parser.add_argument("--script-lift-hold-steps", type=int, default=20)
    parser.add_argument("--script-lift-steps", type=int, default=160)
    parser.add_argument("--script-lift-height", type=float, default=0.22)
    parser.add_argument(
        "--selection-strategy",
        choices=(
            "nearest_neighbor_object",
            "nearest_neighbor_robot_distance",
            "random",
        ),
        default="nearest_neighbor_object",
    )
    parser.add_argument("--reset-from-source-placements", action="store_true")
    parser.add_argument(
        "--filter-source-wall-aligned-placements",
        action=argparse.BooleanOptionalAction,
        default=True,
        help=(
            "When resetting from source placements, keep only cardboard_box "
            "orientations whose local Y axis is close to the world Y wall axis."
        ),
    )
    parser.add_argument(
        "--source-placement-min-abs-box-y-world-y",
        type=float,
        default=0.85,
        help=(
            "Minimum abs(dot(box local Y, world Y)) for "
            "--filter-source-wall-aligned-placements."
        ),
    )
    return parser.parse_args()


def make_effective_config(args: argparse.Namespace) -> str:
    subtask_mode_count = sum(
        bool(mode)
        for mode in (
            args.contact_lift_subtasks,
            args.contact_stand_lift_subtasks,
            args.contact_stand_grasp_lift_subtasks,
        )
    )
    if subtask_mode_count > 1:
        raise ValueError("Use only one contact subtask mode at a time.")
    if args.script_contact_regrasp_after_stood and args.script_position_regrasp_after_stood:
        raise ValueError(
            "Use only one scripted regrasp mode: contact or position."
        )

    config_path = Path(args.config).expanduser().resolve()
    with config_path.open("r", encoding="utf-8") as f:
        config = json.load(f)

    env_kwargs = config["experiment"]["task"]["env_meta_update_kwargs"]["env_kwargs"]
    if args.no_randomize_on_reset:
        env_kwargs["randomize_on_reset"] = False
    if args.reset_from_source_placements:
        env_kwargs["reset_source_dataset"] = (
            args.source or config["experiment"]["source"]["dataset_path"]
        )
    if (
        args.reset_from_source_placements
        or not args.filter_source_wall_aligned_placements
        or args.source_placement_min_abs_box_y_world_y != 0.85
    ):
        env_kwargs["filter_source_wall_aligned_placements"] = (
            args.filter_source_wall_aligned_placements
        )
        env_kwargs["source_placement_min_abs_box_y_world_y"] = (
            args.source_placement_min_abs_box_y_world_y
        )
    if args.freeze_eef_rotation_from_start:
        env_kwargs["freeze_eef_rotation_from_start"] = True
    if args.delay_gripper_close_until_stood:
        env_kwargs["delay_gripper_close_until_stood"] = True
        env_kwargs["stand_axis_threshold"] = args.stand_axis_threshold
    if args.freeze_eef_rotation_after_stood:
        env_kwargs["freeze_eef_rotation_after_stood"] = True
        env_kwargs["stand_axis_threshold"] = args.stand_axis_threshold
    if args.script_stood_point_local_axis_down is not None:
        env_kwargs["script_stood_point_local_axis_down"] = (
            args.script_stood_point_local_axis_down
        )
    if abs(args.script_stood_rotation_tilt_deg) > 1e-9:
        env_kwargs["script_stood_rotation_tilt_deg"] = (
            args.script_stood_rotation_tilt_deg
        )
        env_kwargs["script_stood_rotation_tilt_axis"] = (
            args.script_stood_rotation_tilt_axis
        )
    if args.freeze_eef_rotation_after_close:
        env_kwargs["freeze_eef_rotation_after_close"] = True
    if (
        args.script_lift_after_stood
        or args.script_retreat_regrasp_after_stood
        or args.script_contact_regrasp_after_stood
        or args.script_position_regrasp_after_stood
    ):
        env_kwargs["script_lift_after_stood"] = True
        env_kwargs["script_grasp_steps"] = args.script_grasp_steps
        env_kwargs["script_lift_hold_steps"] = args.script_lift_hold_steps
        env_kwargs["script_lift_steps"] = args.script_lift_steps
        env_kwargs["script_lift_height"] = args.script_lift_height
        env_kwargs["script_grasp_local_offset"] = list(args.script_grasp_local_offset)
        env_kwargs["script_grasp_world_frame"] = args.script_grasp_world_frame
        env_kwargs["script_grasp_world_side_offset"] = (
            args.script_grasp_world_side_offset
        )
        env_kwargs["script_grasp_world_z_offset"] = args.script_grasp_world_z_offset
    if args.script_retreat_regrasp_after_stood:
        env_kwargs["script_retreat_regrasp_after_stood"] = True
        env_kwargs["script_retreat_steps"] = args.script_retreat_steps
        env_kwargs["script_retreat_height"] = args.script_retreat_height
        env_kwargs["script_regrasp_steps"] = args.script_regrasp_steps
    if args.script_contact_regrasp_after_stood:
        env_kwargs["script_contact_regrasp_after_stood"] = True
        env_kwargs["script_retreat_steps"] = args.script_retreat_steps
        env_kwargs["script_retreat_height"] = args.script_retreat_height
        env_kwargs["script_contact_regrasp_max_steps"] = (
            args.script_contact_regrasp_max_steps
        )
        env_kwargs["script_contact_regrasp_step_size"] = (
            args.script_contact_regrasp_step_size
        )
        env_kwargs["script_contact_regrasp_side_offset"] = (
            args.script_contact_regrasp_side_offset
        )
        env_kwargs["script_contact_regrasp_z_offset"] = (
            args.script_contact_regrasp_z_offset
        )
        env_kwargs["script_contact_regrasp_min_contact_count"] = (
            args.script_contact_regrasp_min_contact_count
        )
    if args.script_position_regrasp_after_stood:
        env_kwargs["script_position_regrasp_after_stood"] = True
        env_kwargs["script_retreat_steps"] = args.script_retreat_steps
        env_kwargs["script_retreat_height"] = args.script_retreat_height
        env_kwargs["script_position_regrasp_steps"] = (
            args.script_position_regrasp_steps
        )
        env_kwargs["script_position_regrasp_side_offset"] = (
            args.script_position_regrasp_side_offset
        )
        env_kwargs["script_position_regrasp_z_offset"] = (
            args.script_position_regrasp_z_offset
        )
        env_kwargs["script_position_regrasp_xy_tolerance"] = (
            args.script_position_regrasp_xy_tolerance
        )
        env_kwargs["script_position_regrasp_z_tolerance"] = (
            args.script_position_regrasp_z_tolerance
        )

    generation = config["experiment"]["generation"]
    if args.guarantee_success:
        generation["guarantee"] = True
    if args.transform_first_robot_pose:
        generation["transform_first_robot_pose"] = True
    if args.interpolate_from_current_pose:
        generation["interpolate_from_last_target_pose"] = False
    if args.select_src_per_subtask:
        generation["select_src_per_subtask"] = True

    if args.num_interpolation_steps is not None:
        for subtask in config["task"]["task_spec"].values():
            subtask["num_interpolation_steps"] = args.num_interpolation_steps

    if args.contact_lift_subtasks:
        config["task"]["task_spec"] = {
            "subtask_1": {
                "object_ref": "cardboard_box",
                "subtask_term_signal": "gripper_contact",
                "subtask_term_offset_range": [0, 0],
                "selection_strategy": args.selection_strategy,
                "selection_strategy_kwargs": {"nn_k": 3},
                "action_noise": 0.0,
                "num_interpolation_steps": 10,
                "num_fixed_steps": 0,
                "apply_noise_during_interpolation": False,
            },
            "subtask_2": {
                "object_ref": "cardboard_box",
                "subtask_term_signal": "box_lifted",
                "subtask_term_offset_range": [0, 0],
                "selection_strategy": args.selection_strategy,
                "selection_strategy_kwargs": {"nn_k": 3},
                "action_noise": 0.0,
                "num_interpolation_steps": 5,
                "num_fixed_steps": 0,
                "apply_noise_during_interpolation": False,
            },
            "subtask_3": {
                "object_ref": "cardboard_box",
                "subtask_term_signal": None,
                "subtask_term_offset_range": None,
                "selection_strategy": args.selection_strategy,
                "selection_strategy_kwargs": {"nn_k": 3},
                "action_noise": 0.0,
                "num_interpolation_steps": 5,
                "num_fixed_steps": 0,
                "apply_noise_during_interpolation": False,
            },
        }

    if args.contact_stand_lift_subtasks:
        config["task"]["task_spec"] = {
            "subtask_1": {
                "object_ref": "cardboard_box",
                "subtask_term_signal": "gripper_contact",
                "subtask_term_offset_range": [0, 0],
                "selection_strategy": args.selection_strategy,
                "selection_strategy_kwargs": {"nn_k": 3},
                "action_noise": 0.0,
                "num_interpolation_steps": 10,
                "num_fixed_steps": 0,
                "apply_noise_during_interpolation": False,
            },
            "subtask_2": {
                "object_ref": "cardboard_box",
                "subtask_term_signal": "box_stood",
                "subtask_term_offset_range": [0, 0],
                "selection_strategy": args.selection_strategy,
                "selection_strategy_kwargs": {"nn_k": 3},
                "action_noise": 0.0,
                "num_interpolation_steps": 5,
                "num_fixed_steps": 0,
                "apply_noise_during_interpolation": False,
            },
            "subtask_3": {
                "object_ref": "cardboard_box",
                "subtask_term_signal": None,
                "subtask_term_offset_range": None,
                "selection_strategy": args.selection_strategy,
                "selection_strategy_kwargs": {"nn_k": 3},
                "action_noise": 0.0,
                "num_interpolation_steps": 5,
                "num_fixed_steps": 0,
                "apply_noise_during_interpolation": False,
            },
        }

    if args.contact_stand_grasp_lift_subtasks:
        config["task"]["task_spec"] = {
            "subtask_1": {
                "object_ref": "cardboard_box",
                "subtask_term_signal": "gripper_contact",
                "subtask_term_offset_range": [0, 0],
                "selection_strategy": args.selection_strategy,
                "selection_strategy_kwargs": {"nn_k": 3},
                "action_noise": 0.0,
                "num_interpolation_steps": 10,
                "num_fixed_steps": 0,
                "apply_noise_during_interpolation": False,
            },
            "subtask_2": {
                "object_ref": "cardboard_box",
                "subtask_term_signal": "box_stood",
                "subtask_term_offset_range": [0, 0],
                "selection_strategy": args.selection_strategy,
                "selection_strategy_kwargs": {"nn_k": 3},
                "action_noise": 0.0,
                "num_interpolation_steps": 5,
                "num_fixed_steps": 0,
                "apply_noise_during_interpolation": False,
            },
            "subtask_3": {
                "object_ref": "cardboard_box",
                "subtask_term_signal": "grasp_ready",
                "subtask_term_offset_range": [0, 0],
                "selection_strategy": args.selection_strategy,
                "selection_strategy_kwargs": {"nn_k": 3},
                "action_noise": 0.0,
                "num_interpolation_steps": 5,
                "num_fixed_steps": 0,
                "apply_noise_during_interpolation": False,
            },
            "subtask_4": {
                "object_ref": "cardboard_box",
                "subtask_term_signal": None,
                "subtask_term_offset_range": None,
                "selection_strategy": args.selection_strategy,
                "selection_strategy_kwargs": {"nn_k": 3},
                "action_noise": 0.0,
                "num_interpolation_steps": 5,
                "num_fixed_steps": 0,
                "apply_noise_during_interpolation": False,
            },
        }

    needs_temp_config = (
        args.no_randomize_on_reset
        or args.guarantee_success
        or args.transform_first_robot_pose
        or args.interpolate_from_current_pose
        or args.select_src_per_subtask
        or args.num_interpolation_steps is not None
        or args.contact_lift_subtasks
        or args.contact_stand_lift_subtasks
        or args.contact_stand_grasp_lift_subtasks
        or args.freeze_eef_rotation_from_start
        or args.delay_gripper_close_until_stood
        or args.freeze_eef_rotation_after_stood
        or args.freeze_eef_rotation_after_close
        or args.stand_axis_threshold != 0.85
        or args.script_lift_after_stood
        or args.script_retreat_regrasp_after_stood
        or args.script_contact_regrasp_after_stood
        or args.script_grasp_steps != 80
        or tuple(args.script_grasp_local_offset) != (0.0, 0.052, 0.045)
        or args.script_grasp_world_frame
        or args.script_grasp_world_side_offset != 0.035
        or args.script_grasp_world_z_offset != 0.02
        or args.script_retreat_steps != 60
        or args.script_retreat_height != 0.08
        or args.script_regrasp_steps != 80
        or args.script_contact_regrasp_max_steps != 120
        or args.script_contact_regrasp_step_size != 0.006
        or args.script_contact_regrasp_side_offset != 0.02
        or args.script_contact_regrasp_z_offset != 0.015
        or args.script_lift_hold_steps != 20
        or args.script_lift_steps != 160
        or args.script_lift_height != 0.22
        or args.selection_strategy != "nearest_neighbor_object"
        or args.reset_from_source_placements
        or not args.filter_source_wall_aligned_placements
        or args.source_placement_min_abs_box_y_world_y != 0.85
    )
    if not needs_temp_config:
        return str(config_path)

    output = Path(tempfile.gettempdir()) / "custom_cardboard_box_mimicgen_config.json"
    output.write_text(json.dumps(config, indent=4), encoding="utf-8")
    return str(output)


def main() -> None:
    args = parse_args()
    install_mimicgen_patches()

    from mimicgen.scripts.generate_dataset import main as generate_main

    config_path = make_effective_config(args)

    generate_args = argparse.Namespace(
        config=config_path,
        debug=args.debug,
        auto_remove_exp=args.auto_remove_exp,
        render=args.render,
        video_path=args.video_path,
        video_skip=args.video_skip,
        render_image_names=None,
        pause_subtask=False,
        source=args.source,
        task_name=None,
        folder=args.folder,
        num_demos=args.num_demos,
        seed=args.seed,
    )
    generate_main(generate_args)


if __name__ == "__main__":
    main()
