from __future__ import annotations

from .common import Any, Path, json, mujoco, np
from .control import CartesianController
from .placement import body_geom_ids, geom_world_bounds
from .robots import RobotControlProfile, get_actuator_id

def pose_matrix_from_body(
    model: mujoco.MjModel, data: mujoco.MjData, body_name: str
) -> np.ndarray:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        raise ValueError(f"Missing body: {body_name}")
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = data.xmat[body_id].reshape(3, 3)
    pose[:3, 3] = data.xpos[body_id]
    return pose


def pose_matrix_from_target(controller: CartesianController) -> np.ndarray:
    pose = np.eye(4, dtype=np.float64)
    pose[:3, :3] = controller.target_rot
    pose[:3, 3] = controller.target_pos
    return pose


def load_goal_definition(goal_definition: str | None) -> dict[str, Any] | None:
    if goal_definition is None:
        return None
    path = Path(goal_definition).expanduser()
    if path.exists():
        with path.open("r", encoding="utf-8") as f:
            spec = json.load(f)
    else:
        spec = json.loads(goal_definition)
    if isinstance(spec, str):
        spec = {"type": spec}
    if not isinstance(spec, dict):
        raise ValueError(
            "--goal-definition must be a JSON object, JSON string, or JSON file"
        )
    return spec


class GoalEvaluator:
    def __init__(
        self,
        model: mujoco.MjModel,
        spec: dict[str, Any] | None,
        runtime_state: dict[str, Any] | None = None,
    ):
        self.model = model
        self.spec = spec or {"type": "none"}
        self.name = str(self.spec.get("name") or self.spec.get("type") or "goal")
        self.stable_steps = int(self.spec.get("stable_steps", 1))
        self._body_cache: dict[str, int] = {}
        self.runtime_state = runtime_state if runtime_state is not None else {}

    def _body_id(self, name: str) -> int:
        if name not in self._body_cache:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise ValueError(f"Goal definition references missing body: {name}")
            self._body_cache[name] = body_id
        return self._body_cache[name]

    def _body_min_z(
        self,
        data: mujoco.MjData,
        body_name: str,
        geom_name_contains: str | None = None,
    ) -> float:
        return min(
            geom_world_bounds(self.model, data, geom_id)[0][2]
            for geom_id in body_geom_ids(
                self.model,
                body_name,
                geom_name_contains=geom_name_contains,
                require_geoms=True,
            )
        )

    def _body_max_z(
        self,
        data: mujoco.MjData,
        body_name: str,
        geom_name_contains: str | None = None,
    ) -> float:
        return max(
            geom_world_bounds(self.model, data, geom_id)[1][2]
            for geom_id in body_geom_ids(
                self.model,
                body_name,
                geom_name_contains=geom_name_contains,
                require_geoms=True,
            )
        )

    def evaluate(self, data: mujoco.MjData) -> bool:
        return self._eval_spec(self.spec, data)

    def _eval_spec(self, spec: dict[str, Any], data: mujoco.MjData) -> bool:
        goal_type = spec.get("type", "none")
        if goal_type in ("none", None):
            return False
        if goal_type == "box_in_tray":
            spec = {
                "type": "body_in_region",
                "body": "cardboard_box",
                "region_body": "wooden_tray",
                "xy_tolerance": 0.22,
                "z_min": 0.72,
                "z_max": 1.05,
                **spec,
            }
            return self._eval_spec({**spec, "type": "body_in_region"}, data)
        if goal_type == "body_in_region":
            body_id = self._body_id(spec["body"])
            region_body_id = self._body_id(spec["region_body"])
            body_pos = data.xpos[body_id]
            region_pos = data.xpos[region_body_id]
            xy_tolerance = float(spec.get("xy_tolerance", 0.05))
            z_min = spec.get("z_min")
            z_max = spec.get("z_max")
            xy_ok = np.linalg.norm(body_pos[:2] - region_pos[:2]) <= xy_tolerance
            z_ok = True
            if z_min is not None:
                z_ok = z_ok and body_pos[2] >= float(z_min)
            if z_max is not None:
                z_ok = z_ok and body_pos[2] <= float(z_max)
            return bool(xy_ok and z_ok)
        if goal_type == "body_near_body":
            body_pos = data.xpos[self._body_id(spec["body"])]
            target_pos = data.xpos[self._body_id(spec["target_body"])]
            tolerance = float(spec.get("pos_tolerance", spec.get("tolerance", 0.05)))
            return bool(np.linalg.norm(body_pos - target_pos) <= tolerance)
        if goal_type == "body_height_below":
            body_pos = data.xpos[self._body_id(spec["body"])]
            return bool(body_pos[2] <= float(spec["z"]))
        if goal_type == "body_height_above":
            body_pos = data.xpos[self._body_id(spec["body"])]
            return bool(body_pos[2] >= float(spec["z"]))
        if goal_type == "body_min_z_above_body_max_z":
            body_min_z = self._body_min_z(
                data, spec["body"], spec.get("body_geom_name_contains")
            )
            reference_max_z = self._body_max_z(
                data,
                spec["reference_body"],
                spec.get("reference_geom_name_contains"),
            )
            return bool(body_min_z >= reference_max_z + float(spec.get("margin", 0.0)))
        if goal_type == "red_box_above_tray_wall":
            nested_spec = {
                **spec,
                "type": "body_min_z_above_body_max_z",
                "body": "cardboard_box",
                "reference_body": "wooden_tray",
                "reference_geom_name_contains": "wall",
                "margin": float(spec.get("margin", 0.0)),
            }
            return self._eval_spec(
                nested_spec,
                data,
            )
        if goal_type == "fragile_wall_lift":
            if bool(self.runtime_state.get("fragile_wall_broken", False)) and bool(
                spec.get("require_wall_intact", True)
            ):
                return False
            nested_spec = {
                **spec,
                "type": "body_min_z_above_body_max_z",
                "body": spec.get("body", "cardboard_box"),
                "reference_body": spec.get("reference_body", "wooden_tray"),
                "reference_geom_name_contains": spec.get(
                    "reference_geom_name_contains", "wall"
                ),
                "margin": float(spec.get("margin", 0.0)),
            }
            return self._eval_spec(nested_spec, data)
        if goal_type == "and":
            return all(
                self._eval_spec(child, data) for child in spec.get("conditions", [])
            )
        if goal_type == "or":
            return any(
                self._eval_spec(child, data) for child in spec.get("conditions", [])
            )
        raise ValueError(f"Unsupported goal definition type: {goal_type}")


class RecordingMetadataProvider:
    def __init__(
        self,
        model: mujoco.MjModel,
        robot_profile: RobotControlProfile,
        goal_evaluator: GoalEvaluator,
        language_prompt: str,
        goal_definition: dict[str, Any] | None,
        task_name: str,
        runtime_state: dict[str, Any] | None = None,
    ) -> None:
        self.model = model
        self.robot_profile = robot_profile
        self.goal_evaluator = goal_evaluator
        self.language_prompt = language_prompt
        self.goal_definition = goal_definition or {"type": "none"}
        self.task_name = task_name
        self.runtime_state = runtime_state if runtime_state is not None else {}
        self.object_names = ("cardboard_box", "wooden_tray")
        self.arm_actuator_ids = np.asarray(
            [get_actuator_id(model, name) for name in robot_profile.arm_actuators],
            dtype=np.int32,
        )
        self.gripper_actuator_ids = np.asarray(
            [
                get_actuator_id(model, name)
                for name in robot_profile.gripper_actuators
                if get_actuator_id(model, name) >= 0
            ],
            dtype=np.int32,
        )
        self.last_goal_satisfied = False

    def static_metadata(self) -> dict:
        task_spec = {
            "task_name": self.task_name,
            "language_prompt": self.language_prompt,
            "goal_definition": self.goal_definition,
            "mimicgen": {
                "env_interface_name": "MG_CustomCardboardBox",
                "env_interface_type": "mujoco",
                "object_pose_names": list(self.object_names),
                "subtask_term_signal_names": [self.goal_evaluator.name],
            },
            "vla": {
                "instruction": self.language_prompt,
                "goal": self.goal_definition,
            },
        }
        return {
            "task_name": self.task_name,
            "language_prompt": self.language_prompt,
            "goal_definition_json": json.dumps(self.goal_definition, sort_keys=True),
            "task_metadata_json": json.dumps(task_spec, sort_keys=True),
            "mimicgen_env_interface_name": "MG_CustomCardboardBox",
            "mimicgen_env_interface_type": "mujoco",
            "datagen_object_pose_names": np.asarray(self.object_names),
            "datagen_subtask_signal_names": np.asarray([self.goal_evaluator.name]),
            "episode_success": False,
            "episode_failure": False,
            "episode_goal_satisfied_step": -1,
            "episode_failure_step": -1,
        }

    def __call__(
        self, data: mujoco.MjData, controller: CartesianController
    ) -> dict[str, np.ndarray]:
        eef_pose = pose_matrix_from_body(self.model, data, self.robot_profile.eef_body)
        target_pose = pose_matrix_from_target(controller)
        object_poses = np.stack(
            [
                pose_matrix_from_body(self.model, data, name)
                for name in self.object_names
            ]
        )
        goal_satisfied = self.goal_evaluator.evaluate(data)
        self.last_goal_satisfied = goal_satisfied

        if len(self.gripper_actuator_ids) > 0:
            gripper_action = data.ctrl[self.gripper_actuator_ids].copy()
        else:
            gripper_action = np.zeros(1, dtype=np.float64)
        policy_action = np.concatenate(
            [
                data.ctrl[self.arm_actuator_ids].copy(),
                gripper_action,
            ]
        )
        return {
            "goal_satisfied": np.asarray(goal_satisfied, dtype=np.bool_),
            "fragile_wall_broken": np.asarray(
                bool(self.runtime_state.get("fragile_wall_broken", False)),
                dtype=np.bool_,
            ),
            "datagen_eef_pose": eef_pose,
            "datagen_target_pose": target_pose,
            "datagen_object_poses": object_poses,
            "datagen_subtask_term_signals": np.asarray(
                [int(goal_satisfied)], dtype=np.int32
            ),
            "datagen_gripper_action": gripper_action,
            "policy_action": policy_action,
            "vla_instruction": np.asarray(self.language_prompt),
        }

