from __future__ import annotations

from .common import (
    DEFAULT_EE_COMMAND_FRAME,
    KEY_DOWN,
    KEY_LEFT,
    KEY_LEFT_SHIFT,
    KEY_RIGHT,
    KEY_RIGHT_SHIFT,
    KEY_UP,
    KEY_Z_DOWN,
    KEY_Z_UP,
    MAX_EE_TARGET_POSITION_ERROR,
    ROTATION_SPEED,
    TRANSLATION_SPEED,
    mujoco,
    np,
    threading,
)
from .math_utils import rotation_error, rotation_matrix
from .robots import RobotControlProfile, get_actuator_id


class CartesianController:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        robot_profile: RobotControlProfile,
        translation_speed: float = TRANSLATION_SPEED,
        rotation_speed: float = ROTATION_SPEED,
        max_target_position_error: float = MAX_EE_TARGET_POSITION_ERROR,
        command_frame: str = DEFAULT_EE_COMMAND_FRAME,
    ):
        self.model = model
        self.data = data
        self.robot_profile = robot_profile
        self.translation_speed = float(translation_speed)
        self.rotation_speed = float(rotation_speed)
        if command_frame not in {"base", "world"}:
            raise ValueError(f"Unsupported EE command frame: {command_frame}")
        self.command_frame = command_frame
        self.ik_data = mujoco.MjData(model)
        self.ee_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, robot_profile.eef_body
        )
        if self.ee_body_id < 0:
            raise ValueError(f"Missing body: {robot_profile.eef_body}")

        self.arm_actuator_ids = np.asarray(
            [get_actuator_id(model, name) for name in robot_profile.arm_actuators],
            dtype=np.int32,
        )
        self.arm_joint_ids = np.asarray(
            [
                mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
                for name in robot_profile.arm_joints
            ],
            dtype=np.int32,
        )
        if np.any(self.arm_actuator_ids < 0) or np.any(self.arm_joint_ids < 0):
            raise ValueError(f"Missing {robot_profile.name} joint or actuator")

        self.qpos_adrs = model.jnt_qposadr[self.arm_joint_ids]
        self.dof_adrs = model.jnt_dofadr[self.arm_joint_ids]
        self.q_min = model.jnt_range[self.arm_joint_ids, 0]
        self.q_max = model.jnt_range[self.arm_joint_ids, 1]
        self.command_frame_site_id = self._resolve_command_frame_site_id()
        self.command_frame_body_id = self._resolve_command_frame_body_id()
        self.rotate_mode = False
        self.keyboard_xy_swapped = True
        self.max_target_position_error = float(max_target_position_error)

        mujoco.mj_forward(model, data)
        self.target_pos = data.xpos[self.ee_body_id].copy()
        self.target_rot = data.xmat[self.ee_body_id].reshape(3, 3).copy()
        self.joint_target_qpos = data.qpos[self.qpos_adrs].copy()
        self._target_dirty = False
        self._position_target_dirty = False
        self._rotation_target_dirty = False
        self._rotation_anchor_pos: np.ndarray | None = None

    def _resolve_command_frame_site_id(self) -> int:
        for site_name in (
            "robot0_right_center",
            "right_center",
            "mobilebase0_center",
            "center",
        ):
            site_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_SITE, site_name)
            if site_id >= 0:
                return site_id
        return -1

    def _resolve_command_frame_body_id(self) -> int:
        for body_name in ("robot0_base", "mobilebase0_base", "base"):
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, body_name)
            if body_id >= 0:
                return body_id
        return -1

    def command_frame_rotation(self) -> np.ndarray:
        if self.command_frame == "world":
            return np.eye(3, dtype=np.float64)
        mujoco.mj_forward(self.model, self.data)
        if self.command_frame_site_id >= 0:
            return self.data.site_xmat[self.command_frame_site_id].reshape(3, 3).copy()
        if self.command_frame_body_id >= 0:
            return self.data.xmat[self.command_frame_body_id].reshape(3, 3).copy()
        return np.eye(3, dtype=np.float64)

    def set_target_pose(
        self, pos: np.ndarray | None = None, rot: np.ndarray | None = None
    ) -> None:
        changed = False
        if pos is not None and not np.allclose(pos, self.target_pos, atol=1e-10):
            self.target_pos = pos.copy()
            self.clamp_target_position_error()
            changed = True
            self._position_target_dirty = True
        if rot is not None and not np.allclose(rot, self.target_rot, atol=1e-10):
            self.target_rot = rot.copy()
            changed = True
            self._rotation_target_dirty = True
        if changed:
            self._target_dirty = True

    def apply_target_delta(
        self,
        linear_velocity: np.ndarray | None,
        angular_velocity: np.ndarray | None,
        dt: float,
    ) -> bool:
        mujoco.mj_forward(self.model, self.data)
        frame_rot = self.command_frame_rotation()
        current_pos = self.data.xpos[self.ee_body_id].copy()
        current_rot = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()

        pos = current_pos
        rot = current_rot
        linear_changed = False
        angular_changed = False

        if linear_velocity is not None and np.linalg.norm(linear_velocity) > 0.0:
            self._rotation_anchor_pos = None
            pos = current_pos + frame_rot @ (linear_velocity * dt)
            if abs(float(linear_velocity[2])) < 1e-12:
                pos[2] = self.target_pos[2]
            linear_changed = True

        if angular_velocity is not None and np.linalg.norm(angular_velocity) > 0.0:
            angular_delta = frame_rot @ (angular_velocity * dt)
            angle = float(np.linalg.norm(angular_delta))
            if angle > 0.0:
                axis = angular_delta / angle
                rot = rotation_matrix(axis, angle) @ current_rot
                angular_changed = True

        if linear_changed or angular_changed:
            if linear_changed and not angular_changed:
                self.target_rot = current_rot.copy()
                self.set_target_pose(pos=pos)
            elif angular_changed and not linear_changed:
                if self._rotation_anchor_pos is None:
                    self._rotation_anchor_pos = current_pos.copy()
                self.target_pos = self._rotation_anchor_pos.copy()
                self.target_rot = rot.copy()
                self._target_dirty = True
                self._position_target_dirty = False
                self._rotation_target_dirty = True
            else:
                self.set_target_pose(pos=pos, rot=rot)
        return linear_changed or angular_changed

    def clamp_target_position_error(self) -> None:
        mujoco.mj_forward(self.model, self.data)
        current_pos = self.data.xpos[self.ee_body_id]
        error = self.target_pos - current_pos
        error_norm = float(np.linalg.norm(error))
        if error_norm <= self.max_target_position_error:
            return
        self.target_pos = current_pos + error * (
            self.max_target_position_error / error_norm
        )

    def reset_target_to_current(self) -> None:
        self.hold_current_pose(reset_arm_velocity=True)

    def hold_current_pose(self, reset_arm_velocity: bool = True) -> None:
        mujoco.mj_forward(self.model, self.data)
        self.target_pos = self.data.xpos[self.ee_body_id].copy()
        self.target_rot = self.data.xmat[self.ee_body_id].reshape(3, 3).copy()
        self.joint_target_qpos = self.data.qpos[self.qpos_adrs].copy()
        self.data.ctrl[self.arm_actuator_ids] = self.joint_target_qpos
        if reset_arm_velocity:
            self.data.qvel[self.dof_adrs] = 0.0
            self.data.qacc[self.dof_adrs] = 0.0
        self._target_dirty = False
        self._position_target_dirty = False
        self._rotation_target_dirty = False
        self._rotation_anchor_pos = None

    def toggle_rotate_mode(self) -> None:
        self.rotate_mode = not self.rotate_mode
        mode = "rotation" if self.rotate_mode else "translation"
        self._rotation_anchor_pos = None
        print(f"Cartesian control mode: {mode}")

    def toggle_keyboard_xy_map(self) -> None:
        self.keyboard_xy_swapped = not self.keyboard_xy_swapped
        mode = "swapped X/Y" if self.keyboard_xy_swapped else "normal X/Y"
        print(f"Keyboard translation/rotation map: {mode}", flush=True)
        self.hold_current_pose(reset_arm_velocity=True)

    def keyboard_angular_velocity(
        self, key: int, shift_pressed: bool
    ) -> np.ndarray | None:
        rotation_speed = self.rotation_speed
        if shift_pressed:
            angular_velocity_keys = {
                KEY_UP: np.array([0.0, 0.0, rotation_speed]),
                KEY_DOWN: np.array([0.0, 0.0, -rotation_speed]),
            }
        elif self.keyboard_xy_swapped:
            angular_velocity_keys = {
                KEY_UP: np.array([0.0, -rotation_speed, 0.0]),
                KEY_DOWN: np.array([0.0, rotation_speed, 0.0]),
                KEY_RIGHT: np.array([-rotation_speed, 0.0, 0.0]),
                KEY_LEFT: np.array([rotation_speed, 0.0, 0.0]),
            }
        else:
            angular_velocity_keys = {
                KEY_UP: np.array([-rotation_speed, 0.0, 0.0]),
                KEY_DOWN: np.array([rotation_speed, 0.0, 0.0]),
                KEY_RIGHT: np.array([0.0, -rotation_speed, 0.0]),
                KEY_LEFT: np.array([0.0, rotation_speed, 0.0]),
            }
        return angular_velocity_keys.get(key)

    def keyboard_linear_velocity(
        self, key: int, shift_pressed: bool
    ) -> np.ndarray | None:
        translation_speed = self.translation_speed
        z_velocity_keys = {
            KEY_Z_UP: np.array([0.0, 0.0, translation_speed]),
            KEY_Z_DOWN: np.array([0.0, 0.0, -translation_speed]),
        }
        if key in z_velocity_keys:
            return z_velocity_keys[key]

        if shift_pressed:
            linear_velocity_keys = {
                KEY_UP: np.array([0.0, 0.0, translation_speed]),
                KEY_DOWN: np.array([0.0, 0.0, -translation_speed]),
            }
        elif self.keyboard_xy_swapped:
            linear_velocity_keys = {
                KEY_UP: np.array([translation_speed, 0.0, 0.0]),
                KEY_DOWN: np.array([-translation_speed, 0.0, 0.0]),
                KEY_RIGHT: np.array([0.0, -translation_speed, 0.0]),
                KEY_LEFT: np.array([0.0, translation_speed, 0.0]),
            }
        else:
            linear_velocity_keys = {
                KEY_UP: np.array([0.0, -translation_speed, 0.0]),
                KEY_DOWN: np.array([0.0, translation_speed, 0.0]),
                KEY_RIGHT: np.array([translation_speed, 0.0, 0.0]),
                KEY_LEFT: np.array([-translation_speed, 0.0, 0.0]),
            }
        return linear_velocity_keys.get(key)

    def step_target(self, key: int, shift_pressed: bool, dt: float) -> bool:
        if self.rotate_mode:
            angular_velocity = self.keyboard_angular_velocity(key, shift_pressed)
            if angular_velocity is None:
                return False
            return self.apply_target_delta(None, angular_velocity, dt)

        linear_velocity = self.keyboard_linear_velocity(key, shift_pressed)
        if linear_velocity is None:
            return False
        return self.apply_target_delta(linear_velocity, None, dt)

    def apply_pressed_keys(
        self, pressed_keys: set[int], shift_pressed: bool, dt: float
    ) -> bool:
        linear_velocity = np.zeros(3, dtype=np.float64)
        angular_velocity = np.zeros(3, dtype=np.float64)
        target_changed = False
        for key in (KEY_UP, KEY_DOWN, KEY_LEFT, KEY_RIGHT, KEY_Z_UP, KEY_Z_DOWN):
            if key not in pressed_keys:
                continue
            if self.rotate_mode:
                key_velocity = self.keyboard_angular_velocity(key, shift_pressed)
                if key_velocity is not None:
                    angular_velocity += key_velocity
                    target_changed = True
            else:
                key_velocity = self.keyboard_linear_velocity(key, shift_pressed)
                if key_velocity is not None:
                    linear_velocity += key_velocity
                    target_changed = True

        if target_changed:
            if self.rotate_mode:
                self.apply_target_delta(None, angular_velocity, dt)
            else:
                self.apply_target_delta(linear_velocity, None, dt)
        else:
            self.hold_current_pose(reset_arm_velocity=True)
        return target_changed

    def update_control(self) -> None:
        if not self._target_dirty:
            self.data.ctrl[self.arm_actuator_ids] = self.joint_target_qpos
            return

        self.ik_data.qpos[:] = self.data.qpos
        self.ik_data.qvel[:] = self.data.qvel

        rotation_only = self._rotation_target_dirty and not self._position_target_dirty
        ik_iterations = 8 if rotation_only else 4

        for _ in range(ik_iterations):
            mujoco.mj_forward(self.model, self.ik_data)
            pos_error = self.target_pos - self.ik_data.xpos[self.ee_body_id]
            current_rot = self.ik_data.xmat[self.ee_body_id].reshape(3, 3)
            rot_error = rotation_error(self.target_rot, current_rot)
            rot_weight = 0.5 if self._rotation_target_dirty else 0.1

            jac_pos = np.zeros((3, self.model.nv))
            jac_rot = np.zeros((3, self.model.nv))
            mujoco.mj_jacBody(
                self.model, self.ik_data, jac_pos, jac_rot, self.ee_body_id
            )
            j_pos = jac_pos[:, self.dof_adrs]
            j_rot = jac_rot[:, self.dof_adrs]
            damping = 1e-2
            if rotation_only:
                pos_lhs = j_pos @ j_pos.T + damping * np.eye(3)
                j_pos_pinv = j_pos.T @ np.linalg.solve(pos_lhs, np.eye(3))
                dq_pos = j_pos_pinv @ pos_error
                nullspace = np.eye(len(self.dof_adrs)) - j_pos_pinv @ j_pos
                j_rot_null = rot_weight * j_rot @ nullspace
                rot_residual = rot_weight * (rot_error - j_rot @ dq_pos)
                rot_lhs = j_rot_null @ j_rot_null.T + damping * np.eye(3)
                dq = dq_pos + nullspace @ (
                    j_rot_null.T @ np.linalg.solve(rot_lhs, rot_residual)
                )
                dq = np.clip(dq, -0.025, 0.025)
            else:
                error = np.concatenate([pos_error, rot_weight * rot_error])
                jac = np.vstack([j_pos, rot_weight * j_rot])
                lhs = jac @ jac.T + damping * np.eye(6)
                dq = jac.T @ np.linalg.solve(lhs, error)
                dq = np.clip(dq, -0.04, 0.04)
            self.ik_data.qpos[self.qpos_adrs] = np.clip(
                self.ik_data.qpos[self.qpos_adrs] + dq, self.q_min, self.q_max
            )

        self.joint_target_qpos = self.ik_data.qpos[self.qpos_adrs].copy()
        self.data.ctrl[self.arm_actuator_ids] = self.joint_target_qpos
        self._target_dirty = False
        self._position_target_dirty = False
        self._rotation_target_dirty = False


class KeyboardState:
    def __init__(self) -> None:
        self._lock = threading.Lock()
        self._pressed_keys: set[int] = set()
        self._shift_keys: set[int] = set()
        self._listener = None
        self.active = False

    def start(self) -> None:
        try:
            from pynput import keyboard
        except Exception as exc:
            print(f"Hold-key control disabled: pynput is unavailable ({exc})")
            return

        key_map = {
            keyboard.Key.up: KEY_UP,
            keyboard.Key.down: KEY_DOWN,
            keyboard.Key.left: KEY_LEFT,
            keyboard.Key.right: KEY_RIGHT,
            keyboard.Key.shift: KEY_LEFT_SHIFT,
            keyboard.Key.shift_l: KEY_LEFT_SHIFT,
            keyboard.Key.shift_r: KEY_RIGHT_SHIFT,
        }

        def on_press(key) -> None:
            code = key_map.get(key)
            if code is None and getattr(key, "char", None):
                char = key.char.lower()
                if char == "i":
                    code = KEY_Z_UP
                elif char == "k":
                    code = KEY_Z_DOWN
            if code is None:
                return
            with self._lock:
                if code in (KEY_LEFT_SHIFT, KEY_RIGHT_SHIFT):
                    self._shift_keys.add(code)
                else:
                    self._pressed_keys.add(code)

        def on_release(key) -> None:
            code = key_map.get(key)
            if code is None and getattr(key, "char", None):
                char = key.char.lower()
                if char == "i":
                    code = KEY_Z_UP
                elif char == "k":
                    code = KEY_Z_DOWN
            if code is None:
                return
            with self._lock:
                if code in (KEY_LEFT_SHIFT, KEY_RIGHT_SHIFT):
                    self._shift_keys.discard(code)
                else:
                    self._pressed_keys.discard(code)

        try:
            self._listener = keyboard.Listener(on_press=on_press, on_release=on_release)
            self._listener.start()
            self.active = True
            print("Hold-key control enabled.")
        except Exception as exc:
            print(f"Hold-key control disabled: could not start listener ({exc})")

    def snapshot(self) -> tuple[set[int], bool]:
        with self._lock:
            return set(self._pressed_keys), bool(self._shift_keys)

    def stop(self) -> None:
        if self._listener is not None:
            self._listener.stop()
