from __future__ import annotations

from .common import (
    DEFAULT_AXIS_ROTATION,
    DEFAULT_BTN_GRIPPER,
    DEFAULT_BTN_RECORD,
    DEFAULT_BTN_RESET,
    DEFAULT_BTN_ROTATION,
    DEFAULT_BTN_TOGGLE_CONTROL,
    DS4_BTN_GRIPPER,
    DS4_BTN_RECORD,
    DS4_BTN_RESET,
    DS4_BTN_ROTATION,
    DS4_BTN_TOGGLE_CONTROL,
    LOCAL_T265_SERVER_PY,
    T265_ROTATION_DEADBAND,
    T265_TRANSLATION_DEADBAND,
    TELEOP_REALSENSE_SRC,
    argparse,
    mujoco,
    np,
    os,
    sys,
    time,
)
from .control import CartesianController
from .math_utils import invert_transform, pose_json_to_matrix, rotation_angle
from .robots import RobotControlProfile, clamp_ctrl, get_actuator_id


class Gamepad:
    def __init__(
        self,
        joystick_index: int = 0,
        toggle_button: int | None = None,
        reset_button: int | None = None,
        gripper_button: int | None = None,
        record_button: int | None = None,
        rotation_button: int | None = None,
        rotation_axis: int = DEFAULT_AXIS_ROTATION,
        rotation_axis_threshold: float = 0.5,
    ) -> None:
        try:
            import pygame
        except Exception as exc:
            raise RuntimeError("pygame is required for gamepad input.") from exc

        self._pygame = pygame
        self._pending: set[str] = set()
        self._rotation_button_down = False
        self._rotation_axis_down = False
        self.rotation_axis = rotation_axis
        self.rotation_axis_threshold = rotation_axis_threshold

        pygame.init()
        pygame.joystick.init()
        joystick_count = pygame.joystick.get_count()
        if joystick_count <= joystick_index:
            raise RuntimeError(
                f"No gamepad at index {joystick_index}. "
                f"Detected {joystick_count} joystick(s)."
            )
        self.joy = pygame.joystick.Joystick(joystick_index)
        self.joy.init()
        name = self.joy.get_name()
        defaults = self._default_buttons_for_name(name)
        self.toggle_button = (
            defaults["toggle"] if toggle_button is None else toggle_button
        )
        self.reset_button = defaults["reset"] if reset_button is None else reset_button
        self.gripper_button = (
            defaults["gripper"] if gripper_button is None else gripper_button
        )
        self.record_button = (
            defaults["record"] if record_button is None else record_button
        )
        self.rotation_button = (
            defaults["rotation"] if rotation_button is None else rotation_button
        )
        print(f"Using gamepad: {name}")
        print(
            "Gamepad button map: "
            f"toggle={self.toggle_button}, reset={self.reset_button}, "
            f"gripper={self.gripper_button}, record={self.record_button}, "
            f"rotation={self.rotation_button}"
        )

    @staticmethod
    def _default_buttons_for_name(name: str) -> dict[str, int]:
        if "sony" in name.lower() or "wireless controller" in name.lower():
            return {
                "toggle": DS4_BTN_TOGGLE_CONTROL,
                "reset": DS4_BTN_RESET,
                "gripper": DS4_BTN_GRIPPER,
                "record": DS4_BTN_RECORD,
                "rotation": DS4_BTN_ROTATION,
            }
        return {
            "toggle": DEFAULT_BTN_TOGGLE_CONTROL,
            "reset": DEFAULT_BTN_RESET,
            "gripper": DEFAULT_BTN_GRIPPER,
            "record": DEFAULT_BTN_RECORD,
            "rotation": DEFAULT_BTN_ROTATION,
        }

    def poll(self) -> None:
        pygame = self._pygame
        for event in pygame.event.get():
            if event.type == pygame.JOYAXISMOTION:
                if event.axis == self.rotation_axis:
                    self._rotation_axis_down = (
                        float(event.value) > self.rotation_axis_threshold
                    )
                continue
            if event.type == pygame.JOYHATMOTION:
                hat_x, hat_y = event.value
                if hat_x > 0:
                    self._pending.add("record")
                elif hat_x < 0:
                    self._pending.add("reset_randomize")
                if hat_y > 0:
                    self._pending.add("success")
                elif hat_y < 0:
                    self._pending.add("failure")
                continue
            if event.type == pygame.JOYBUTTONUP:
                if event.button in {
                    self.rotation_button,
                    DEFAULT_BTN_ROTATION,
                    DS4_BTN_ROTATION,
                }:
                    self._rotation_button_down = False
                continue
            if event.type != pygame.JOYBUTTONDOWN:
                continue
            print(f"[gamepad] button {event.button} pressed", flush=True)
            if event.button in {
                self.rotation_button,
                DEFAULT_BTN_ROTATION,
                DS4_BTN_ROTATION,
            }:
                self._rotation_button_down = True
            elif event.button in {
                self.toggle_button,
                DEFAULT_BTN_TOGGLE_CONTROL,
                DS4_BTN_TOGGLE_CONTROL,
            }:
                self._pending.add("toggle_control")
            elif event.button in {self.reset_button, DEFAULT_BTN_RESET, DS4_BTN_RESET}:
                self._pending.add("reset")
            elif event.button in {
                self.gripper_button,
                DEFAULT_BTN_GRIPPER,
                DS4_BTN_GRIPPER,
            }:
                self._pending.add("gripper")
            elif event.button in {
                self.record_button,
                DEFAULT_BTN_RECORD,
                DS4_BTN_RECORD,
            }:
                print(
                    "Recording starts from D-pad right in this teleop.",
                    flush=True,
                )

    def rotation_enabled(self) -> bool:
        return self._rotation_button_down or self._rotation_axis_down

    def consume(self, name: str) -> bool:
        if name not in self._pending:
            return False
        self._pending.discard(name)
        return True

    def get_axis(self, axis: int, deadzone: float) -> float:
        if axis < 0 or axis >= self.joy.get_numaxes():
            return 0.0
        value = float(self.joy.get_axis(axis))
        if abs(value) < deadzone:
            return 0.0
        return value

    def close(self) -> None:
        self._pygame.joystick.quit()
        self._pygame.quit()


class T265Teleop:
    def __init__(
        self,
        model: mujoco.MjModel,
        data: mujoco.MjData,
        cartesian_controller: CartesianController,
        serial: str,
        rate_hz: float,
        start_container: bool,
        image: str | None,
        translation_map: str = "xzy",
        rotation_map: str = "quat_axes",
        rotation_axis_map: str = "yzx",
        translation_frame: str = "world",
        debug_pose: bool = False,
    ) -> None:
        self.model = model
        self.data = data
        self.cartesian_controller = cartesian_controller
        self.serial = serial
        self.rate_hz = rate_hz
        self.start_container = start_container
        self.image = image
        self.translation_map = translation_map
        self.rotation_map = rotation_map
        self.rotation_axis_map = rotation_axis_map
        self.translation_frame = translation_frame
        self.debug_pose = debug_pose
        self.t265 = None
        self.in_control = False
        self.ee_anchor: np.ndarray | None = None
        self.t265_anchor: np.ndarray | None = None
        self.t265_motion_anchor: np.ndarray | None = None
        self.ee_rotation_anchor: np.ndarray | None = None
        self.t265_rotation_anchor: np.ndarray | None = None
        self.rotation_control_enabled = False
        self._last_debug_time = 0.0

    def hold_current_pose(self) -> None:
        self.cartesian_controller.hold_current_pose(reset_arm_velocity=True)

    def start(self) -> None:
        if self.image is not None:
            os.environ.setdefault("T265_IMAGE", self.image)
        if LOCAL_T265_SERVER_PY.exists():
            os.environ.setdefault("T265_SERVER_PY", str(LOCAL_T265_SERVER_PY))
        if TELEOP_REALSENSE_SRC.exists():
            sys.path.insert(0, str(TELEOP_REALSENSE_SRC))
        try:
            from TeleopRealSense.devices.gamepad.realsense_container import (
                T265ContainerManager,
            )
        except Exception as exc:
            raise RuntimeError(
                "Could not import T265ContainerManager. Expected TeleopRealSense at "
                f"{TELEOP_REALSENSE_SRC}"
            ) from exc

        self.t265 = T265ContainerManager()
        if self.start_container:
            self.t265.start()
        print("Starting T265 stream...")
        self.t265.start_stream(rate_hz=self.rate_hz, serial=self.serial)
        print("Stream started.", self.t265.status())

    def stop(self) -> None:
        if self.t265 is not None and self.start_container:
            self.t265.stop()

    def reset_stream(self) -> None:
        if self.t265 is None:
            return
        try:
            self.t265.reset_stream()
        except Exception as exc:
            print(f"T265 stream reset failed: {exc}")

    def anchor(self) -> None:
        if self.t265 is None:
            raise RuntimeError("T265 stream has not been started.")

        mujoco.mj_forward(self.model, self.data)
        ee_body_id = self.cartesian_controller.ee_body_id
        self.ee_anchor = np.eye(4, dtype=np.float64)
        self.ee_anchor[:3, :3] = self.data.xmat[ee_body_id].reshape(3, 3).copy()
        self.ee_anchor[:3, 3] = self.data.xpos[ee_body_id].copy()

        pose_json = self.t265.wait_for_pose(max_wait_s=5.0, poll_s=0.01)
        while pose_json is None:
            pose_json = self.t265.wait_for_pose(max_wait_s=5.0, poll_s=0.01)
        self.t265_anchor = pose_json_to_matrix(
            pose_json,
            translation_map=self.translation_map,
            rotation_map=self.rotation_map,
            rotation_axis_map=self.rotation_axis_map,
        )
        self.t265_motion_anchor = pose_json_to_matrix(
            pose_json,
            translation_map=self.translation_map,
            rotation_map="legacy",
        )
        self.ee_rotation_anchor = self.ee_anchor[:3, :3].copy()
        self.t265_rotation_anchor = self.t265_anchor.copy()
        self.rotation_control_enabled = False

        self.cartesian_controller.set_target_pose(
            pos=self.ee_anchor[:3, 3],
            rot=self.ee_anchor[:3, :3],
        )

    def toggle_control(self) -> None:
        self.in_control = not self.in_control
        print(f"Control {'enabled' if self.in_control else 'disabled'}.", flush=True)
        if self.in_control:
            self.anchor()
        else:
            self.hold_current_pose()

    def toggle_translation_xy_map(self) -> None:
        map_pairs = {
            "xzy": "zxy",
            "zxy": "xzy",
            "zyx": "yzx",
            "yzx": "zyx",
            "swap_xz": "yzx",
        }
        next_map = map_pairs.get(self.translation_map)
        if next_map is None:
            print(
                f"Cannot toggle X/Y for T265 translation map {self.translation_map}.",
                flush=True,
            )
            return

        self.translation_map = next_map
        if not self.in_control or self.t265 is None or self.ee_anchor is None:
            print(f"T265 translation map set to {self.translation_map}.", flush=True)
            return

        pose_json = self.t265.fetch_pose()
        while pose_json is None:
            time.sleep(0.005)
            pose_json = self.t265.fetch_pose()

        self.ee_anchor[:3, 3] = self.cartesian_controller.target_pos.copy()
        self.ee_rotation_anchor = self.cartesian_controller.target_rot.copy()
        self.t265_anchor = pose_json_to_matrix(
            pose_json,
            translation_map=self.translation_map,
            rotation_map=self.rotation_map,
            rotation_axis_map=self.rotation_axis_map,
        )
        self.t265_motion_anchor = pose_json_to_matrix(
            pose_json,
            translation_map=self.translation_map,
            rotation_map="legacy",
        )
        self.t265_rotation_anchor = self.t265_anchor.copy()
        self.rotation_control_enabled = False
        print(
            f"T265 translation map set to {self.translation_map}; re-anchored.",
            flush=True,
        )

    def update_target(self, apply_rotation: bool = False) -> bool:
        if not self.in_control:
            self.hold_current_pose()
            return False
        if (
            self.t265 is None
            or self.ee_anchor is None
            or self.t265_anchor is None
            or self.t265_motion_anchor is None
            or self.ee_rotation_anchor is None
            or self.t265_rotation_anchor is None
        ):
            self.hold_current_pose()
            return False

        pose_json = self.t265.fetch_pose()
        if pose_json is None:
            self.hold_current_pose()
            return False

        t265_current = pose_json_to_matrix(
            pose_json,
            translation_map=self.translation_map,
            rotation_map=self.rotation_map,
            rotation_axis_map=self.rotation_axis_map,
        )
        t265_motion_current = pose_json_to_matrix(
            pose_json,
            translation_map=self.translation_map,
            rotation_map="legacy",
        )
        t265_motion_delta = t265_motion_current[:3, 3] - self.t265_motion_anchor[:3, 3]
        if apply_rotation and not self.rotation_control_enabled:
            self.hold_current_pose()
            self.ee_rotation_anchor = self.cartesian_controller.target_rot.copy()
            self.t265_rotation_anchor = t265_current.copy()
            self.rotation_control_enabled = True
            return False
        elif not apply_rotation and self.rotation_control_enabled:
            self.hold_current_pose()
            self.ee_anchor[:3, 3] = self.cartesian_controller.target_pos.copy()
            self.t265_motion_anchor = t265_motion_current.copy()
            t265_motion_delta = np.zeros(3, dtype=np.float64)
            self.rotation_control_enabled = False
            return False

        if not apply_rotation:
            if np.linalg.norm(t265_motion_delta) <= T265_TRANSLATION_DEADBAND:
                self.t265_motion_anchor = t265_motion_current.copy()
                self.ee_anchor[:3, 3] = self.cartesian_controller.target_pos.copy()
                self.hold_current_pose()
                return False

            target_pos: np.ndarray
            if self.translation_frame == "ee":
                target_pos = (
                    self.cartesian_controller.target_pos
                    + self.ee_anchor[:3, :3] @ t265_motion_delta
                )
            elif self.translation_frame == "world":
                target_pos = self.cartesian_controller.target_pos + t265_motion_delta
            else:
                raise ValueError(
                    f"Unsupported T265 translation frame: {self.translation_frame}"
                )
            self.cartesian_controller.set_target_pose(pos=target_pos)
            self.t265_motion_anchor = t265_motion_current.copy()
        if apply_rotation:
            t265_rotation_delta = (
                invert_transform(self.t265_rotation_anchor) @ t265_current
            )
            if rotation_angle(t265_rotation_delta[:3, :3]) <= T265_ROTATION_DEADBAND:
                self.t265_rotation_anchor = t265_current.copy()
                self.hold_current_pose()
                return False

            self.cartesian_controller.set_target_pose(
                rot=self.cartesian_controller.target_rot @ t265_rotation_delta[:3, :3]
            )
            self.t265_rotation_anchor = t265_current.copy()
        if self.debug_pose and time.time() - self._last_debug_time > 0.5:
            self._last_debug_time = time.time()
            print(
                "T265 delta xyz: "
                f"{t265_motion_delta[0]:+.4f}, "
                f"{t265_motion_delta[1]:+.4f}, "
                f"{t265_motion_delta[2]:+.4f}",
                flush=True,
            )
        return True


def print_teleop_help(input_mode: str, random_scene: bool = False) -> None:
    print("Keyboard controls:")
    print("  hold arrow keys : move gripper target on X/Y")
    print("  hold i/k : move gripper target up/down on Z")
    print("  hold arrow keys + i/k : move gripper target diagonally in X/Y and Z")
    print("  hold shift + up/down : move gripper target on Z")
    print("  space : toggle translate/rotate target mode")
    print("  rotate mode arrows : rotate target around X/Y")
    print("  rotate mode shift + up/down : rotate target around Z")
    print("  0   : toggle gripper open/close")
    if random_scene:
        print("  r   : generate and reload a random RoboCasa island scene")
    else:
        print("  r   : reset to initial pose")
    print("  v   : toggle fixed/free camera")
    print("  p   : print current camera view to terminal")
    print("  w/a/s/d : move free-camera lookat on table plane")
    print("  q/e : move free-camera lookat up/down")
    print("  z/x : zoom free camera in/out")
    print("  n   : toggle side OpenCV view left/right")
    if input_mode == "keyboard":
        print("  m   : swap keyboard translation/rotation X/Y map")
        print("  c   : start/stop recording")
        print("  keyboard input mode: hold keys to command the EE target")
        return

    print("  c   : disabled; use D-pad right to start recording")
    if input_mode == "t265":
        print("  m   : swap T265 translation X/Y map and re-anchor")
    print("Gamepad controls:")
    if input_mode == "t265":
        print("  RB or u : toggle T265 control on/off")
        if random_scene:
            print("  LB : generate and reload a random RoboCasa island scene")
        else:
            print("  LB : reset scene and T265 stream")
        print("  R2 : enable T265 rotation while held")
        print("  T265 input mode: move the T265 to command relative EE pose")
    else:
        print("  RB or u : toggle gamepad control on/off")
        if random_scene:
            print("  LB : generate and reload a random RoboCasa island scene")
        else:
            print("  LB : reset scene")
        print("  gamepad fallback mode: left stick X/Y, right stick Z/yaw")
    print("  B  : toggle gripper open/close")
    print("  D-pad right : start recording")
    if random_scene:
        print("  D-pad left : generate and reload a random RoboCasa island scene")
    else:
        print("  D-pad left : reset scene and randomize red box XY inside tray walls")
    print("  D-pad up/down : mark recording success/failure and save")


def toggle_gripper(
    model: mujoco.MjModel, data: mujoco.MjData, robot_profile: RobotControlProfile
) -> None:
    actuator_ids = [
        get_actuator_id(model, name) for name in robot_profile.gripper_actuators
    ]
    if not actuator_ids or any(actuator_id < 0 for actuator_id in actuator_ids):
        return
    current = np.asarray([data.ctrl[actuator_id] for actuator_id in actuator_ids])
    open_ctrl = np.asarray(robot_profile.gripper_open_ctrl, dtype=np.float64)
    closed_ctrl = np.asarray(robot_profile.gripper_closed_ctrl, dtype=np.float64)
    if len(open_ctrl) != len(actuator_ids) or len(closed_ctrl) != len(actuator_ids):
        return
    target = (
        closed_ctrl
        if np.linalg.norm(current - open_ctrl) <= np.linalg.norm(current - closed_ctrl)
        else open_ctrl
    )
    for actuator_id, value in zip(actuator_ids, target):
        data.ctrl[actuator_id] = clamp_ctrl(model, actuator_id, float(value))
    mujoco.mj_forward(model, data)


def apply_gamepad_target_delta(
    gamepad: Gamepad,
    cartesian_controller: CartesianController,
    args: argparse.Namespace,
    dt: float,
) -> bool:
    move_x = gamepad.get_axis(args.axis_x, args.deadzone)
    move_y = -gamepad.get_axis(args.axis_y, args.deadzone)
    move_z = -gamepad.get_axis(args.axis_z, args.deadzone)
    yaw = gamepad.get_axis(args.axis_yaw, args.deadzone)

    translation = np.array([move_x, move_y, move_z], dtype=np.float64)
    command_active = bool(np.linalg.norm(translation) > 0.0 or yaw != 0.0)
    if not command_active:
        cartesian_controller.hold_current_pose(reset_arm_velocity=True)
        return False

    linear_velocity = translation * args.translation_speed
    angular_velocity = np.array([0.0, 0.0, yaw * args.rotation_speed])
    return cartesian_controller.apply_target_delta(
        linear_velocity if np.linalg.norm(translation) > 0.0 else None,
        angular_velocity if yaw != 0.0 else None,
        dt,
    )
