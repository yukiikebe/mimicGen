from __future__ import annotations

from .common import VIEW_STEP, ZOOM_STEP, mujoco, os, sys
from .cameras import apply_visual_only_viewer_options, print_view_camera_debug, set_world_camera_defaults
from .control import CartesianController, KeyboardState
from .inputs import toggle_gripper
from .recording import TeleopRecorder
from .robots import RobotControlProfile, set_initial_pose

def make_key_callback(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    viewer_holder: dict,
    camera_id: int,
    cartesian_controller: CartesianController,
    robot_profile: RobotControlProfile,
    keyboard_state: KeyboardState,
    recorder: TeleopRecorder,
):
    def toggle_gripper_local() -> None:
        toggle_gripper(model, data, robot_profile)

    def toggle_camera_mode() -> None:
        viewer = viewer_holder.get("viewer")
        if viewer is None:
            return

        if viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
            set_world_camera_defaults(viewer.cam, model, data)
            print("Camera: free")
        elif camera_id >= 0:
            viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
            viewer.cam.fixedcamid = camera_id
            print("Camera: main_camera")
        else:
            set_world_camera_defaults(viewer.cam, model, data)
            print("Camera: free reset")

    def print_current_view() -> None:
        viewer = viewer_holder.get("viewer")
        if viewer is None:
            return
        print_view_camera_debug(model, data, viewer.cam)

    def restore_scene_visibility() -> None:
        viewer = viewer_holder.get("viewer")
        if viewer is None:
            return

        apply_visual_only_viewer_options(viewer)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SELECT] = 0

    def adjust_free_camera(char: str) -> bool:
        viewer = viewer_holder.get("viewer")
        if viewer is None or viewer.cam.type != mujoco.mjtCamera.mjCAMERA_FREE:
            return False

        if char == "w":
            viewer.cam.lookat[1] += VIEW_STEP
        elif char == "s":
            viewer.cam.lookat[1] -= VIEW_STEP
        elif char == "a":
            viewer.cam.lookat[0] -= VIEW_STEP
        elif char == "d":
            viewer.cam.lookat[0] += VIEW_STEP
        elif char == "q":
            viewer.cam.lookat[2] += VIEW_STEP
        elif char == "e":
            viewer.cam.lookat[2] -= VIEW_STEP
        elif char == "z":
            viewer.cam.distance = max(0.2, viewer.cam.distance - ZOOM_STEP)
        elif char == "x":
            viewer.cam.distance += ZOOM_STEP
        else:
            return False
        return True

    def key_callback(key: int) -> None:
        if not keyboard_state.active and cartesian_controller.step_target(
            key, False, 0.05
        ):
            return

        try:
            char = chr(key).lower()
        except ValueError:
            return

        if char == " ":
            cartesian_controller.toggle_rotate_mode()
        elif adjust_free_camera(char):
            return
        elif char == "0":
            toggle_gripper_local()
            restore_scene_visibility()
        elif char == "c":
            recorder.toggle()
            restore_scene_visibility()
        elif char == "r":
            set_initial_pose(model, data, robot_profile)
            cartesian_controller.reset_target_to_current()
        elif char == "v":
            toggle_camera_mode()
        elif char == "p":
            print_current_view()
        elif char == "m":
            cartesian_controller.toggle_keyboard_xy_map()

    return key_callback


def configure_live_terminal_output() -> None:
    if sys.stdout.isatty() and sys.stderr.isatty():
        try:
            sys.stdout.reconfigure(line_buffering=True)
            sys.stderr.reconfigure(line_buffering=True)
        except AttributeError:
            pass
        return

    try:
        tty_fd = os.open("/dev/tty", os.O_WRONLY)
    except OSError:
        return

    os.dup2(tty_fd, 1)
    os.dup2(tty_fd, 2)
    os.close(tty_fd)
    encoding = sys.__stdout__.encoding or "utf-8"
    sys.stdout = open(
        1, "w", buffering=1, encoding=encoding, errors="replace", closefd=False
    )
    sys.stderr = open(
        2, "w", buffering=1, encoding=encoding, errors="replace", closefd=False
    )
