from __future__ import annotations

from .common import (
    CARDBOARD_BOX_FREEJOINT,
    DEBUG_CONTROL_INTERVAL,
    DEBUG_VIEW_INTERVAL,
    DEFAULT_AXIS_ROTATION,
    DEFAULT_EE_COMMAND_FRAME,
    DEFAULT_RANDOM_SCENE_OUTPUT_DIR,
    DEFAULT_ROBOCASA_ROOT,
    DEFAULT_ROBOSUITE_ROOT,
    DEFAULT_VIDEO_VIEWS,
    FRAGILE_WALL_BREAK_FORCE_THRESHOLD,
    MAX_EE_TARGET_POSITION_ERROR,
    MODEL_PATH,
    Path,
    ROTATION_SPEED,
    SIDE_VIEW_DISTANCE,
    SIDE_VIEW_ELEVATION,
    SIDE_VIEW_FPS,
    SIDE_VIEW_HEIGHT,
    SIDE_VIEW_WIDTH,
    TRANSLATION_SPEED,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_RENDER_EVERY,
    VIDEO_WIDTH,
    argparse,
    mujoco,
    sys,
)
from .control import CartesianController
from .placement import align_freejoint_body_bottom_to_support, reset_freejoint_to_model_qpos0
from .recording import render_recording_to_mp4
from .robots import resolve_robot_control_profile, set_initial_pose
from .session import launch_teleop
from .viewer_controls import configure_live_terminal_output

def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Teleoperate the PandaOmron cardboard-box fragile-wall scene."
    )
    parser.add_argument(
        "--render-recording",
        type=Path,
        help="Render a saved teleop .npz recording to MP4 without opening the viewer.",
    )
    parser.add_argument(
        "--mp4-output",
        type=Path,
        help="Output MP4 path for --render-recording. Defaults to the .npz path with .mp4.",
    )
    parser.add_argument(
        "--render-view",
        type=str,
        default="main_camera",
        help="Camera/view to render for --render-recording.",
    )
    parser.add_argument("--width", type=int, default=VIDEO_WIDTH)
    parser.add_argument("--height", type=int, default=VIDEO_HEIGHT)
    parser.add_argument("--fps", type=int, default=VIDEO_FPS)
    parser.add_argument("--render-every", type=int, default=VIDEO_RENDER_EVERY)
    parser.add_argument(
        "--model-xml",
        type=str,
        default=str(MODEL_PATH),
        help="MuJoCo MJCF scene XML to load.",
    )
    parser.add_argument(
        "--random-scene",
        action="store_true",
        help=(
            "On reset, generate and reload a random RoboCasa scene whose layout "
            "contains the central island fixture required by fragile-wall-lift."
        ),
    )
    parser.add_argument(
        "--robocasa-root",
        type=Path,
        default=DEFAULT_ROBOCASA_ROOT,
        help="Local RoboCasa checkout used for --random-scene generation.",
    )
    parser.add_argument(
        "--robosuite-root",
        type=Path,
        default=DEFAULT_ROBOSUITE_ROOT,
        help="Local robosuite checkout used for --random-scene generation.",
    )
    parser.add_argument(
        "--random-scene-output-dir",
        type=Path,
        default=DEFAULT_RANDOM_SCENE_OUTPUT_DIR,
        help="Directory for generated random RoboCasa fragile-wall scene XML files.",
    )
    parser.add_argument(
        "--random-scene-layout-ids",
        type=int,
        nargs="+",
        help=(
            "Optional explicit RoboCasa layout IDs for --random-scene. "
            "Defaults to layouts whose YAML contains island_island_group."
        ),
    )
    parser.add_argument(
        "--random-scene-style-ids",
        type=int,
        nargs="+",
        help="Optional explicit RoboCasa style IDs for --random-scene. Defaults to 1..60.",
    )
    parser.add_argument(
        "--serial",
        type=str,
        default="144222110948",
        help="Serial number of the T265 camera.",
    )
    parser.add_argument("--rate-hz", type=float, default=200.0)
    parser.add_argument("--joystick", type=int, default=0)
    parser.add_argument(
        "--input",
        choices=("gamepad", "t265", "keyboard"),
        default="t265",
        help=(
            "Use T265 pose tracking, DualShock/gamepad sticks, or keyboard "
            "arrow-key control without external pose tracking."
        ),
    )
    parser.add_argument(
        "--t265-image",
        type=str,
        default="realsense_client",
        help="Docker image name for the T265 pose server.",
    )
    parser.add_argument(
        "--t265-translation-map",
        choices=("xzy", "zxy", "zyx", "yzx", "swap_xz", "xyz", "legacy"),
        default="xzy",
        help=(
            "Map T265 translation axes as x/z/y (default), z/x/y, y/z/x, "
            "z/y/x, raw x/y/z, or use the legacy teleop_rs_t265 mapping "
            "[-z, y, x]."
        ),
    )
    parser.add_argument(
        "--t265-translation-frame",
        choices=("world", "ee"),
        default="world",
        help=(
            "Apply T265 translation in robot world coordinates, or in the "
            "initial end-effector frame used by the legacy controller."
        ),
    )
    parser.add_argument(
        "--t265-rotation-map",
        choices=("quat_axes", "mapped", "legacy"),
        default="quat_axes",
        help=(
            "Map T265 rotation by reordering quaternion xyz components with "
            "the translation axis map, use the previous matrix-basis mapped "
            "rotation, or use the legacy teleop_rs_t265 rotation."
        ),
    )
    parser.add_argument(
        "--t265-rotation-axis-map",
        choices=("yzx", "zyx", "swap_xz", "xyz"),
        default="yzx",
        help="Axis map used only for T265 rotation when rotation map is not legacy.",
    )
    parser.add_argument(
        "--no-start-container",
        action="store_true",
        help="Use an already-running T265 HTTP server instead of starting Docker.",
    )
    parser.add_argument(
        "--no-video",
        action="store_true",
        help="Do not render MP4 when a recording is saved.",
    )
    parser.add_argument(
        "--video-views",
        nargs="+",
        default=list(DEFAULT_VIDEO_VIEWS),
        help="Views to render after each recording is saved.",
    )
    parser.add_argument(
        "--world-view",
        choices=("auto", "free", "fixed"),
        default="auto",
        help=(
            "Initial MuJoCo viewer camera. auto uses a computed diagonal "
            "workspace-centered free camera; fixed uses main_camera when present."
        ),
    )
    parser.add_argument(
        "--debug-view",
        action="store_true",
        help="Print the current MuJoCo viewer camera state periodically.",
    )
    parser.add_argument(
        "--debug-view-interval",
        type=float,
        default=DEBUG_VIEW_INTERVAL,
        help="Seconds between --debug-view camera prints.",
    )
    parser.add_argument(
        "--debug-control",
        action="store_true",
        help=(
            "Print keyboard EE position, target position, and target change "
            "direction while keyboard commands are active."
        ),
    )
    parser.add_argument(
        "--debug-control-interval",
        type=float,
        default=DEBUG_CONTROL_INTERVAL,
        help="Seconds between --debug-control prints.",
    )
    parser.add_argument(
        "--no-side-view",
        action="store_true",
        help="Do not open the additional OpenCV robot side-view window.",
    )
    parser.add_argument("--side-view-width", type=int, default=SIDE_VIEW_WIDTH)
    parser.add_argument("--side-view-height", type=int, default=SIDE_VIEW_HEIGHT)
    parser.add_argument("--side-view-fps", type=float, default=SIDE_VIEW_FPS)
    parser.add_argument(
        "--side-view-lookat",
        type=float,
        nargs=3,
        default=None,
        metavar=("X", "Y", "Z"),
        help="Free-camera lookat point for the OpenCV side-view window. Defaults to the tray center when present.",
    )
    parser.add_argument(
        "--side-view-distance",
        type=float,
        default=SIDE_VIEW_DISTANCE,
        help="Free-camera distance for the OpenCV side-view window.",
    )
    parser.add_argument(
        "--side-view-azimuth",
        type=float,
        default=None,
        help=(
            "Free-camera azimuth in degrees for the OpenCV side-view window. "
            "Defaults to the robot-facing right side of the current task area."
        ),
    )
    parser.add_argument(
        "--side-view-elevation",
        type=float,
        default=SIDE_VIEW_ELEVATION,
        help="Free-camera elevation in degrees for the OpenCV side-view window.",
    )
    parser.add_argument(
        "--task-name",
        type=str,
        default="panda_omron_cardboard_box_scene",
        help="Task name to store in recordings.",
    )
    parser.add_argument(
        "--language-prompt",
        type=str,
        default=(
            "Lift the red cardboard box by bracing it against the wooden tray wall "
            "until the entire box clears the wall."
        ),
        help="Language instruction to store for VLA training.",
    )
    parser.add_argument(
        "--goal-definition",
        type=str,
        help=(
            "Goal definition as JSON or path to JSON. Example: "
            '\'{"type":"box_in_tray","stable_steps":15}\''
        ),
    )
    parser.add_argument(
        "--auto-stop-on-goal",
        action="store_true",
        help="Stop and save recording automatically when the goal is satisfied.",
    )
    parser.add_argument(
        "--randomize-on-reset",
        action="store_true",
        help="Randomize the task object inside the surrounding walls at startup and reset.",
    )
    parser.add_argument(
        "--fragile-wall-break-force",
        type=float,
        default=FRAGILE_WALL_BREAK_FORCE_THRESHOLD,
        help="Contact force threshold in N for fragile-wall scenes.",
    )
    parser.add_argument("--toggle-button", type=int)
    parser.add_argument("--reset-button", type=int)
    parser.add_argument("--gripper-button", type=int)
    parser.add_argument("--record-button", type=int)
    parser.add_argument("--rotation-button", type=int)
    parser.add_argument("--rotation-axis", type=int, default=DEFAULT_AXIS_ROTATION)
    parser.add_argument("--rotation-axis-threshold", type=float, default=0.5)
    parser.add_argument("--axis-x", type=int, default=0)
    parser.add_argument("--axis-y", type=int, default=1)
    parser.add_argument("--axis-z", type=int, default=3)
    parser.add_argument("--axis-yaw", type=int, default=2)
    parser.add_argument("--deadzone", type=float, default=0.08)
    parser.add_argument("--translation-speed", type=float, default=TRANSLATION_SPEED)
    parser.add_argument("--rotation-speed", type=float, default=ROTATION_SPEED)
    parser.add_argument(
        "--ee-command-frame",
        choices=("base", "world"),
        default=DEFAULT_EE_COMMAND_FRAME,
        help=(
            "Frame for keyboard/gamepad EE translation and rotation commands. "
            "RoboCasa PandaOmron data collection uses base."
        ),
    )
    parser.add_argument(
        "--max-ee-target-position-error",
        type=float,
        default=MAX_EE_TARGET_POSITION_ERROR,
        help=(
            "Maximum distance in meters between the actual EE position and the "
            "commanded EE target position."
        ),
    )
    parser.add_argument(
        "--debug-pose",
        "--debug-pos",
        action="store_true",
        dest="debug_pose",
        help="Print T265 relative translation while T265 control is enabled.",
    )
    parser.add_argument(
        "--no-direct-tty-output",
        action="store_true",
        help="Do not bypass conda-run output capture by writing directly to /dev/tty.",
    )
    parser.add_argument(
        "--check-model-only",
        action="store_true",
        help="Load the scene and resolve controller names, then exit.",
    )
    return parser.parse_args()


def main() -> None:
    if "--no-direct-tty-output" not in sys.argv:
        configure_live_terminal_output()
    args = parse_args()
    if args.render_recording is not None:
        try:
            render_recording_to_mp4(
                args.render_recording,
                output_path=args.mp4_output,
                view_name=args.render_view,
                width=args.width,
                height=args.height,
                fps=args.fps,
                render_every=args.render_every,
            )
        except Exception as exc:
            print(f"MP4 rendering failed: {exc}", file=sys.stderr)
            sys.exit(1)
        return
    if args.check_model_only:
        model_path = Path(args.model_xml).expanduser().resolve()
        model = mujoco.MjModel.from_xml_path(str(model_path))
        data = mujoco.MjData(model)
        robot_profile = resolve_robot_control_profile(model)
        set_initial_pose(model, data, robot_profile)
        reset_freejoint_to_model_qpos0(model, data, CARDBOARD_BOX_FREEJOINT)
        align_freejoint_body_bottom_to_support(model, data)
        CartesianController(model, data, robot_profile)
        print(f"Loaded model: {model_path}")
        return
    launch_teleop(args)
