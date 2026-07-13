from __future__ import annotations

from .common import (
    Any,
    CARDBOARD_BOX_FREEJOINT,
    Path,
    argparse,
    mujoco,
    np,
    sys,
    time,
)
from .cameras import SideViewRenderer, apply_visual_only_viewer_options, print_scene_anchor_positions, print_view_camera_debug, resolve_side_view_azimuth, resolve_side_view_lookat, set_world_camera_defaults
from .control import CartesianController, KeyboardState
from .fragile_wall import FragileWall
from .goals import GoalEvaluator, RecordingMetadataProvider, load_goal_definition
from .inputs import Gamepad, T265Teleop, apply_gamepad_target_delta, print_teleop_help, toggle_gripper
from .math_utils import print_keyboard_control_debug
from .placement import align_freejoint_body_bottom_to_support, randomize_red_box_xy_in_tray, reset_freejoint_to_model_qpos0
from .recording import TeleopRecorder
from .robocasa_scene import generate_random_island_scene_xml
from .robots import lock_mobile_base, resolve_robot_control_profile, set_initial_pose
from .viewer_controls import make_key_callback

def launch_teleop(args: argparse.Namespace) -> None:
    model_path = Path(args.model_xml).expanduser().resolve()
    randomize_at_start = False
    while True:
        print(f"Starting teleop session: model={model_path}", flush=True)
        next_model_path = run_teleop_session(
            args,
            model_path=model_path,
            randomize_at_start=randomize_at_start,
        )
        if next_model_path is None:
            return
        model_path = next_model_path
        randomize_at_start = True


def run_teleop_session(
    args: argparse.Namespace, model_path: Path, randomize_at_start: bool = False
) -> Path | None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    robot_profile = resolve_robot_control_profile(model)
    set_initial_pose(model, data, robot_profile)
    lock_mobile_base(model, data, robot_profile)
    reset_freejoint_to_model_qpos0(model, data, CARDBOARD_BOX_FREEJOINT)
    align_freejoint_body_bottom_to_support(model, data)
    print_scene_anchor_positions(model, data)
    rng = np.random.default_rng()
    runtime_state: dict[str, Any] = {"fragile_wall_broken": False}
    fragile_wall = FragileWall(
        model=model,
        runtime_state=runtime_state,
        robot_profile=robot_profile,
        break_force_threshold=args.fragile_wall_break_force,
    )
    fragile_wall.reset(data)
    if args.randomize_on_reset or randomize_at_start:
        new_xy = randomize_red_box_xy_in_tray(model, data, rng)
        print(
            f"Randomized object XY: x={new_xy[0]:+.3f}, y={new_xy[1]:+.3f}",
            flush=True,
        )

    camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, "main_camera")
    cartesian_controller = CartesianController(
        model,
        data,
        robot_profile,
        translation_speed=args.translation_speed,
        rotation_speed=args.rotation_speed,
        max_target_position_error=args.max_ee_target_position_error,
        command_frame=args.ee_command_frame,
    )
    keyboard_state = KeyboardState()
    goal_definition = load_goal_definition(args.goal_definition)
    goal_evaluator = GoalEvaluator(model, goal_definition, runtime_state=runtime_state)
    metadata_provider = RecordingMetadataProvider(
        model=model,
        robot_profile=robot_profile,
        goal_evaluator=goal_evaluator,
        language_prompt=args.language_prompt,
        goal_definition=goal_definition,
        task_name=args.task_name,
        runtime_state=runtime_state,
    )
    recorder = TeleopRecorder(
        model,
        camera_id,
        robot_profile,
        enable_video=not args.no_video,
        video_views=tuple(args.video_views),
        extra_metadata=metadata_provider.static_metadata(),
        frame_metadata_fn=metadata_provider,
        model_xml_path=model_path,
    )
    gamepad = None
    if args.input in {"gamepad", "t265"}:
        gamepad = Gamepad(
            joystick_index=args.joystick,
            toggle_button=args.toggle_button,
            reset_button=args.reset_button,
            gripper_button=args.gripper_button,
            record_button=args.record_button,
            rotation_button=args.rotation_button,
            rotation_axis=args.rotation_axis,
            rotation_axis_threshold=args.rotation_axis_threshold,
        )
    t265_teleop = None
    if args.input == "t265":
        t265_teleop = T265Teleop(
            model=model,
            data=data,
            cartesian_controller=cartesian_controller,
            serial=args.serial,
            rate_hz=args.rate_hz,
            start_container=not args.no_start_container,
            image=args.t265_image,
            translation_map=args.t265_translation_map,
            rotation_map=args.t265_rotation_map,
            rotation_axis_map=args.t265_rotation_axis_map,
            translation_frame=args.t265_translation_frame,
            debug_pose=args.debug_pose,
        )
    control_enabled = False
    goal_stable_count = 0
    auto_stop_on_goal = args.auto_stop_on_goal or args.goal_definition is not None
    pending_keyboard_reset = False
    pending_gripper_toggle = False
    pending_recording_toggle = False
    pending_translation_map_toggle = False
    last_debug_view_time = 0.0
    last_debug_control_time = 0.0

    viewer_holder: dict[str, Any] = {}
    next_model_path: Path | None = None
    side_view = None

    def refresh_camera_views() -> None:
        """Recompute live cameras from the current wall/object task center."""
        viewer = viewer_holder.get("viewer")
        if viewer is not None and viewer.cam.type == mujoco.mjtCamera.mjCAMERA_FREE:
            set_world_camera_defaults(viewer.cam, model, data)
        if side_view is None:
            return
        side_view.camera.lookat[:] = resolve_side_view_lookat(
            model,
            data,
            args.side_view_lookat,
        )
        if args.side_view_azimuth is None:
            side_view_azimuth = resolve_side_view_azimuth(model, data, None)
            side_view.primary_azimuth = side_view_azimuth
            side_view.alternate_azimuth = side_view_azimuth + 180.0
            side_view.camera.azimuth = (
                side_view.alternate_azimuth
                if side_view._using_alternate_azimuth
                else side_view.primary_azimuth
            )

    def hide_viewer_overlays() -> None:
        viewer = viewer_holder.get("viewer")
        if viewer is None:
            return
        apply_visual_only_viewer_options(viewer)
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTPOINT] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_CONTACTFORCE] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_COM] = 0
        viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_SELECT] = 0

    base_key_callback = make_key_callback(
        model,
        data,
        viewer_holder,
        camera_id,
        cartesian_controller,
        robot_profile,
        keyboard_state,
        recorder,
    )

    def reset_scene(randomize_box: bool = False) -> None:
        nonlocal control_enabled, goal_stable_count, next_model_path
        control_enabled = False
        goal_stable_count = 0
        if t265_teleop is not None:
            t265_teleop.in_control = False
        if args.random_scene:
            if recorder.recording:
                recorder.stop_and_save()
            try:
                next_model_path = generate_random_island_scene_xml(args, rng)
            except Exception as exc:
                next_model_path = None
                print(f"Random scene reset failed: {exc}", file=sys.stderr, flush=True)
                return
            print(f"Reloading random scene: {next_model_path}", flush=True)
            return
        set_initial_pose(model, data, robot_profile)
        lock_mobile_base(model, data, robot_profile)
        reset_freejoint_to_model_qpos0(model, data, CARDBOARD_BOX_FREEJOINT)
        align_freejoint_body_bottom_to_support(model, data)
        fragile_wall.reset(data)
        if randomize_box or args.randomize_on_reset:
            new_xy = randomize_red_box_xy_in_tray(model, data, rng)
            print(
                f"Randomized object XY: x={new_xy[0]:+.3f}, y={new_xy[1]:+.3f}",
                flush=True,
            )
        refresh_camera_views()
        cartesian_controller.reset_target_to_current()
        if t265_teleop is not None:
            t265_teleop.reset_stream()
        print("Resetting scene.")

    def toggle_control() -> None:
        nonlocal control_enabled, goal_stable_count
        if t265_teleop is not None:
            t265_teleop.toggle_control()
            control_now_enabled = t265_teleop.in_control
        else:
            control_enabled = not control_enabled
            cartesian_controller.reset_target_to_current()
            control_now_enabled = control_enabled
            print(
                f"Control {'enabled' if control_enabled else 'disabled'}.", flush=True
            )
        if control_now_enabled:
            goal_stable_count = 0

    def reset_episode_flags() -> None:
        recorder.set_extra_metadata(
            episode_success=False,
            episode_failure=False,
            episode_goal_satisfied_step=-1,
            episode_failure_step=-1,
        )

    def start_recording() -> None:
        reset_episode_flags()
        recorder.start()

    def start_recording_if_needed() -> None:
        if recorder.recording:
            print("Recording is already active.", flush=True)
            return
        start_recording()

    def finish_recording(success: bool, reason: str) -> None:
        nonlocal control_enabled, goal_stable_count
        if not recorder.recording:
            print(f"Ignoring {reason}: recording is not active.", flush=True)
            return
        recorder.set_extra_metadata(
            episode_success=success,
            episode_failure=not success,
            episode_goal_satisfied_step=recorder._record_step if success else -1,
            episode_failure_step=recorder._record_step if not success else -1,
        )
        recorder.stop_and_save()
        if t265_teleop is not None:
            t265_teleop.in_control = False
        control_enabled = False
        goal_stable_count = 0
        print(
            f"Recording marked {'success' if success else 'failure'} by {reason}.",
            flush=True,
        )

    def key_callback(key: int) -> None:
        nonlocal pending_gripper_toggle, pending_keyboard_reset
        nonlocal pending_recording_toggle, pending_translation_map_toggle
        try:
            char = chr(key).lower()
        except ValueError:
            char = ""
        if char == "r":
            pending_keyboard_reset = True
            return
        if char == "0":
            pending_gripper_toggle = True
            return
        if char == "u" and args.input != "keyboard":
            toggle_control()
            return
        if char == "c":
            if args.input == "keyboard":
                pending_recording_toggle = True
                return
            print(
                "Recording starts only from D-pad right in this teleop.",
                flush=True,
            )
            hide_viewer_overlays()
            return
        if char == "m":
            pending_translation_map_toggle = True
            return
        if char == "n" and side_view is not None:
            side_view.toggle_side()
            hide_viewer_overlays()
            return
        base_key_callback(key)
        hide_viewer_overlays()

    try:
        print_teleop_help(args.input, random_scene=args.random_scene)
        keyboard_state.start()
        if t265_teleop is not None:
            t265_teleop.start()

        with mujoco.viewer.launch_passive(
            model, data, key_callback=key_callback
        ) as viewer:
            viewer_holder["viewer"] = viewer
            apply_visual_only_viewer_options(viewer)
            use_fixed_world_view = args.world_view == "fixed" and camera_id >= 0
            if use_fixed_world_view and camera_id >= 0:
                viewer.cam.type = mujoco.mjtCamera.mjCAMERA_FIXED
                viewer.cam.fixedcamid = camera_id
            else:
                set_world_camera_defaults(viewer.cam, model, data)

            if not args.no_side_view:
                fixed_side_camera_id = mujoco.mj_name2id(
                    model, mujoco.mjtObj.mjOBJ_CAMERA, "robot_side_view"
                )
                if fixed_side_camera_id < 0:
                    fixed_side_camera_id = None
                side_view_azimuth = resolve_side_view_azimuth(
                    model,
                    data,
                    args.side_view_azimuth,
                )
                side_view = SideViewRenderer(
                    model,
                    width=args.side_view_width,
                    height=args.side_view_height,
                    fps=args.side_view_fps,
                    lookat=resolve_side_view_lookat(
                        model,
                        data,
                        args.side_view_lookat,
                    ),
                    distance=args.side_view_distance,
                    azimuth=side_view_azimuth,
                    alternate_azimuth=side_view_azimuth + 180.0,
                    elevation=args.side_view_elevation,
                    fixed_camera_id=fixed_side_camera_id,
                )

            while viewer.is_running():
                if args.debug_view:
                    now = time.monotonic()
                    if now - last_debug_view_time >= args.debug_view_interval:
                        last_debug_view_time = now
                        print_view_camera_debug(model, data, viewer.cam)

                if pending_keyboard_reset:
                    pending_keyboard_reset = False
                    if args.random_scene:
                        reset_scene()
                    else:
                        with viewer.lock():
                            reset_scene()
                if pending_gripper_toggle:
                    pending_gripper_toggle = False
                    with viewer.lock():
                        toggle_gripper(model, data, robot_profile)
                        hide_viewer_overlays()
                if pending_recording_toggle:
                    pending_recording_toggle = False
                    if recorder.recording:
                        recorder.stop_and_save()
                    else:
                        start_recording()
                    hide_viewer_overlays()
                if pending_translation_map_toggle:
                    pending_translation_map_toggle = False
                    with viewer.lock():
                        if t265_teleop is not None:
                            t265_teleop.toggle_translation_xy_map()
                        else:
                            cartesian_controller.toggle_keyboard_xy_map()
                        hide_viewer_overlays()

                if gamepad is not None:
                    gamepad.poll()
                    if gamepad.consume("toggle_control"):
                        toggle_control()
                    if gamepad.consume("reset"):
                        if args.random_scene:
                            reset_scene()
                        else:
                            with viewer.lock():
                                reset_scene()
                    if gamepad.consume("reset_randomize"):
                        if args.random_scene:
                            reset_scene(randomize_box=True)
                        else:
                            with viewer.lock():
                                reset_scene(randomize_box=True)
                    if gamepad.consume("gripper"):
                        toggle_gripper(model, data, robot_profile)
                    if gamepad.consume("record"):
                        start_recording_if_needed()
                    if gamepad.consume("success"):
                        finish_recording(success=True, reason="D-pad up")
                    if gamepad.consume("failure"):
                        finish_recording(success=False, reason="D-pad down")

                if next_model_path is not None:
                    print("Leaving current viewer for random scene reload.", flush=True)
                    if side_view is not None:
                        side_view.close()
                        side_view = None
                    break

                if t265_teleop is not None and t265_teleop.in_control:
                    command_active = t265_teleop.update_target(
                        apply_rotation=gamepad is not None
                        and gamepad.rotation_enabled()
                    )
                elif (
                    args.input == "gamepad" and control_enabled and gamepad is not None
                ):
                    command_active = apply_gamepad_target_delta(
                        gamepad, cartesian_controller, args, model.opt.timestep
                    )
                else:
                    pressed_keys, shift_pressed = keyboard_state.snapshot()
                    previous_target_pos = cartesian_controller.target_pos.copy()
                    previous_target_rot = cartesian_controller.target_rot.copy()
                    command_active = cartesian_controller.apply_pressed_keys(
                        pressed_keys, shift_pressed, model.opt.timestep
                    )
                    if command_active and args.debug_control:
                        now = time.monotonic()
                        if now - last_debug_control_time >= args.debug_control_interval:
                            last_debug_control_time = now
                            print_keyboard_control_debug(
                                model,
                                data,
                                cartesian_controller,
                                previous_target_pos,
                                previous_target_rot,
                            )
                if not command_active:
                    cartesian_controller.hold_current_pose(reset_arm_velocity=True)

                cartesian_controller.update_control()
                lock_mobile_base(model, data, robot_profile)
                mujoco.mj_step(model, data)
                lock_mobile_base(model, data, robot_profile)
                mujoco.mj_forward(model, data)
                wall_broke_this_step = fragile_wall.update(data)
                recorder.record(data, cartesian_controller)
                if recorder.recording and wall_broke_this_step:
                    finish_recording(success=False, reason="fragile wall break")
                if recorder.recording and auto_stop_on_goal:
                    if metadata_provider.last_goal_satisfied:
                        goal_stable_count += 1
                    else:
                        goal_stable_count = 0
                    if goal_stable_count >= goal_evaluator.stable_steps:
                        finish_recording(success=True, reason="goal")
                hide_viewer_overlays()
                if side_view is not None:
                    side_view.update(data)
                viewer.sync()
                time.sleep(model.opt.timestep)
    finally:
        if side_view is not None:
            side_view.close()
        recorder.stop_and_save()
        keyboard_state.stop()
        if t265_teleop is not None:
            t265_teleop.stop()
        if gamepad is not None:
            gamepad.close()
    return next_model_path
