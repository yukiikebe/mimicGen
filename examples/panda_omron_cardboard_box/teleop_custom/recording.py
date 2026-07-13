from __future__ import annotations

from .common import (
    CARDBOARD_BOX_BODY,
    DEFAULT_RECORDING_DIR,
    DEFAULT_VIDEO_VIEWS,
    ENTRYPOINT_PATH,
    MAX_RECORDED_CONTACTS,
    MODEL_PATH,
    Path,
    TACTILE_BOX_BODY,
    VIDEO_FPS,
    VIDEO_HEIGHT,
    VIDEO_RENDER_EVERY,
    VIDEO_WIDTH,
    WOODEN_TRAY_BODY,
    contextlib,
    datetime,
    io,
    mujoco,
    np,
    os,
    subprocess,
    sys,
    warnings,
)
from .cameras import make_visual_only_scene_option, resolve_recording_camera, view_video_path
from .contacts import extract_tactile_contacts, make_tactile_contact_filter
from .control import CartesianController
from .robots import RobotControlProfile

class TeleopRecorder:
    def __init__(
        self,
        model: mujoco.MjModel,
        camera_id: int,
        robot_profile: RobotControlProfile,
        output_dir: Path = DEFAULT_RECORDING_DIR,
        enable_video: bool = True,
        video_views: tuple[str, ...] = DEFAULT_VIDEO_VIEWS,
        extra_metadata: dict | None = None,
        frame_metadata_fn=None,
        model_xml_path: str | Path = MODEL_PATH,
    ) -> None:
        self.model = model
        self.camera_id = camera_id
        self.robot_profile = robot_profile
        self.output_dir = output_dir
        self.enable_video = enable_video
        self.video_views = tuple(video_views)
        self.extra_metadata = dict(extra_metadata or {})
        self.frame_metadata_fn = frame_metadata_fn
        self.model_xml_path = str(model_xml_path)
        self.recording = False
        self._tracked_body_names = self._resolve_tracked_body_names()
        self._body_ids = self._resolve_body_ids()
        self._contact_filter = make_tactile_contact_filter(
            model, robot_profile.tactile_gripper_bodies
        )
        self._frames: list[dict[str, np.ndarray | float]] = []
        self._start_wall_time: datetime.datetime | None = None
        self._base_path: Path | None = None
        self._record_step = 0

    def _resolve_tracked_body_names(self) -> tuple[str, ...]:
        names = [
            CARDBOARD_BOX_BODY,
            WOODEN_TRAY_BODY,
            self.robot_profile.eef_body,
            *self.robot_profile.tactile_gripper_bodies,
        ]
        return tuple(dict.fromkeys(names))

    def _resolve_body_ids(self) -> np.ndarray:
        ids = []
        for name in self._tracked_body_names:
            body_id = mujoco.mj_name2id(self.model, mujoco.mjtObj.mjOBJ_BODY, name)
            if body_id < 0:
                raise ValueError(f"Missing body: {name}")
            ids.append(body_id)
        return np.asarray(ids, dtype=np.int32)

    def toggle(self) -> None:
        if self.recording:
            self.stop_and_save()
        else:
            self.start()

    def set_extra_metadata(self, **metadata) -> None:
        self.extra_metadata.update(metadata)

    def start(self) -> None:
        self._frames = []
        self._start_wall_time = datetime.datetime.now()
        self._base_path = (
            self.output_dir
            / f"teleop_{self._start_wall_time.strftime('%Y%m%d_%H%M%S')}"
        )
        self._record_step = 0
        self.recording = True
        if self.enable_video and not self.video_views:
            print("MP4 rendering disabled: no video views were requested.")
        print("Recording started.")

    def record(
        self,
        data: mujoco.MjData,
        controller: CartesianController,
    ) -> None:
        if not self.recording:
            return

        frame = {
            "time": float(data.time),
            "qpos": data.qpos.copy(),
            "qvel": data.qvel.copy(),
            "ctrl": data.ctrl.copy(),
            "body_pos": data.xpos[self._body_ids].copy(),
            "body_quat": data.xquat[self._body_ids].copy(),
            "target_pos": controller.target_pos.copy(),
            "target_rot": controller.target_rot.copy(),
            **extract_tactile_contacts(
                self.model, data, self._contact_filter, MAX_RECORDED_CONTACTS
            ),
        }
        if self.frame_metadata_fn is not None:
            frame.update(self.frame_metadata_fn(data, controller))
        self._frames.append(frame)
        self._record_step += 1

    def _render_video_in_subprocess(
        self, recording_path: Path, video_path: Path, view_name: str
    ) -> None:
        if not self.enable_video:
            return

        command = [
            sys.executable,
            str(ENTRYPOINT_PATH),
            "--render-recording",
            str(recording_path),
            "--mp4-output",
            str(video_path),
            "--render-view",
            view_name,
        ]
        try:
            subprocess.Popen(command)
            print(
                f"MP4 rendering started in a separate process: {view_name} -> {video_path}"
            )
        except Exception as exc:
            print(f"MP4 rendering could not be started for {view_name}: {exc}")

    def stop_and_save(self) -> Path | None:
        if not self.recording:
            return None

        self.recording = False
        if not self._frames:
            print("Recording stopped. No frames captured.")
            return None

        output = (
            self._base_path.with_suffix(".npz")
            if self._base_path is not None
            else self.output_dir / "teleop_recording.npz"
        )
        output.parent.mkdir(parents=True, exist_ok=True)
        video_outputs = []
        if self.enable_video and self._base_path is not None:
            video_outputs = [
                (view_name, view_video_path(self._base_path, view_name))
                for view_name in self.video_views
            ]

        payload = {
            "time": np.asarray([frame["time"] for frame in self._frames]),
            "qpos": np.stack([frame["qpos"] for frame in self._frames]),
            "qvel": np.stack([frame["qvel"] for frame in self._frames]),
            "ctrl": np.stack([frame["ctrl"] for frame in self._frames]),
            "body_pos": np.stack([frame["body_pos"] for frame in self._frames]),
            "body_quat": np.stack([frame["body_quat"] for frame in self._frames]),
            "target_pos": np.stack([frame["target_pos"] for frame in self._frames]),
            "target_rot": np.stack([frame["target_rot"] for frame in self._frames]),
            "contact_count": np.asarray(
                [frame["contact_count"] for frame in self._frames], dtype=np.int32
            ),
            "contact_pos": np.stack([frame["contact_pos"] for frame in self._frames]),
            "contact_frame": np.stack(
                [frame["contact_frame"] for frame in self._frames]
            ),
            "contact_dist": np.stack([frame["contact_dist"] for frame in self._frames]),
            "contact_geom1": np.stack(
                [frame["contact_geom1"] for frame in self._frames]
            ),
            "contact_geom2": np.stack(
                [frame["contact_geom2"] for frame in self._frames]
            ),
            "contact_force": np.stack(
                [frame["contact_force"] for frame in self._frames]
            ),
            "body_names": np.asarray(self._tracked_body_names),
            "actuator_names": np.asarray(
                [
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_ACTUATOR, i)
                    for i in range(self.model.nu)
                ]
            ),
            "joint_names": np.asarray(
                [
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_JOINT, i)
                    for i in range(self.model.njnt)
                ]
            ),
            "geom_names": np.asarray(
                [
                    mujoco.mj_id2name(self.model, mujoco.mjtObj.mjOBJ_GEOM, i) or ""
                    for i in range(self.model.ngeom)
                ]
            ),
            "contact_max_contacts": np.asarray(MAX_RECORDED_CONTACTS, dtype=np.int32),
            "contact_filter_box_body": np.asarray(TACTILE_BOX_BODY),
            "contact_filter_gripper_bodies": np.asarray(
                self.robot_profile.tactile_gripper_bodies
            ),
            "model_xml": self.model_xml_path,
            "video_path": str(video_outputs[0][1]) if video_outputs else "",
            "video_paths": np.asarray([str(path) for _, path in video_outputs]),
            "video_views": np.asarray([view_name for view_name, _ in video_outputs]),
        }
        core_keys = set(payload)
        for key in self._frames[0]:
            if key in core_keys:
                continue
            values = [frame[key] for frame in self._frames]
            try:
                payload[key] = np.stack(values)
            except ValueError:
                payload[key] = np.asarray(values)
        for key, value in self.extra_metadata.items():
            payload[key] = np.asarray(value)
        np.savez_compressed(output, **payload)
        print(f"Recording saved: {output} ({len(self._frames)} frames)")
        for view_name, video_output in video_outputs:
            self._render_video_in_subprocess(output, video_output, view_name)
        self._frames = []
        return output


def render_recording_to_mp4(
    recording_path: Path,
    output_path: Path | None = None,
    view_name: str = "main_camera",
    width: int = VIDEO_WIDTH,
    height: int = VIDEO_HEIGHT,
    fps: int = VIDEO_FPS,
    render_every: int = VIDEO_RENDER_EVERY,
) -> Path:
    warnings.filterwarnings(
        "ignore",
        message="pkg_resources is deprecated as an API.*",
        category=UserWarning,
    )
    import imageio.v2 as imageio

    recording_path = recording_path.expanduser().resolve()
    if output_path is None:
        output_path = view_video_path(recording_path.with_suffix(""), view_name)
    output_path = output_path.expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)

    recording = np.load(recording_path)
    model_xml = (
        str(recording["model_xml"]) if "model_xml" in recording else str(MODEL_PATH)
    )
    model = mujoco.MjModel.from_xml_path(model_xml)
    data = mujoco.MjData(model)

    qpos = recording["qpos"]
    if qpos.ndim != 2 or qpos.shape[1] != model.nq:
        raise ValueError(
            f"Recording qpos shape {qpos.shape} does not match model nq={model.nq}"
        )

    qvel = recording["qvel"] if "qvel" in recording else None
    camera = resolve_recording_camera(model, view_name)
    scene_option = make_visual_only_scene_option()

    mujoco_gl = os.environ.get("MUJOCO_GL", "").lower()
    if mujoco_gl in ("", "glfw"):
        import glfw

        glfw_stderr = io.StringIO()
        try:
            with contextlib.redirect_stderr(glfw_stderr):
                glfw_ready = glfw.init()
        except Exception as exc:
            raise RuntimeError(f"GLFW could not initialize ({exc})") from exc
        if not glfw_ready:
            detail = glfw_stderr.getvalue().strip()
            suffix = f": {detail}" if detail else ""
            raise RuntimeError(f"GLFW could not initialize{suffix}")

    renderer = None
    writer = None
    frame_count = 0
    try:
        renderer_stderr = io.StringIO()
        with contextlib.redirect_stderr(renderer_stderr):
            renderer = mujoco.Renderer(model, height=height, width=width)
        writer = imageio.get_writer(output_path, fps=fps)
        for index in range(0, len(qpos), max(1, render_every)):
            data.qpos[:] = qpos[index]
            if qvel is not None and qvel.shape == (len(qpos), model.nv):
                data.qvel[:] = qvel[index]
            mujoco.mj_forward(model, data)
            renderer.update_scene(data, camera=camera, scene_option=scene_option)
            writer.append_data(renderer.render())
            frame_count += 1
    finally:
        if writer is not None:
            writer.close()
        if renderer is not None:
            renderer.close()

    print(f"MP4 saved: {output_path} ({view_name}, {frame_count} frames)")
    return output_path
