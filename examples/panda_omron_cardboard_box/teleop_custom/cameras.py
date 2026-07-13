from __future__ import annotations

from .common import (
    CARDBOARD_BOX_BODY,
    LEFT_SIDE_VIEW_AZIMUTH,
    Path,
    SIDE_VIEW_AZIMUTH,
    SIDE_VIEW_DISTANCE,
    SIDE_VIEW_ELEVATION,
    SIDE_VIEW_FPS,
    SIDE_VIEW_HEIGHT,
    SIDE_VIEW_LOOKAT,
    SIDE_VIEW_WIDTH,
    SIDE_VIEW_WINDOW,
    VISUAL_GEOM_GROUP,
    WOODEN_TRAY_BODY,
    WORKSPACE_VIEW_AZIMUTH,
    WORKSPACE_VIEW_DISTANCE,
    WORKSPACE_VIEW_ELEVATION,
    WORKSPACE_VIEW_REAR_SIDE_OFFSET,
    WORLD_VIEW_AZIMUTH,
    WORLD_VIEW_DISTANCE,
    WORLD_VIEW_ELEVATION,
    WORLD_VIEW_GLASS_WALL_Z_OFFSET,
    WORLD_VIEW_MAX_DISTANCE,
    WORLD_VIEW_MIN_DISTANCE,
    WRIST_CAMERA_NAME,
    contextlib,
    io,
    mujoco,
    np,
    time,
)
from .robots import resolve_robot_control_profile

def set_free_camera_defaults(
    cam,
    lookat: np.ndarray | None = None,
    distance: float = WORLD_VIEW_DISTANCE,
    azimuth: float = WORLD_VIEW_AZIMUTH,
    elevation: float = WORLD_VIEW_ELEVATION,
) -> None:
    cam.type = mujoco.mjtCamera.mjCAMERA_FREE
    cam.lookat[:] = SIDE_VIEW_LOOKAT if lookat is None else lookat
    cam.distance = float(distance)
    cam.azimuth = float(azimuth)
    cam.elevation = float(elevation)


def make_visual_only_scene_option() -> mujoco.MjvOption:
    scene_option = mujoco.MjvOption()
    scene_option.geomgroup[:] = 0
    scene_option.geomgroup[VISUAL_GEOM_GROUP] = 1
    scene_option.sitegroup[:] = 0
    return scene_option


def apply_visual_only_viewer_options(viewer) -> None:
    viewer.opt.geomgroup[:] = 0
    viewer.opt.geomgroup[VISUAL_GEOM_GROUP] = 1
    viewer.opt.sitegroup[:] = 0
    viewer.opt.flags[mujoco.mjtVisFlag.mjVIS_INERTIA] = 0


class SideViewRenderer:
    def __init__(
        self,
        model: mujoco.MjModel,
        width: int = SIDE_VIEW_WIDTH,
        height: int = SIDE_VIEW_HEIGHT,
        fps: float = SIDE_VIEW_FPS,
        window_name: str = SIDE_VIEW_WINDOW,
        lookat: np.ndarray = SIDE_VIEW_LOOKAT,
        distance: float = SIDE_VIEW_DISTANCE,
        azimuth: float = SIDE_VIEW_AZIMUTH,
        alternate_azimuth: float | None = None,
        elevation: float = SIDE_VIEW_ELEVATION,
        fixed_camera_id: int | None = None,
    ) -> None:
        self.enabled = False
        self.cv2 = None
        self.renderer = None
        self.window_name = window_name
        self.fixed_camera_id = fixed_camera_id
        self.min_interval = 1.0 / fps if fps > 0.0 else 0.0
        self._last_render = 0.0
        self.primary_azimuth = float(azimuth)
        self.alternate_azimuth = (
            float(alternate_azimuth)
            if alternate_azimuth is not None
            else float(azimuth) + 180.0
        )
        self._using_alternate_azimuth = False
        self.scene_option = make_visual_only_scene_option()
        self.camera = mujoco.MjvCamera()
        mujoco.mjv_defaultCamera(self.camera)
        if fixed_camera_id is not None:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FIXED
            self.camera.fixedcamid = int(fixed_camera_id)
        else:
            self.camera.type = mujoco.mjtCamera.mjCAMERA_FREE
            self.camera.lookat[:] = lookat
            self.camera.distance = float(distance)
            self.camera.azimuth = float(azimuth)
            self.camera.elevation = float(elevation)

        try:
            import cv2
        except Exception as exc:
            print(f"Side OpenCV view disabled: cv2 is unavailable ({exc})")
            return

        try:
            renderer_stderr = io.StringIO()
            with contextlib.redirect_stderr(renderer_stderr):
                self.renderer = mujoco.Renderer(model, height=height, width=width)
            cv2.namedWindow(self.window_name, cv2.WINDOW_NORMAL)
        except Exception as exc:
            if self.renderer is not None:
                self.renderer.close()
                self.renderer = None
            print(f"Side OpenCV view disabled: renderer could not start ({exc})")
            return

        self.cv2 = cv2
        self.enabled = True
        if fixed_camera_id is not None:
            camera_name = mujoco.mj_id2name(
                model, mujoco.mjtObj.mjOBJ_CAMERA, int(fixed_camera_id)
            )
            print(f"Side OpenCV view enabled ({self.window_name}, fixed={camera_name}).")
            return
        print(
            "Side OpenCV view enabled "
            f"({self.window_name}, azimuth={self.camera.azimuth:.1f})."
        )

    def update(self, data: mujoco.MjData) -> None:
        if not self.enabled or self.renderer is None or self.cv2 is None:
            return

        now = time.monotonic()
        if now - self._last_render < self.min_interval:
            return
        self._last_render = now

        try:
            camera = self.fixed_camera_id if self.fixed_camera_id is not None else self.camera
            self.renderer.update_scene(data, camera=camera, scene_option=self.scene_option)
            rgb = self.renderer.render()
            bgr = np.ascontiguousarray(rgb[:, :, ::-1])
            self.cv2.imshow(self.window_name, bgr)
            key = self.cv2.waitKey(1) & 0xFF
            if key in (ord("q"), 27):
                self.close()
            elif key == ord("n"):
                self.toggle_side()
        except Exception as exc:
            print(f"Side OpenCV view disabled: render failed ({exc})")
            self.close()

    def toggle_side(self) -> None:
        if not self.enabled:
            return
        if self.fixed_camera_id is not None:
            print(
                "Side OpenCV view uses fixed robot_side_view; left/right toggle is disabled.",
                flush=True,
            )
            return
        self._using_alternate_azimuth = not self._using_alternate_azimuth
        self.camera.azimuth = (
            self.alternate_azimuth
            if self._using_alternate_azimuth
            else self.primary_azimuth
        )
        side = "left" if self._using_alternate_azimuth else "right"
        print(
            "Side OpenCV view: "
            f"{side} side, azimuth={self.camera.azimuth:.4f}, "
            f"distance={self.camera.distance:.4f}, "
            f"elevation={self.camera.elevation:.4f}",
            flush=True,
        )

    def close(self) -> None:
        if self.renderer is not None:
            self.renderer.close()
            self.renderer = None
        if self.cv2 is not None:
            try:
                self.cv2.destroyWindow(self.window_name)
            except Exception:
                pass
        self.enabled = False


def make_free_camera(
    lookat: np.ndarray = SIDE_VIEW_LOOKAT,
    distance: float = SIDE_VIEW_DISTANCE,
    azimuth: float = SIDE_VIEW_AZIMUTH,
    elevation: float = SIDE_VIEW_ELEVATION,
) -> mujoco.MjvCamera:
    camera = mujoco.MjvCamera()
    mujoco.mjv_defaultCamera(camera)
    camera.type = mujoco.mjtCamera.mjCAMERA_FREE
    camera.lookat[:] = lookat
    camera.distance = float(distance)
    camera.azimuth = float(azimuth)
    camera.elevation = float(elevation)
    return camera


def resolve_side_view_lookat(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configured_lookat: list[float] | None,
) -> np.ndarray:
    if configured_lookat is not None:
        return np.asarray(configured_lookat, dtype=np.float64)
    mujoco.mj_forward(model, data)
    glass_wall_points = collect_glass_wall_points(model, data)
    if glass_wall_points:
        lookat = np.mean(np.asarray(glass_wall_points, dtype=np.float64), axis=0)
        lookat[2] += WORLD_VIEW_GLASS_WALL_Z_OFFSET
        return lookat

    tray_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WOODEN_TRAY_BODY)
    if tray_body_id >= 0:
        return data.xpos[tray_body_id].copy() + np.array([0.0, 0.0, 0.25])
    return SIDE_VIEW_LOOKAT.copy()


def normalized_xy(vector: np.ndarray) -> np.ndarray:
    vector = np.asarray(vector, dtype=np.float64)
    norm = float(np.linalg.norm(vector))
    if norm < 1e-9:
        raise ValueError("Cannot normalize near-zero XY vector.")
    return vector / norm


def camera_azimuth_from_eye_direction(direction_xy: np.ndarray) -> float:
    direction_xy = normalized_xy(np.asarray(direction_xy, dtype=np.float64))
    return float(np.degrees(np.arctan2(-direction_xy[0], direction_xy[1])))


def resolve_side_view_azimuth(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configured_azimuth: float | None,
) -> float:
    if configured_azimuth is not None:
        return float(configured_azimuth)
    mujoco.mj_forward(model, data)
    tray_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WOODEN_TRAY_BODY)
    robot_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base")
    if tray_body_id < 0 or robot_body_id < 0:
        return SIDE_VIEW_AZIMUTH
    robot_to_tray = data.xpos[tray_body_id, :2] - data.xpos[robot_body_id, :2]
    if np.linalg.norm(robot_to_tray) < 1e-6:
        return SIDE_VIEW_AZIMUTH
    robot_to_tray = normalized_xy(robot_to_tray)
    robot_view_right = np.asarray([robot_to_tray[1], -robot_to_tray[0]])
    return camera_azimuth_from_eye_direction(robot_view_right)


def valid_camera_point(point: np.ndarray) -> bool:
    return bool(np.all(np.isfinite(point)) and -0.5 <= point[2] <= 3.0)


def named_body_position(
    model: mujoco.MjModel, data: mujoco.MjData, name: str
) -> np.ndarray | None:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
    if body_id < 0:
        return None
    point = data.xpos[body_id].copy()
    return point if valid_camera_point(point) else None


def print_scene_anchor_positions(model: mujoco.MjModel, data: mujoco.MjData) -> None:
    mujoco.mj_forward(model, data)
    entries = []
    for body_name in ("robot0_base", WOODEN_TRAY_BODY, CARDBOARD_BOX_BODY):
        point = named_body_position(model, data, body_name)
        if point is None:
            continue
        entries.append(
            f"{body_name}=({point[0]:+.3f}, {point[1]:+.3f}, {point[2]:+.3f})"
        )
    if entries:
        print("Loaded scene anchors: " + ", ".join(entries), flush=True)


def collect_named_body_points(
    model: mujoco.MjModel, data: mujoco.MjData, names: tuple[str, ...]
) -> list[np.ndarray]:
    points = []
    for name in names:
        point = named_body_position(model, data, name)
        if point is not None:
            points.append(point)
    return points


def collect_named_geom_points(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    name_hints: tuple[str, ...],
    *,
    require_visual_group: bool = False,
) -> list[np.ndarray]:
    points = []
    for geom_id in range(model.ngeom):
        if require_visual_group and int(model.geom_group[geom_id]) != VISUAL_GEOM_GROUP:
            continue
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        lowered = name.lower()
        if not any(hint in lowered for hint in name_hints):
            continue
        point = data.geom_xpos[geom_id].copy()
        if valid_camera_point(point):
            points.append(point)
    return points


def collect_glass_wall_points(
    model: mujoco.MjModel, data: mujoco.MjData
) -> list[np.ndarray]:
    points = []
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        lowered = name.lower()
        is_fragile_wall = (
            (lowered.startswith("tray_") and "wall" in lowered)
            or "fragile_wall" in lowered
            or "glass_wall" in lowered
        )
        if not is_fragile_wall:
            continue
        point = data.geom_xpos[geom_id].copy()
        if valid_camera_point(point):
            points.append(point)
    return points


def nearby_points(
    points: list[np.ndarray], references: list[np.ndarray], radius: float
) -> list[np.ndarray]:
    if not points or not references:
        return []
    reference_xy = np.asarray([point[:2] for point in references], dtype=np.float64)
    nearby = []
    for point in points:
        distances = np.linalg.norm(reference_xy - point[:2], axis=1)
        if bool(np.any(distances <= radius)):
            nearby.append(point)
    return nearby


def resolve_world_view_points(
    model: mujoco.MjModel, data: mujoco.MjData
) -> list[np.ndarray]:
    mujoco.mj_forward(model, data)
    robot_profile = resolve_robot_control_profile(model)
    robot_points = collect_named_body_points(
        model,
        data,
        (
            "mobilebase0_base",
            "robot0_base",
            "base",
            robot_profile.eef_body,
        ),
    )
    task_points = collect_named_body_points(
        model,
        data,
        (
            CARDBOARD_BOX_BODY,
            WOODEN_TRAY_BODY,
            "cereal",
            "cube",
            "object",
        ),
    )
    task_points.extend(
        collect_named_geom_points(
            model,
            data,
            ("tray_", "cardboard", "cereal", "object"),
            require_visual_group=False,
        )
    )
    workspace_points = collect_named_geom_points(
        model,
        data,
        ("counter", "island", "sink", "table", "stove", "tray"),
        require_visual_group=True,
    )
    references = task_points or robot_points
    focused_workspace = nearby_points(workspace_points, references, radius=1.75)
    points = [*robot_points, *task_points, *focused_workspace]
    if not points:
        points = [*robot_points, *nearby_points(workspace_points, robot_points, 2.5)]
    return points


def resolve_world_view_lookat(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    configured_lookat: list[float] | None = None,
) -> np.ndarray:
    if configured_lookat is not None:
        return np.asarray(configured_lookat, dtype=np.float64)
    mujoco.mj_forward(model, data)
    glass_wall_points = collect_glass_wall_points(model, data)
    if glass_wall_points:
        lookat = np.mean(np.asarray(glass_wall_points, dtype=np.float64), axis=0)
        lookat[2] += WORLD_VIEW_GLASS_WALL_Z_OFFSET
        return lookat

    points = resolve_world_view_points(model, data)
    if not points:
        return np.asarray(model.stat.center, dtype=np.float64)
    point_array = np.asarray(points, dtype=np.float64)
    lookat = np.mean(point_array, axis=0)
    lookat[2] = float(np.clip(np.median(point_array[:, 2]) + 0.15, 0.75, 1.2))
    return lookat


def resolve_world_view_azimuth(
    model: mujoco.MjModel, data: mujoco.MjData, lookat: np.ndarray
) -> float:
    if collect_glass_wall_points(model, data):
        return WORLD_VIEW_AZIMUTH

    robot_profile = resolve_robot_control_profile(model)
    for body_name in ("mobilebase0_base", "robot0_base", "base", robot_profile.eef_body):
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
        if body_id < 0:
            continue
        vector = data.xpos[body_id, :2] - lookat[:2]
        if np.linalg.norm(vector) > 1e-6:
            return float(
                np.degrees(np.arctan2(vector[1], vector[0]))
                + WORKSPACE_VIEW_REAR_SIDE_OFFSET
            )
    return WORKSPACE_VIEW_AZIMUTH


def resolve_world_view_distance(
    model: mujoco.MjModel, data: mujoco.MjData, lookat: np.ndarray
) -> float:
    if collect_glass_wall_points(model, data):
        return WORLD_VIEW_DISTANCE

    points = resolve_world_view_points(model, data)
    if not points:
        return WORKSPACE_VIEW_DISTANCE
    point_array = np.asarray(points, dtype=np.float64)
    radius = float(np.max(np.linalg.norm(point_array[:, :2] - lookat[:2], axis=1)))
    distance = 1.35 * radius + 1.0
    return float(
        np.clip(
            max(distance, WORLD_VIEW_DISTANCE),
            WORLD_VIEW_MIN_DISTANCE,
            WORLD_VIEW_MAX_DISTANCE,
        )
    )


def set_world_camera_defaults(
    cam,
    model: mujoco.MjModel,
    data: mujoco.MjData,
) -> None:
    lookat = resolve_world_view_lookat(model, data)
    azimuth = resolve_world_view_azimuth(model, data, lookat)
    distance = resolve_world_view_distance(model, data, lookat)
    set_free_camera_defaults(
        cam,
        lookat=lookat,
        distance=distance,
        azimuth=azimuth,
        elevation=(
            WORLD_VIEW_ELEVATION
            if collect_glass_wall_points(model, data)
            else WORKSPACE_VIEW_ELEVATION
        ),
    )


def free_camera_eye_position(cam) -> np.ndarray:
    azimuth = np.deg2rad(float(cam.azimuth))
    elevation = np.deg2rad(float(cam.elevation))
    distance = float(cam.distance)
    offset = distance * np.array(
        [
            -np.cos(elevation) * np.sin(azimuth),
            np.cos(elevation) * np.cos(azimuth),
            -np.sin(elevation),
        ],
        dtype=np.float64,
    )
    return np.asarray(cam.lookat, dtype=np.float64) + offset


def print_view_camera_debug(model: mujoco.MjModel, data: mujoco.MjData, cam) -> None:
    if cam.type == mujoco.mjtCamera.mjCAMERA_FIXED:
        camera_id = int(cam.fixedcamid)
        camera_name = mujoco.mj_id2name(
            model, mujoco.mjtObj.mjOBJ_CAMERA, camera_id
        )
        mujoco.mj_forward(model, data)
        print(
            "Camera fixed: "
            f"name={camera_name or camera_id} "
            f"pos={np.array2string(data.cam_xpos[camera_id], precision=4)} "
            f"xmat={np.array2string(data.cam_xmat[camera_id].reshape(3, 3), precision=4)}",
            flush=True,
        )
        return

    print(
        "Camera free: "
        f"eye={np.array2string(free_camera_eye_position(cam), precision=4)} "
        f"lookat={np.array2string(np.asarray(cam.lookat), precision=4)} "
        f"distance={float(cam.distance):.4f} "
        f"azimuth={float(cam.azimuth):.4f} "
        f"elevation={float(cam.elevation):.4f}",
        flush=True,
    )


def resolve_recording_camera(model: mujoco.MjModel, view_name: str) -> int | mujoco.MjvCamera:
    view_key = view_name.lower().replace("-", "_")
    if view_key == "wrist":
        profile = resolve_robot_control_profile(model)
        camera_name = profile.wrist_camera_name or WRIST_CAMERA_NAME
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id < 0:
            raise ValueError(f"Missing camera: {camera_name}")
        return camera_id
    if view_key == "left_side":
        return make_free_camera(azimuth=LEFT_SIDE_VIEW_AZIMUTH)
    if view_key == "right_side":
        return make_free_camera(azimuth=SIDE_VIEW_AZIMUTH)
    camera_aliases = {
        "agentview_left": ("agentview_left", "robot0_agentview_left"),
        "agentview_right": ("agentview_right", "robot0_agentview_right"),
        "robot0_agentview_left": ("robot0_agentview_left", "agentview_left"),
        "robot0_agentview_right": ("robot0_agentview_right", "agentview_right"),
        "robot0_eye_in_hand": ("robot0_eye_in_hand", "eye_in_hand", WRIST_CAMERA_NAME),
    }
    names_to_try = camera_aliases.get(view_key, (view_name,))
    camera_id = -1
    for camera_name in names_to_try:
        camera_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_CAMERA, camera_name)
        if camera_id >= 0:
            break
    if camera_id < 0:
        raise ValueError(f"Missing camera or recording view: {view_name}")
    return camera_id


def view_video_path(base_path: Path, view_name: str) -> Path:
    safe_name = view_name.lower().replace("-", "_")
    return base_path.with_name(f"{base_path.name}_{safe_name}").with_suffix(".mp4")
