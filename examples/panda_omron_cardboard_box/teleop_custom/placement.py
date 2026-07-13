from __future__ import annotations

from .common import (
    CARDBOARD_BOX_BODY,
    CARDBOARD_BOX_FREEJOINT,
    OBJECT_SUPPORT_CLEARANCE,
    RANDOMIZE_BOX_DEFAULT_REGION,
    RANDOMIZE_BOX_MAX_ATTEMPTS,
    RANDOMIZE_BOX_WALL_MARGIN,
    WOODEN_TRAY_BODY,
    mujoco,
    np,
)

def body_descendant_ids(model: mujoco.MjModel, root_body_id: int) -> set[int]:
    descendants = {root_body_id}
    for body_id in range(model.nbody):
        current = body_id
        while current > 0:
            if current == root_body_id:
                descendants.add(body_id)
                break
            current = int(model.body_parentid[current])
    return descendants


def geom_world_bounds(
    model: mujoco.MjModel, data: mujoco.MjData, geom_id: int
) -> tuple[np.ndarray, np.ndarray]:
    center = data.geom_xpos[geom_id]
    geom_type = model.geom_type[geom_id]
    if geom_type == mujoco.mjtGeom.mjGEOM_BOX:
        rot = data.geom_xmat[geom_id].reshape(3, 3)
        radius = np.abs(rot) @ model.geom_size[geom_id, :3]
    elif geom_type == mujoco.mjtGeom.mjGEOM_MESH:
        mesh_id = int(model.geom_dataid[geom_id])
        vert_adr = int(model.mesh_vertadr[mesh_id])
        vert_num = int(model.mesh_vertnum[mesh_id])
        vertices = model.mesh_vert[vert_adr : vert_adr + vert_num]
        rot = data.geom_xmat[geom_id].reshape(3, 3)
        world_vertices = center + vertices @ rot.T
        return world_vertices.min(axis=0), world_vertices.max(axis=0)
    else:
        radius = np.full(3, float(model.geom_rbound[geom_id]), dtype=np.float64)
    return center - radius, center + radius


def multiply_mj_quat(left: np.ndarray, right: np.ndarray) -> np.ndarray:
    left = np.asarray(left, dtype=np.float64)
    right = np.asarray(right, dtype=np.float64)
    lw, lx, ly, lz = left
    rw, rx, ry, rz = right
    quat = np.array(
        [
            lw * rw - lx * rx - ly * ry - lz * rz,
            lw * rx + lx * rw + ly * rz - lz * ry,
            lw * ry - lx * rz + ly * rw + lz * rx,
            lw * rz + lx * ry - ly * rx + lz * rw,
        ],
        dtype=np.float64,
    )
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.array([1.0, 0.0, 0.0, 0.0], dtype=np.float64)
    return quat / norm


def reset_freejoint_to_model_qpos0(
    model: mujoco.MjModel, data: mujoco.MjData, joint_name: str
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        return
    qpos_adr = model.jnt_qposadr[joint_id]
    qvel_adr = model.jnt_dofadr[joint_id]
    data.qpos[qpos_adr : qpos_adr + 7] = model.qpos0[qpos_adr : qpos_adr + 7]
    data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    mujoco.mj_forward(model, data)


def body_geom_ids(
    model: mujoco.MjModel,
    body_name: str,
    geom_name_prefix: str | None = None,
    geom_name_contains: str | None = None,
    *,
    missing_body_ok: bool = False,
    require_geoms: bool = False,
) -> list[int]:
    body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, body_name)
    if body_id < 0:
        if missing_body_ok:
            return []
        raise ValueError(f"Missing body: {body_name}")
    descendants = body_descendant_ids(model, body_id)
    geom_ids = []
    for geom_id in range(model.ngeom):
        if int(model.geom_bodyid[geom_id]) not in descendants:
            continue
        geom_name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if geom_name_prefix is not None and not geom_name.startswith(geom_name_prefix):
            continue
        if geom_name_contains is not None and geom_name_contains not in geom_name:
            continue
        geom_ids.append(geom_id)
    if require_geoms and not geom_ids:
        suffix = ""
        if geom_name_prefix is not None:
            suffix += f" starting with '{geom_name_prefix}'"
        if geom_name_contains is not None:
            suffix += f" containing '{geom_name_contains}'"
        raise ValueError(f"No geoms found for body {body_name}{suffix}")
    return geom_ids


def align_freejoint_body_bottom_to_support(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    body_name: str = CARDBOARD_BOX_BODY,
    joint_name: str = CARDBOARD_BOX_FREEJOINT,
    support_geom_name: str = "tray_bottom_panel",
    clearance: float = OBJECT_SUPPORT_CLEARANCE,
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    support_geom_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_GEOM, support_geom_name
    )
    if joint_id < 0 or support_geom_id < 0:
        return

    mujoco.mj_forward(model, data)
    geom_ids = body_geom_ids(model, body_name, geom_name_prefix="cardboard_")
    if not geom_ids:
        return
    body_min_z = min(
        geom_world_bounds(model, data, geom_id)[0][2] for geom_id in geom_ids
    )
    support_max_z = geom_world_bounds(model, data, support_geom_id)[1][2]
    qpos_adr = model.jnt_qposadr[joint_id]
    data.qpos[qpos_adr + 2] += float(support_max_z - body_min_z + clearance)
    mujoco.mj_forward(model, data)


def randomize_red_box_xy_in_tray(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    rng: np.random.Generator,
    margin: float = RANDOMIZE_BOX_WALL_MARGIN,
    randomize_yaw: bool = True,
    region: str | None = RANDOMIZE_BOX_DEFAULT_REGION,
) -> np.ndarray:
    mujoco.mj_forward(model, data)
    tray_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, WOODEN_TRAY_BODY)
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, CARDBOARD_BOX_BODY)
    box_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, CARDBOARD_BOX_FREEJOINT
    )
    if tray_body_id < 0 or box_body_id < 0 or box_joint_id < 0:
        raise ValueError("Missing tray body, cardboard box body, or box freejoint.")

    qpos_adr = model.jnt_qposadr[box_joint_id]
    qvel_adr = model.jnt_dofadr[box_joint_id]
    if randomize_yaw:
        yaw = float(rng.uniform(-np.pi, np.pi))
        yaw_quat = np.array(
            [np.cos(0.5 * yaw), 0.0, 0.0, np.sin(0.5 * yaw)],
            dtype=np.float64,
        )
        data.qpos[qpos_adr + 3 : qpos_adr + 7] = yaw_quat
        data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        align_freejoint_body_bottom_to_support(model, data)
        mujoco.mj_forward(model, data)

    tray_center = data.xpos[tray_body_id]
    inner_x_min = -np.inf
    inner_x_max = np.inf
    inner_y_min = -np.inf
    inner_y_max = np.inf
    for geom_id in range(model.ngeom):
        name = mujoco.mj_id2name(model, mujoco.mjtObj.mjOBJ_GEOM, geom_id) or ""
        if not name.startswith("tray_") or "wall" not in name:
            continue
        center = data.geom_xpos[geom_id]
        size = model.geom_size[geom_id]
        if size[0] < size[1]:
            if center[0] >= tray_center[0]:
                inner_x_max = min(inner_x_max, float(center[0] - size[0]))
            else:
                inner_x_min = max(inner_x_min, float(center[0] + size[0]))
        else:
            if center[1] >= tray_center[1]:
                inner_y_max = min(inner_y_max, float(center[1] - size[1]))
            else:
                inner_y_min = max(inner_y_min, float(center[1] + size[1]))

    if not np.all(np.isfinite([inner_x_min, inner_x_max, inner_y_min, inner_y_max])):
        raise ValueError("Could not resolve tray wall interior bounds.")

    box_center = data.xpos[box_body_id]
    box_x_min = np.inf
    box_x_max = -np.inf
    box_y_min = np.inf
    box_y_max = -np.inf
    for geom_id in body_geom_ids(
        model, CARDBOARD_BOX_BODY, geom_name_prefix="cardboard_"
    ):
        lower, upper = geom_world_bounds(model, data, geom_id)
        box_x_min = min(box_x_min, float(lower[0] - box_center[0]))
        box_x_max = max(box_x_max, float(upper[0] - box_center[0]))
        box_y_min = min(box_y_min, float(lower[1] - box_center[1]))
        box_y_max = max(box_y_max, float(upper[1] - box_center[1]))

    if not np.all(np.isfinite([box_x_min, box_x_max, box_y_min, box_y_max])):
        raise ValueError("Could not resolve task object bounds.")

    x_low = inner_x_min - box_x_min + margin
    x_high = inner_x_max - box_x_max - margin
    y_low = inner_y_min - box_y_min + margin
    y_high = inner_y_max - box_y_max - margin
    if x_low > x_high or y_low > y_high:
        raise ValueError("Cardboard box does not fit inside tray wall bounds.")

    if region is not None:
        region_key = region.lower().replace("-", "_")
        if region_key == "front_right":
            robot_body_id = mujoco.mj_name2id(
                model, mujoco.mjtObj.mjOBJ_BODY, "robot0_base"
            )
            if robot_body_id < 0:
                raise ValueError("Missing robot0_base body for robot-facing region.")
            robot_to_tray = tray_center[:2] - data.xpos[robot_body_id, :2]
            robot_to_tray = robot_to_tray / np.linalg.norm(robot_to_tray)
            robot_view_right = np.asarray(
                [robot_to_tray[1], -robot_to_tray[0]], dtype=np.float64
            )
        elif region_key in {"full", "all"}:
            robot_to_tray = None
            robot_view_right = None
        else:
            raise ValueError(f"Unsupported randomize region: {region}")
    else:
        region_key = None
        robot_to_tray = None
        robot_view_right = None

    box_geom_ids = body_geom_ids(
        model, CARDBOARD_BOX_BODY, geom_name_prefix="cardboard_"
    )
    for _ in range(RANDOMIZE_BOX_MAX_ATTEMPTS):
        new_xy = np.array(
            [rng.uniform(x_low, x_high), rng.uniform(y_low, y_high)],
            dtype=np.float64,
        )
        data.qpos[qpos_adr : qpos_adr + 2] = new_xy
        data.qvel[qvel_adr : qvel_adr + 6] = 0.0
        align_freejoint_body_bottom_to_support(model, data)
        mujoco.mj_forward(model, data)

        bounds = [geom_world_bounds(model, data, geom_id) for geom_id in box_geom_ids]
        lower = np.min(np.asarray([bound[0] for bound in bounds]), axis=0)
        upper = np.max(np.asarray([bound[1] for bound in bounds]), axis=0)
        inside_walls = (
            lower[0] >= inner_x_min + margin
            and upper[0] <= inner_x_max - margin
            and lower[1] >= inner_y_min + margin
            and upper[1] <= inner_y_max - margin
        )
        if not inside_walls:
            continue
        if region_key == "front_right":
            rel_xy = data.xpos[box_body_id, :2] - tray_center[:2]
            if (
                float(np.dot(rel_xy, robot_view_right)) < 0.0
                or float(np.dot(rel_xy, robot_to_tray)) < 0.0
            ):
                continue
        if region_key in {None, "front_right", "full", "all"}:
            return data.xpos[box_body_id, :2].copy()

    raise ValueError("Could not sample cardboard_box inside the requested region.")


def freejoint_addresses(model: mujoco.MjModel, joint_name: str) -> tuple[int, int]:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
    if joint_id < 0:
        raise ValueError(f"Missing freejoint: {joint_name}")
    return int(model.jnt_qposadr[joint_id]), int(model.jnt_dofadr[joint_id])


def set_freejoint(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    joint_name: str,
    qpos: np.ndarray,
    qvel: np.ndarray | None = None,
) -> None:
    qpos_adr, qvel_adr = freejoint_addresses(model, joint_name)
    data.qpos[qpos_adr : qpos_adr + 7] = qpos
    if qvel is None:
        data.qvel[qvel_adr : qvel_adr + 6] = 0.0
    else:
        data.qvel[qvel_adr : qvel_adr + 6] = qvel

