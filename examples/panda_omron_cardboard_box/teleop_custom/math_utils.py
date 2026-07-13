from __future__ import annotations

from .common import Any, mujoco, np

def rotation_matrix(axis: np.ndarray, angle: float) -> np.ndarray:
    axis = axis / np.linalg.norm(axis)
    x, y, z = axis
    c = np.cos(angle)
    s = np.sin(angle)
    one_c = 1.0 - c
    return np.array(
        [
            [c + x * x * one_c, x * y * one_c - z * s, x * z * one_c + y * s],
            [y * x * one_c + z * s, c + y * y * one_c, y * z * one_c - x * s],
            [z * x * one_c - y * s, z * y * one_c + x * s, c + z * z * one_c],
        ],
        dtype=np.float64,
    )


def rotation_error(target: np.ndarray, current: np.ndarray) -> np.ndarray:
    error_mat = target @ current.T
    cos_angle = np.clip((np.trace(error_mat) - 1.0) * 0.5, -1.0, 1.0)
    angle = float(np.arccos(cos_angle))
    if angle < 1e-8:
        return np.zeros(3)
    axis = np.array(
        [
            error_mat[2, 1] - error_mat[1, 2],
            error_mat[0, 2] - error_mat[2, 0],
            error_mat[1, 0] - error_mat[0, 1],
        ],
        dtype=np.float64,
    )
    axis /= 2.0 * np.sin(angle)
    return axis * angle


def unit_or_zero(vector: np.ndarray) -> np.ndarray:
    norm = float(np.linalg.norm(vector))
    if norm < 1e-12:
        return np.zeros_like(vector)
    return vector / norm


def format_vector(vector: np.ndarray, precision: int = 4) -> str:
    return np.array2string(np.asarray(vector), precision=precision, suppress_small=True)


def print_keyboard_control_debug(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    controller: Any,
    previous_target_pos: np.ndarray,
    previous_target_rot: np.ndarray,
) -> None:
    mujoco.mj_forward(model, data)
    eef_pos = data.xpos[controller.ee_body_id].copy()
    target_delta = controller.target_pos - previous_target_pos
    target_error = controller.target_pos - eef_pos
    rotation_delta = rotation_error(controller.target_rot, previous_target_rot)
    print(
        "[keyboard-control] "
        f"mode={'rotation' if controller.rotate_mode else 'translation'} "
        f"map={'swapped' if controller.keyboard_xy_swapped else 'normal'} "
        f"eef={format_vector(eef_pos)} "
        f"target={format_vector(controller.target_pos)} "
        f"err={np.linalg.norm(target_error):.4f} "
        f"dpos={format_vector(target_delta)} "
        f"dir={format_vector(unit_or_zero(target_delta), precision=3)} "
        f"drot={format_vector(rotation_delta)}",
        flush=True,
    )


def rotation_angle(rotation: np.ndarray) -> float:
    cos_angle = np.clip((np.trace(rotation) - 1.0) * 0.5, -1.0, 1.0)
    return float(np.arccos(cos_angle))

def quat_xyzw_to_matrix(quat: np.ndarray) -> np.ndarray:
    quat = np.asarray(quat, dtype=np.float64)
    norm = np.linalg.norm(quat)
    if norm <= 0.0:
        return np.eye(3)
    x, y, z, w = quat / norm
    xx, yy, zz = x * x, y * y, z * z
    xy, xz, yz = x * y, x * z, y * z
    wx, wy, wz = w * x, w * y, w * z
    return np.array(
        [
            [1.0 - 2.0 * (yy + zz), 2.0 * (xy - wz), 2.0 * (xz + wy)],
            [2.0 * (xy + wz), 1.0 - 2.0 * (xx + zz), 2.0 * (yz - wx)],
            [2.0 * (xz - wy), 2.0 * (yz + wx), 1.0 - 2.0 * (xx + yy)],
        ],
        dtype=np.float64,
    )


def pose_json_to_matrix(
    pose_json: dict[str, Any],
    translation_map: str = "xzy",
    rotation_map: str = "quat_axes",
    rotation_axis_map: str | None = None,
) -> np.ndarray:
    """Convert a T265 pose into the teleop control frame."""
    rotation = pose_json["rotation"]
    translation = pose_json["translation"]
    transform = np.eye(4, dtype=np.float64)
    if translation_map == "legacy" or rotation_map == "legacy":
        transform[:3, :3] = quat_xyzw_to_matrix(
            np.array(
                [-rotation["z"], rotation["y"], rotation["x"], rotation["w"]],
                dtype=np.float64,
            )
        )
    if translation_map == "legacy":
        transform[:3, 3] = np.array(
            [-translation["z"], translation["y"], translation["x"]],
            dtype=np.float64,
        )
        return transform

    axis_maps = {
        "xyz": (0, 1, 2),
        "xzy": (0, 2, 1),
        "zxy": (2, 0, 1),
        "zyx": (2, 1, 0),
        "swap_xz": (2, 1, 0),
        "yzx": (1, 2, 0),
    }
    if translation_map not in axis_maps:
        raise ValueError(f"Unsupported T265 translation map: {translation_map}")
    axis_order = axis_maps[translation_map]
    rotation_axis_map = (
        translation_map if rotation_axis_map is None else rotation_axis_map
    )
    if rotation_axis_map not in axis_maps:
        raise ValueError(f"Unsupported T265 rotation axis map: {rotation_axis_map}")
    rotation_axis_order = axis_maps[rotation_axis_map]
    if rotation_map == "quat_axes":
        raw_quat = np.array(
            [rotation["x"], rotation["y"], rotation["z"]],
            dtype=np.float64,
        )
        transform[:3, :3] = quat_xyzw_to_matrix(
            np.array(
                [
                    raw_quat[rotation_axis_order[0]],
                    raw_quat[rotation_axis_order[1]],
                    raw_quat[rotation_axis_order[2]],
                    rotation["w"],
                ],
                dtype=np.float64,
            )
        )
    elif rotation_map == "mapped":
        permutation = np.eye(3, dtype=np.float64)[list(rotation_axis_order)]
        raw_rotation = quat_xyzw_to_matrix(
            np.array(
                [rotation["x"], rotation["y"], rotation["z"], rotation["w"]],
                dtype=np.float64,
            )
        )
        transform[:3, :3] = permutation @ raw_rotation @ permutation.T
    elif rotation_map != "legacy":
        raise ValueError(f"Unsupported T265 rotation map: {rotation_map}")
    raw_translation = np.array(
        [translation["x"], translation["y"], translation["z"]],
        dtype=np.float64,
    )
    transform[:3, 3] = raw_translation[list(axis_order)]
    return transform


def invert_transform(transform: np.ndarray) -> np.ndarray:
    inverse = np.eye(4, dtype=np.float64)
    inverse[:3, :3] = transform[:3, :3].T
    inverse[:3, 3] = -inverse[:3, :3] @ transform[:3, 3]
    return inverse
