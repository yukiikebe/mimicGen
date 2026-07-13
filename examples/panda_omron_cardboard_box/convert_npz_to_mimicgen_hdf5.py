from __future__ import annotations

import argparse
import datetime
import glob
import json
from pathlib import Path
import time

import h5py
import numpy as np


REQUIRED_DATAGEN_KEYS = (
    "datagen_eef_pose",
    "datagen_target_pose",
    "datagen_object_poses",
    "datagen_object_pose_names",
    "datagen_subtask_term_signals",
    "datagen_subtask_signal_names",
    "datagen_gripper_action",
)


def np_scalar_to_python(value):
    array = np.asarray(value)
    if array.shape == ():
        item = array.item()
        if isinstance(item, bytes):
            return item.decode("utf-8")
        return item
    return value


def np_string_array_to_list(value) -> list[str]:
    return [
        item.decode("utf-8") if isinstance(item, bytes) else str(item)
        for item in np.asarray(value).tolist()
    ]


def expand_input_paths(patterns: list[str]) -> list[Path]:
    paths: list[Path] = []
    for pattern in patterns:
        matches = glob.glob(str(Path(pattern).expanduser()))
        if matches:
            for match in matches:
                match_path = Path(match).resolve()
                if match_path.is_dir():
                    paths.extend(sorted(match_path.glob("*.npz")))
                else:
                    paths.append(match_path)
        else:
            path = Path(pattern).expanduser().resolve()
            if path.is_dir():
                paths.extend(sorted(path.glob("*.npz")))
            else:
                paths.append(path)
    unique_paths = sorted(dict.fromkeys(paths))
    missing = [path for path in unique_paths if not path.exists()]
    if missing:
        raise FileNotFoundError(
            "Missing input recording(s): " + ", ".join(str(path) for path in missing)
        )
    if not unique_paths:
        raise FileNotFoundError(
            "No input recordings matched. If passing a directory, it must contain "
            "*.npz files."
        )
    return unique_paths


def validate_npz(recording: np.lib.npyio.NpzFile, path: Path) -> None:
    missing = [key for key in REQUIRED_DATAGEN_KEYS if key not in recording]
    if missing:
        raise ValueError(
            f"{path} is missing MimicGen datagen keys: {missing}. "
            "Collect a new recording with teleop_custom_scene_t265.py after the "
            "MimicGen metadata update."
        )


def get_actions(recording: np.lib.npyio.NpzFile) -> np.ndarray:
    if "policy_action" in recording:
        return np.asarray(recording["policy_action"])
    if "ctrl" in recording:
        return np.asarray(recording["ctrl"])
    raise ValueError("Recording has neither policy_action nor ctrl.")


def get_states(recording: np.lib.npyio.NpzFile) -> np.ndarray:
    if "states" in recording:
        return np.asarray(recording["states"])
    if "qpos" in recording and "qvel" in recording:
        return np.concatenate(
            [np.asarray(recording["qpos"]), np.asarray(recording["qvel"])], axis=1
        )
    if "qpos" in recording:
        return np.asarray(recording["qpos"])
    raise ValueError("Recording has neither states nor qpos/qvel.")


def step_signal(horizon: int, step: int) -> np.ndarray:
    signal = np.zeros(horizon, dtype=np.int32)
    signal[step:] = 1
    return signal


def first_true_step(mask: np.ndarray, name: str, path: Path) -> int:
    indices = np.flatnonzero(mask)
    if len(indices) == 0:
        raise ValueError(f"{path} has no valid {name} transition.")
    return int(indices[0])


def get_cardboard_box_poses(recording: np.lib.npyio.NpzFile, path: Path) -> np.ndarray:
    object_names = np_string_array_to_list(recording["datagen_object_pose_names"])
    if "cardboard_box" not in object_names:
        raise ValueError(f"{path} has no cardboard_box object pose.")
    box_index = object_names.index("cardboard_box")
    return np.asarray(recording["datagen_object_poses"])[:, box_index]


def box_lifted_step(
    recording: np.lib.npyio.NpzFile,
    path: Path,
    lift_threshold: float,
) -> int:
    box_poses = get_cardboard_box_poses(recording, path)
    box_z = box_poses[:, 2, 3]
    return first_true_step(
        box_z > float(box_z[0] + lift_threshold),
        "box_lifted",
        path,
    )


def box_stood_step(
    recording: np.lib.npyio.NpzFile,
    path: Path,
    stand_axis_threshold: float,
) -> int:
    box_poses = get_cardboard_box_poses(recording, path)
    box_rot = box_poses[:, :3, :3]
    vertical_axis_score = np.maximum(
        np.abs(box_rot[:, 2, 0]),
        np.abs(box_rot[:, 2, 1]),
    )
    return first_true_step(
        vertical_axis_score >= float(stand_axis_threshold),
        "box_stood",
        path,
    )


def recorded_close_step(
    recording: np.lib.npyio.NpzFile,
    path: Path,
    fallback_step: int | None = None,
) -> int:
    gripper_action = np.asarray(recording["datagen_gripper_action"])
    close_steps = np.flatnonzero(
        np.linalg.norm(gripper_action - gripper_action[0], axis=1) > 1e-4
    )
    if len(close_steps) == 0:
        if fallback_step is not None:
            return int(fallback_step)
        raise ValueError(f"{path} has no recorded gripper close transition.")
    return int(close_steps[0])


def grasp_ready_step(
    recording: np.lib.npyio.NpzFile,
    path: Path,
    pre_close_steps: int,
) -> int:
    close_step = recorded_close_step(recording, path)
    return max(1, close_step - max(0, int(pre_close_steps)))


def get_subtask_signals(
    recording: np.lib.npyio.NpzFile,
    path: Path,
    scheme: str,
    lift_threshold: float,
    stand_axis_threshold: float,
    grasp_ready_pre_close_steps: int,
) -> tuple[list[str], np.ndarray]:
    signal_names = np_string_array_to_list(recording["datagen_subtask_signal_names"])
    signals = np.asarray(recording["datagen_subtask_term_signals"], dtype=np.int32)
    if scheme == "recorded":
        return signal_names, signals

    if scheme not in (
        "contact_lift",
        "contact_stand_lift",
        "contact_stand_grasp_lift",
    ):
        raise ValueError(f"Unsupported subtask signal scheme: {scheme}")

    horizon = int(signals.shape[0])
    if "contact_count" not in recording:
        raise ValueError(f"{path} is missing contact_count for {scheme} scheme.")
    contact_step = first_true_step(
        np.asarray(recording["contact_count"]) > 0,
        "contact",
        path,
    )
    lift_step = box_lifted_step(recording, path, lift_threshold)

    synthesized_names = ["gripper_contact", "box_lifted"]
    synthesized_signals = [
        step_signal(horizon, contact_step),
        step_signal(horizon, lift_step),
    ]

    if scheme in ("contact_stand_lift", "contact_stand_grasp_lift"):
        stood_step = box_stood_step(recording, path, stand_axis_threshold)
        if not 0 < contact_step < stood_step < horizon:
            raise ValueError(
                f"{path} produced invalid {scheme} boundaries: "
                f"contact={contact_step}, stood={stood_step}, horizon={horizon}"
            )
        synthesized_names.insert(1, "box_stood")
        synthesized_signals.insert(1, step_signal(horizon, stood_step))

        if scheme == "contact_stand_grasp_lift":
            grasp_step = grasp_ready_step(
                recording,
                path,
                grasp_ready_pre_close_steps,
            )
            if not stood_step < grasp_step < horizon:
                raise ValueError(
                    f"{path} produced invalid contact_stand_grasp_lift boundaries: "
                    f"stood={stood_step}, grasp_ready={grasp_step}, horizon={horizon}"
                )
            synthesized_names.insert(2, "grasp_ready")
            synthesized_signals.insert(2, step_signal(horizon, grasp_step))
    elif not 0 < contact_step < lift_step < horizon:
        raise ValueError(
            f"{path} produced invalid contact_lift boundaries: "
            f"contact={contact_step}, lift={lift_step}, horizon={horizon}"
        )

    for signal_index, signal_name in enumerate(signal_names):
        if signal_name not in synthesized_names:
            synthesized_names.append(signal_name)
            synthesized_signals.append(signals[:, signal_index].astype(np.int32))
    return synthesized_names, np.stack(synthesized_signals, axis=1)


def get_gripper_action(
    recording: np.lib.npyio.NpzFile,
    path: Path,
    scheme: str,
    lift_threshold: float,
    stand_axis_threshold: float,
    grasp_ready_pre_close_steps: int,
) -> np.ndarray:
    gripper_action = np.asarray(recording["datagen_gripper_action"]).copy()
    if scheme not in (
        "contact_lift",
        "contact_stand_lift",
        "contact_stand_grasp_lift",
    ):
        return gripper_action

    if scheme == "contact_stand_grasp_lift":
        boundary_step = grasp_ready_step(
            recording,
            path,
            grasp_ready_pre_close_steps,
        )
    elif scheme == "contact_stand_lift":
        boundary_step = box_stood_step(recording, path, stand_axis_threshold)
    else:
        boundary_step = box_lifted_step(recording, path, lift_threshold)
    close_step = max(
        boundary_step,
        recorded_close_step(recording, path, fallback_step=boundary_step),
    )
    gripper_action[:close_step] = gripper_action[0]
    gripper_action[close_step:] = gripper_action[-1]
    return gripper_action


def write_metadata_attrs(
    demo_group: h5py.Group,
    recording: np.lib.npyio.NpzFile,
    source_path: Path,
) -> None:
    metadata_group = demo_group.create_group("metadata")
    metadata_group.attrs["source_npz"] = str(source_path)
    for key in (
        "task_name",
        "language_prompt",
        "goal_definition_json",
        "task_metadata_json",
        "model_xml",
        "video_path",
        "episode_success",
        "episode_goal_satisfied_step",
        "mimicgen_env_interface_name",
        "mimicgen_env_interface_type",
    ):
        if key in recording:
            metadata_group.attrs[key] = np_scalar_to_python(recording[key])


def write_demo(
    data_group: h5py.Group,
    demo_name: str,
    recording_path: Path,
    env_interface_name: str,
    env_interface_type: str,
    compression: str | None,
    subtask_scheme: str,
    lift_threshold: float,
    stand_axis_threshold: float,
    grasp_ready_pre_close_steps: int,
) -> int:
    recording = np.load(recording_path, allow_pickle=False)
    validate_npz(recording, recording_path)

    actions = get_actions(recording)
    states = get_states(recording)
    horizon = int(actions.shape[0])
    if states.shape[0] != horizon:
        raise ValueError(
            f"{recording_path} has actions length {horizon}, "
            f"but states length {states.shape[0]}"
        )

    demo_group = data_group.create_group(demo_name)
    demo_group.create_dataset("actions", data=actions, compression=compression)
    demo_group.create_dataset("states", data=states, compression=compression)
    if "qpos" in recording:
        demo_group.create_dataset(
            "qpos", data=recording["qpos"], compression=compression
        )
    if "qvel" in recording:
        demo_group.create_dataset(
            "qvel", data=recording["qvel"], compression=compression
        )
    if "ctrl" in recording:
        demo_group.create_dataset(
            "ctrl", data=recording["ctrl"], compression=compression
        )

    datagen_group = demo_group.create_group("datagen_info")
    datagen_group.create_dataset(
        "eef_pose", data=recording["datagen_eef_pose"], compression=compression
    )
    datagen_group.create_dataset(
        "target_pose", data=recording["datagen_target_pose"], compression=compression
    )
    datagen_group.create_dataset(
        "gripper_action",
        data=get_gripper_action(
            recording,
            recording_path,
            subtask_scheme,
            lift_threshold,
            stand_axis_threshold,
            grasp_ready_pre_close_steps,
        ),
        compression=compression,
    )

    object_names = np_string_array_to_list(recording["datagen_object_pose_names"])
    object_poses = np.asarray(recording["datagen_object_poses"])
    object_group = datagen_group.create_group("object_poses")
    for object_index, object_name in enumerate(object_names):
        object_group.create_dataset(
            object_name,
            data=object_poses[:, object_index],
            compression=compression,
        )

    signal_names, signals = get_subtask_signals(
        recording,
        recording_path,
        subtask_scheme,
        lift_threshold,
        stand_axis_threshold,
        grasp_ready_pre_close_steps,
    )
    signal_group = datagen_group.create_group("subtask_term_signals")
    for signal_index, signal_name in enumerate(signal_names):
        signal_group.create_dataset(
            signal_name,
            data=signals[:, signal_index],
            compression=compression,
        )

    datagen_group.attrs["env_interface_name"] = env_interface_name
    datagen_group.attrs["env_interface_type"] = env_interface_type
    demo_group.attrs["num_samples"] = horizon
    if "model_xml" in recording:
        demo_group.attrs["model_file"] = np_scalar_to_python(recording["model_xml"])
    write_metadata_attrs(demo_group, recording, recording_path)
    return horizon


def make_env_args(
    args: argparse.Namespace, first_recording: np.lib.npyio.NpzFile
) -> str:
    env_args = {
        "env_name": args.env_name,
        "type": args.env_type,
        "env_kwargs": {
            "model_xml": np_scalar_to_python(first_recording["model_xml"])
            if "model_xml" in first_recording
            else "",
            "source_format": "mujoco_custom_scene_npz",
            "task_name": np_scalar_to_python(first_recording["task_name"])
            if "task_name" in first_recording
            else args.env_name,
        },
    }
    return json.dumps(env_args, indent=4)


def convert_npz_to_hdf5(args: argparse.Namespace) -> Path:
    input_paths = expand_input_paths(args.inputs)
    output_path = Path(args.output).expanduser().resolve()
    output_path.parent.mkdir(parents=True, exist_ok=True)
    if output_path.exists() and not args.overwrite:
        raise FileExistsError(
            f"Output exists: {output_path}. Pass --overwrite to replace."
        )

    timestamp = time.time()
    readable_timestamp = datetime.datetime.fromtimestamp(timestamp).strftime(
        "date_%m_%d_%Y_time_%H_%M_%S"
    )

    total_samples = 0
    with h5py.File(output_path, "w") as h5:
        data_group = h5.create_group("data")
        data_group.attrs["timestamp"] = timestamp
        data_group.attrs["readable_timestamp"] = readable_timestamp

        success_mask = []
        kept_demo_names = []
        for demo_index, input_path in enumerate(input_paths):
            recording = np.load(input_path, allow_pickle=False)
            episode_success = (
                bool(recording["episode_success"])
                if "episode_success" in recording
                else True
            )
            if args.only_success and not episode_success:
                continue

            demo_name = f"demo_{len(kept_demo_names)}"
            total_samples += write_demo(
                data_group=data_group,
                demo_name=demo_name,
                recording_path=input_path,
                env_interface_name=args.env_interface_name,
                env_interface_type=args.env_interface_type,
                compression=None if args.no_compression else "gzip",
                subtask_scheme=args.subtask_scheme,
                lift_threshold=args.lift_threshold,
                stand_axis_threshold=args.stand_axis_threshold,
                grasp_ready_pre_close_steps=args.grasp_ready_pre_close_steps,
            )
            kept_demo_names.append(demo_name)
            success_mask.append(demo_name.encode("utf-8"))

        if not kept_demo_names:
            raise ValueError("No recordings were converted.")

        with np.load(input_paths[0], allow_pickle=False) as first_recording:
            data_group.attrs["env_args"] = make_env_args(args, first_recording)
        data_group.attrs["total"] = total_samples

        mask_group = h5.create_group("mask")
        mask_group.create_dataset("all", data=np.asarray(success_mask, dtype="S"))
        if args.only_success:
            mask_group.create_dataset(
                "successful", data=np.asarray(success_mask, dtype="S")
            )

    print(
        f"Wrote {len(kept_demo_names)} demos / {total_samples} samples: {output_path}"
    )
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Convert custom MuJoCo teleop .npz recordings to MimicGen source HDF5."
    )
    parser.add_argument(
        "--inputs",
        nargs="+",
        required=True,
        help="Input .npz recording paths, directories containing .npz files, or glob patterns.",
    )
    parser.add_argument("--output", required=True, help="Output .hdf5 path.")
    parser.add_argument("--overwrite", action="store_true")
    parser.add_argument(
        "--only-success",
        action="store_true",
        help="Only include recordings with episode_success=True.",
    )
    parser.add_argument(
        "--env-interface-name",
        default="MG_CustomCardboardBox",
        help="MimicGen env interface name to store in datagen_info attrs.",
    )
    parser.add_argument(
        "--env-interface-type",
        default="mujoco",
        help="MimicGen env interface type to store in datagen_info attrs.",
    )
    parser.add_argument("--env-name", default="CustomCardboardBox")
    parser.add_argument("--env-type", default="mujoco")
    parser.add_argument(
        "--subtask-scheme",
        choices=(
            "recorded",
            "contact_lift",
            "contact_stand_lift",
            "contact_stand_grasp_lift",
        ),
        default="recorded",
        help=(
            "Subtask termination signals to write. recorded preserves the .npz "
            "signals; contact_lift adds gripper_contact and box_lifted signals; "
            "contact_stand_lift also adds box_stood and keeps the gripper open "
            "until the box is upright; contact_stand_grasp_lift also adds "
            "grasp_ready at the recorded close timing."
        ),
    )
    parser.add_argument(
        "--lift-threshold",
        type=float,
        default=0.015,
        help="Box z increase in meters used for contact_lift box_lifted signal.",
    )
    parser.add_argument(
        "--stand-axis-threshold",
        type=float,
        default=0.85,
        help=(
            "World-z component threshold for a cardboard_box local x/y axis used "
            "for contact_stand_lift and contact_stand_grasp_lift box_stood signal."
        ),
    )
    parser.add_argument(
        "--grasp-ready-pre-close-steps",
        type=int,
        default=1,
        help=(
            "For contact_stand_grasp_lift, place grasp_ready this many steps "
            "before the original recorded gripper-close transition."
        ),
    )
    parser.add_argument("--no-compression", action="store_true")
    return parser.parse_args()


def main() -> None:
    convert_npz_to_hdf5(parse_args())


if __name__ == "__main__":
    main()
