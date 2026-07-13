from __future__ import annotations

import argparse
import json
from pathlib import Path

import h5py
import imageio.v2 as imageio

import run_custom_cardboard_box_mimicgen as mimicgen_runner


def model_xml_from_dataset(h5: h5py.File) -> str:
    env_args_raw = h5["data"].attrs.get("env_args")
    if env_args_raw is None:
        raise ValueError("Dataset is missing data/env_args metadata.")
    if isinstance(env_args_raw, bytes):
        env_args_raw = env_args_raw.decode("utf-8")
    env_args = json.loads(env_args_raw)
    model_xml = env_args.get("env_kwargs", {}).get("model_xml")
    if not model_xml:
        raise ValueError("Dataset env_args is missing env_kwargs.model_xml.")
    return str(model_xml)


def render_demo(args: argparse.Namespace) -> Path:
    mimicgen_runner.install_mimicgen_patches()
    import mimicgen.utils.robomimic_utils as robomimic_utils

    dataset_path = Path(args.dataset).expanduser().resolve()
    with h5py.File(dataset_path, "r") as h5:
        demo_key = args.demo
        if demo_key not in h5["data"]:
            raise ValueError(f"Missing demo '{demo_key}' in {dataset_path}")
        states = h5[f"data/{demo_key}/states"][:]
        model_xml = args.model_xml or model_xml_from_dataset(h5)

    output_path = (
        Path(args.output).expanduser().resolve()
        if args.output is not None
        else dataset_path.with_name(
            f"{dataset_path.stem}_{demo_key}_{args.camera}.mp4"
        )
    )
    output_path.parent.mkdir(parents=True, exist_ok=True)

    env = robomimic_utils.EnvCustomCardboardBox(
        env_name="CustomCardboardBox",
        model_xml=model_xml,
        randomize_on_reset=False,
        render_offscreen=True,
        camera_names=[args.camera],
        camera_height=args.height,
        camera_width=args.width,
    )

    writer = imageio.get_writer(output_path, fps=args.fps)
    frame_count = 0
    try:
        for index, state in enumerate(states):
            if index % args.video_skip != 0:
                continue
            env.reset_to({"states": state})
            frame = env.render(
                mode="rgb_array",
                height=args.height,
                width=args.width,
                camera_name=args.camera,
            )
            writer.append_data(frame)
            frame_count += 1
    finally:
        writer.close()

    print(f"MP4 saved: {output_path} ({demo_key}, {args.camera}, {frame_count} frames)")
    return output_path


def parse_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser(
        description="Render a generated MimicGen HDF5 state trajectory to MP4."
    )
    parser.add_argument("dataset", help="Generated HDF5 path, e.g. demo_failed.hdf5.")
    parser.add_argument("--demo", default="demo_0")
    parser.add_argument("--camera", default="robot_side_view")
    parser.add_argument("--output", default=None)
    parser.add_argument("--model-xml", default=None)
    parser.add_argument("--width", type=int, default=640)
    parser.add_argument("--height", type=int, default=480)
    parser.add_argument("--fps", type=int, default=30)
    parser.add_argument("--video-skip", type=int, default=8)
    return parser.parse_args()


def main() -> None:
    render_demo(parse_args())


if __name__ == "__main__":
    main()
