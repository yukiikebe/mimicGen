from __future__ import annotations

from .common import (
    CARDBOARD_BOX_FREEJOINT,
    DEFAULT_RANDOM_SCENE_STYLE_IDS,
    MUJOCO_ROOT,
    Path,
    RANDOM_SCENE_GENERATION_MAX_ATTEMPTS,
    ROBOCASA_ISLAND_FIXTURE_MARKER,
    argparse,
    datetime,
    mujoco,
    np,
    re,
    subprocess,
    sys,
)
from .control import CartesianController
from .placement import align_freejoint_body_bottom_to_support, randomize_red_box_xy_in_tray, reset_freejoint_to_model_qpos0
from .robots import resolve_robot_control_profile, set_initial_pose

def robocasa_layout_root(robocasa_root: Path) -> Path:
    return (
        robocasa_root
        / "robocasa"
        / "models"
        / "assets"
        / "scenes"
        / "kitchen_layouts"
    )


def discover_robocasa_island_layout_ids(robocasa_root: Path) -> tuple[int, ...]:
    layout_root = robocasa_layout_root(robocasa_root)
    if not layout_root.exists():
        raise FileNotFoundError(f"RoboCasa kitchen layout directory not found: {layout_root}")

    layout_ids = []
    for layout_path in sorted(layout_root.glob("*/layout*.yaml")):
        text = layout_path.read_text(encoding="utf-8")
        if ROBOCASA_ISLAND_FIXTURE_MARKER not in text:
            continue
        match = re.fullmatch(r"layout(\d+)", layout_path.stem)
        if match is None:
            continue
        layout_ids.append(int(match.group(1)))
    if not layout_ids:
        raise ValueError(
            "No RoboCasa layouts containing "
            f"{ROBOCASA_ISLAND_FIXTURE_MARKER!r} were found under {layout_root}."
        )
    return tuple(sorted(set(layout_ids)))


def sample_choice(rng: np.random.Generator, values: tuple[int, ...]) -> int:
    if not values:
        raise ValueError("Cannot sample from an empty choice set.")
    index = int(rng.integers(0, len(values)))
    return int(values[index])


def validate_generated_fragile_wall_model(model_path: Path) -> None:
    model = mujoco.MjModel.from_xml_path(str(model_path))
    data = mujoco.MjData(model)
    robot_profile = resolve_robot_control_profile(model)
    set_initial_pose(model, data, robot_profile)
    reset_freejoint_to_model_qpos0(model, data, CARDBOARD_BOX_FREEJOINT)
    align_freejoint_body_bottom_to_support(model, data)
    randomize_red_box_xy_in_tray(model, data, np.random.default_rng(0))
    CartesianController(model, data, robot_profile)


def generate_random_island_scene_xml(
    args: argparse.Namespace, rng: np.random.Generator
) -> Path:
    robocasa_root = Path(args.robocasa_root).expanduser().resolve()
    robosuite_root = Path(args.robosuite_root).expanduser().resolve()
    discovered_layout_ids = discover_robocasa_island_layout_ids(robocasa_root)
    if args.random_scene_layout_ids:
        layout_ids = tuple(int(value) for value in args.random_scene_layout_ids)
        invalid_layout_ids = sorted(set(layout_ids) - set(discovered_layout_ids))
        if invalid_layout_ids:
            raise ValueError(
                "--random-scene-layout-ids includes layout(s) without "
                f"{ROBOCASA_ISLAND_FIXTURE_MARKER}: {invalid_layout_ids}"
            )
    else:
        layout_ids = discovered_layout_ids
    style_ids = (
        tuple(int(value) for value in args.random_scene_style_ids)
        if args.random_scene_style_ids
        else DEFAULT_RANDOM_SCENE_STYLE_IDS
    )
    output_dir = Path(args.random_scene_output_dir).expanduser().resolve()
    output_dir.mkdir(parents=True, exist_ok=True)
    generator = MUJOCO_ROOT / "generate_robocasa_fragile_wall_scene.py"
    last_error = ""
    for attempt in range(1, RANDOM_SCENE_GENERATION_MAX_ATTEMPTS + 1):
        layout_id = sample_choice(rng, layout_ids)
        style_id = sample_choice(rng, style_ids)
        seed = int(rng.integers(0, np.iinfo(np.int32).max))
        timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S_%f")
        output_path = (
            output_dir
            / (
                f"panda_omron_fragile_wall_layout{layout_id:03d}_"
                f"style{style_id:03d}_{timestamp}.xml"
            )
        )
        command = [
            sys.executable,
            str(generator),
            "--robocasa-root",
            str(robocasa_root),
            "--robosuite-root",
            str(robosuite_root),
            "--layout-id",
            str(layout_id),
            "--style-id",
            str(style_id),
            "--seed",
            str(seed),
            "--task-fixture",
            ROBOCASA_ISLAND_FIXTURE_MARKER,
            "--robot-base-ref",
            ROBOCASA_ISLAND_FIXTURE_MARKER,
            "--output",
            str(output_path),
        ]
        print(
            "Generating random RoboCasa island scene: "
            f"layout={layout_id}, style={style_id}, seed={seed}",
            flush=True,
        )
        result = subprocess.run(command, check=False, capture_output=True, text=True)
        combined_output = "\n".join(part for part in (result.stdout, result.stderr) if part)
        if result.returncode == 0:
            if result.stdout:
                print(result.stdout, flush=True)
            validate_generated_fragile_wall_model(output_path)
            return output_path

        last_error = combined_output
        placement_error = any(
            marker in combined_output
            for marker in (
                "does not fit on fixture",
                "would overhang tabletop",
                "Could not sample a valid tabletop task center",
            )
        )
        if not placement_error:
            if result.stdout:
                print(result.stdout, flush=True)
            if result.stderr:
                print(result.stderr, file=sys.stderr, flush=True)
            raise RuntimeError(
                "Random RoboCasa scene generation failed. "
                "The mujoco_t265 environment must be able to import RoboCasa "
                "dependencies such as yaml and gymnasium."
            )
        print(
            "Rejected random scene candidate that cannot fit the fragile wall "
            f"enclosure on the island (attempt {attempt}/"
            f"{RANDOM_SCENE_GENERATION_MAX_ATTEMPTS}).",
            flush=True,
        )

    raise RuntimeError(
        "Could not generate a valid random RoboCasa island scene after "
        f"{RANDOM_SCENE_GENERATION_MAX_ATTEMPTS} attempts. Last error:\n{last_error}"
    )

