from __future__ import annotations

from teleop_custom.common import (
    ARM_ACTUATORS,
    FRAGILE_WALL_BREAK_FORCE_THRESHOLD,
    GRIPPER_ACTUATOR,
)
from teleop_custom.cli import main, parse_args
from teleop_custom.control import CartesianController
from teleop_custom.fragile_wall import FragileWall
from teleop_custom.goals import GoalEvaluator, pose_matrix_from_body
from teleop_custom.placement import randomize_red_box_xy_in_tray
from teleop_custom.robocasa_scene import (
    discover_robocasa_island_layout_ids,
    generate_random_island_scene_xml,
    validate_generated_fragile_wall_model,
)
from teleop_custom.robots import get_actuator_id, set_initial_pose
from teleop_custom.session import launch_teleop, run_teleop_session


__all__ = (
    "ARM_ACTUATORS",
    "CartesianController",
    "FRAGILE_WALL_BREAK_FORCE_THRESHOLD",
    "FragileWall",
    "GRIPPER_ACTUATOR",
    "GoalEvaluator",
    "discover_robocasa_island_layout_ids",
    "generate_random_island_scene_xml",
    "get_actuator_id",
    "launch_teleop",
    "main",
    "parse_args",
    "pose_matrix_from_body",
    "randomize_red_box_xy_in_tray",
    "run_teleop_session",
    "set_initial_pose",
    "validate_generated_fragile_wall_model",
)


if __name__ == "__main__":
    main()
