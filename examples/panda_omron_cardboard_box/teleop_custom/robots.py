from __future__ import annotations

from .common import (
    ARM_ACTUATORS,
    ARM_JOINTS,
    CARDBOARD_BOX_FREEJOINT,
    GRIPPER_ACTUATOR,
    TACTILE_GRIPPER_BODIES,
    WRIST_CAMERA_NAME,
    dataclass,
    mujoco,
)

@dataclass(frozen=True)
class RobotControlProfile:
    name: str
    eef_body: str
    arm_actuators: tuple[str, ...]
    arm_joints: tuple[str, ...]
    gripper_actuators: tuple[str, ...]
    gripper_open_ctrl: tuple[float, ...]
    gripper_closed_ctrl: tuple[float, ...]
    tactile_gripper_bodies: tuple[str, ...]
    initial_joint_qpos: dict[str, float]
    initial_actuator_ctrl: dict[str, float]
    wrist_camera_name: str | None = None


UR10E_PROFILE = RobotControlProfile(
    name="ur10e",
    eef_body="ur10e_gripper_base",
    arm_actuators=ARM_ACTUATORS,
    arm_joints=ARM_JOINTS,
    gripper_actuators=(GRIPPER_ACTUATOR,),
    gripper_open_ctrl=(0.0,),
    gripper_closed_ctrl=(255.0,),
    tactile_gripper_bodies=TACTILE_GRIPPER_BODIES,
    initial_joint_qpos={
        "ur10e_shoulder_pan_joint": -3.04,
        "ur10e_shoulder_lift_joint": -1.45,
        "ur10e_elbow_joint": 1.95,
        "ur10e_wrist_1_joint": -2.32,
        "ur10e_wrist_2_joint": -1.51,
        "ur10e_wrist_3_joint": -1.76,
    },
    initial_actuator_ctrl={
        "ur10e_shoulder_pan": -3.04,
        "ur10e_shoulder_lift": -1.45,
        "ur10e_elbow": 1.95,
        "ur10e_wrist_1": -2.32,
        "ur10e_wrist_2": -1.51,
        "ur10e_wrist_3": -1.76,
        "ur10e_gripper_fingers_actuator": 0.0,
    },
    wrist_camera_name=WRIST_CAMERA_NAME,
)

PANDA_OMRON_PROFILE = RobotControlProfile(
    name="panda_omron",
    eef_body="robot0_eef",
    arm_actuators=(
        "robot0_joint1",
        "robot0_joint2",
        "robot0_joint3",
        "robot0_joint4",
        "robot0_joint5",
        "robot0_joint6",
        "robot0_joint7",
    ),
    arm_joints=(
        "robot0_joint1",
        "robot0_joint2",
        "robot0_joint3",
        "robot0_joint4",
        "robot0_joint5",
        "robot0_joint6",
        "robot0_joint7",
    ),
    gripper_actuators=("robot0_gripper_finger_joint1", "robot0_gripper_finger_joint2"),
    gripper_open_ctrl=(0.04, -0.04),
    gripper_closed_ctrl=(0.0, 0.0),
    tactile_gripper_bodies=(
        "robot0_finger_joint1_tip",
        "robot0_finger_joint2_tip",
    ),
    initial_joint_qpos={
        "robot0_joint1": 0.94992841,
        "robot0_joint2": -1.00369184,
        "robot0_joint3": -0.38400643,
        "robot0_joint4": -2.07811249,
        "robot0_joint5": -0.35612679,
        "robot0_joint6": 1.13453909,
        "robot0_joint7": -0.27537354,
        "robot0_finger_joint1": 0.04,
        "robot0_finger_joint2": -0.04,
        "mobilebase0_joint_torso_height": 0.0,
    },
    initial_actuator_ctrl={
        "robot0_joint1": 0.94992841,
        "robot0_joint2": -1.00369184,
        "robot0_joint3": -0.38400643,
        "robot0_joint4": -2.07811249,
        "robot0_joint5": -0.35612679,
        "robot0_joint6": 1.13453909,
        "robot0_joint7": -0.27537354,
        "robot0_gripper_finger_joint1": 0.04,
        "robot0_gripper_finger_joint2": -0.04,
        "mobilebase0_actuator_mobile_forward": 0.0,
        "mobilebase0_actuator_mobile_side": 0.0,
        "mobilebase0_actuator_mobile_yaw": 0.0,
        "mobilebase0_actuator_torso_height": 0.0,
    },
    wrist_camera_name="robot0_eye_in_hand",
)

PANDA_OMRON_ROBOCASA_PROFILE = RobotControlProfile(
    name="panda_omron_robocasa",
    eef_body="gripper0_right_eef",
    arm_actuators=(
        "robot0_joint1",
        "robot0_joint2",
        "robot0_joint3",
        "robot0_joint4",
        "robot0_joint5",
        "robot0_joint6",
        "robot0_joint7",
    ),
    arm_joints=(
        "robot0_joint1",
        "robot0_joint2",
        "robot0_joint3",
        "robot0_joint4",
        "robot0_joint5",
        "robot0_joint6",
        "robot0_joint7",
    ),
    gripper_actuators=(
        "gripper0_right_gripper_finger_joint1",
        "gripper0_right_gripper_finger_joint2",
    ),
    gripper_open_ctrl=(0.04, -0.04),
    gripper_closed_ctrl=(0.0, 0.0),
    tactile_gripper_bodies=(
        "gripper0_right_finger_joint1_tip",
        "gripper0_right_finger_joint2_tip",
    ),
    initial_joint_qpos={
        "robot0_joint1": 0.94992841,
        "robot0_joint2": -1.00369184,
        "robot0_joint3": -0.38400643,
        "robot0_joint4": -2.07811249,
        "robot0_joint5": -0.35612679,
        "robot0_joint6": 1.13453909,
        "robot0_joint7": -0.27537354,
        "gripper0_right_finger_joint1": 0.04,
        "gripper0_right_finger_joint2": -0.04,
        "mobilebase0_joint_torso_height": 0.0,
    },
    initial_actuator_ctrl={
        "robot0_joint1": 0.94992841,
        "robot0_joint2": -1.00369184,
        "robot0_joint3": -0.38400643,
        "robot0_joint4": -2.07811249,
        "robot0_joint5": -0.35612679,
        "robot0_joint6": 1.13453909,
        "robot0_joint7": -0.27537354,
        "gripper0_right_gripper_finger_joint1": 0.04,
        "gripper0_right_gripper_finger_joint2": -0.04,
        "mobilebase0_actuator_mobile_forward": 0.0,
        "mobilebase0_actuator_mobile_side": 0.0,
        "mobilebase0_actuator_mobile_yaw": 0.0,
        "mobilebase0_actuator_torso_height": 0.0,
    },
    wrist_camera_name="robot0_eye_in_hand",
)


MOBILE_BASE_JOINTS = (
    "mobilebase0_joint_mobile_forward",
    "mobilebase0_joint_mobile_side",
    "mobilebase0_joint_mobile_yaw",
    "mobilebase0_joint_torso_height",
)

MOBILE_BASE_ACTUATORS = (
    "mobilebase0_actuator_mobile_forward",
    "mobilebase0_actuator_mobile_side",
    "mobilebase0_actuator_mobile_yaw",
    "mobilebase0_actuator_torso_height",
)


def resolve_robot_control_profile(model: mujoco.MjModel) -> RobotControlProfile:
    profiles = (PANDA_OMRON_PROFILE, PANDA_OMRON_ROBOCASA_PROFILE, UR10E_PROFILE)
    for profile in profiles:
        eef_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, profile.eef_body)
        actuator_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
            for name in profile.arm_actuators
        ]
        joint_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
            for name in profile.arm_joints
        ]
        if eef_id >= 0 and all(i >= 0 for i in actuator_ids + joint_ids):
            return profile
    raise ValueError(
        "Could not resolve a supported robot profile. Expected UR10e or PandaOmron names."
    )


def set_joint_qpos(
    model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float
) -> None:
    joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, name)
    if joint_id >= 0:
        data.qpos[model.jnt_qposadr[joint_id]] = value


def set_actuator_ctrl(
    model: mujoco.MjModel, data: mujoco.MjData, name: str, value: float
) -> None:
    actuator_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)
    if actuator_id >= 0:
        data.ctrl[actuator_id] = value


def lock_mobile_base(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_profile: RobotControlProfile | None = None,
) -> None:
    if robot_profile is None:
        robot_profile = resolve_robot_control_profile(model)

    for joint_name in MOBILE_BASE_JOINTS:
        joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
        if joint_id < 0:
            continue
        qpos_adr = model.jnt_qposadr[joint_id]
        qvel_adr = model.jnt_dofadr[joint_id]
        data.qpos[qpos_adr] = robot_profile.initial_joint_qpos.get(joint_name, 0.0)
        data.qvel[qvel_adr] = 0.0

    for actuator_name in MOBILE_BASE_ACTUATORS:
        actuator_id = mujoco.mj_name2id(
            model,
            mujoco.mjtObj.mjOBJ_ACTUATOR,
            actuator_name,
        )
        if actuator_id < 0:
            continue
        data.ctrl[actuator_id] = robot_profile.initial_actuator_ctrl.get(
            actuator_name,
            0.0,
        )


def set_initial_pose(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    robot_profile: RobotControlProfile | None = None,
) -> None:
    data.qvel[:] = 0.0

    box_joint_id = mujoco.mj_name2id(
        model, mujoco.mjtObj.mjOBJ_JOINT, CARDBOARD_BOX_FREEJOINT
    )
    if box_joint_id >= 0:
        qpos_adr = model.jnt_qposadr[box_joint_id]
        qvel_adr = model.jnt_dofadr[box_joint_id]
        data.qpos[qpos_adr : qpos_adr + 7] = model.qpos0[qpos_adr : qpos_adr + 7]
        data.qvel[qvel_adr : qvel_adr + 6] = 0.0

    if robot_profile is None:
        robot_profile = resolve_robot_control_profile(model)

    for joint_name, qpos in robot_profile.initial_joint_qpos.items():
        set_joint_qpos(model, data, joint_name, qpos)

    for actuator_name, ctrl in robot_profile.initial_actuator_ctrl.items():
        set_actuator_ctrl(model, data, actuator_name, ctrl)

    lock_mobile_base(model, data, robot_profile)
    mujoco.mj_forward(model, data)


def get_actuator_id(model: mujoco.MjModel, name: str) -> int:
    return mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_ACTUATOR, name)


def clamp_ctrl(model: mujoco.MjModel, actuator_id: int, value: float) -> float:
    if model.actuator_ctrllimited[actuator_id]:
        low, high = model.actuator_ctrlrange[actuator_id]
        return float(min(max(value, low), high))
    return value
