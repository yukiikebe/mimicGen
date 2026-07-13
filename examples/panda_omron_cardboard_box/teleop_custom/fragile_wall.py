from __future__ import annotations

from .common import (
    Any,
    CARDBOARD_BOX_BODY,
    FRAGILE_WALL_BREAK_FORCE_THRESHOLD,
    FRAGILE_WALL_SHARD_OFFSETS,
    FRAGILE_WALL_SHARD_QUATS,
    FRAGILE_WALL_SHARD_STASH_POS,
    WOODEN_TRAY_BODY,
    mujoco,
    np,
)
from .placement import body_geom_ids, set_freejoint
from .robots import RobotControlProfile

def body_descends_from(
    model: mujoco.MjModel, body_id: int, ancestor_body_id: int
) -> bool:
    while body_id > 0:
        if body_id == ancestor_body_id:
            return True
        body_id = int(model.body_parentid[body_id])
    return body_id == ancestor_body_id


class FragileWall:
    def __init__(
        self,
        model: mujoco.MjModel,
        runtime_state: dict[str, Any],
        robot_profile: RobotControlProfile,
        break_force_threshold: float = FRAGILE_WALL_BREAK_FORCE_THRESHOLD,
    ) -> None:
        self.model = model
        self.runtime_state = runtime_state
        self.break_force_threshold = break_force_threshold
        self.broken = False
        self.enabled = False

        self.wall_geom_ids = np.asarray(
            body_geom_ids(
                model,
                WOODEN_TRAY_BODY,
                geom_name_prefix="tray_",
                geom_name_contains="wall",
                missing_body_ok=True,
            ),
            dtype=np.int32,
        )
        shard_geom_ids = []
        shard_indices = []
        shard_joint_names = []
        for index in range(len(FRAGILE_WALL_SHARD_OFFSETS)):
            geom_id = mujoco.mj_name2id(
                model,
                mujoco.mjtObj.mjOBJ_GEOM,
                f"fragile_wall_shard_{index}_geom",
            )
            joint_name = f"fragile_wall_shard_{index}_freejoint"
            joint_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_JOINT, joint_name)
            if geom_id < 0 or joint_id < 0:
                continue
            shard_geom_ids.append(geom_id)
            shard_indices.append(index)
            shard_joint_names.append(joint_name)
        self.shard_geom_ids = np.asarray(shard_geom_ids, dtype=np.int32)
        self.shard_indices = tuple(shard_indices)
        self.shard_joint_names = tuple(shard_joint_names)
        self.enabled = len(self.wall_geom_ids) > 0 and len(self.shard_geom_ids) > 0

        self.box_body_id = mujoco.mj_name2id(
            model, mujoco.mjtObj.mjOBJ_BODY, CARDBOARD_BOX_BODY
        )
        self.gripper_body_ids = [
            mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
            for name in robot_profile.tactile_gripper_bodies
        ]
        self._initial_geom_rgba = model.geom_rgba.copy()
        self._initial_geom_contype = model.geom_contype.copy()
        self._initial_geom_conaffinity = model.geom_conaffinity.copy()
        self._initial_mat_rgba = model.mat_rgba.copy()
        if self.enabled:
            self.wall_mat_ids = np.unique(model.geom_matid[self.wall_geom_ids])
            self.shard_mat_ids = np.unique(model.geom_matid[self.shard_geom_ids])
        else:
            self.wall_mat_ids = np.asarray([], dtype=np.int32)
            self.shard_mat_ids = np.asarray([], dtype=np.int32)
        self.runtime_state["fragile_wall_broken"] = False

    def reset(self, data: mujoco.MjData) -> None:
        self.broken = False
        self.runtime_state["fragile_wall_broken"] = False
        if not self.enabled:
            return
        self.model.geom_rgba[:] = self._initial_geom_rgba
        self.model.geom_contype[:] = self._initial_geom_contype
        self.model.geom_conaffinity[:] = self._initial_geom_conaffinity
        self.model.mat_rgba[:] = self._initial_mat_rgba
        self.model.geom_rgba[self.shard_geom_ids, 3] = 0.0
        self.model.mat_rgba[self.shard_mat_ids, 3] = 0.0
        self.model.geom_contype[self.shard_geom_ids] = 0
        self.model.geom_conaffinity[self.shard_geom_ids] = 0
        for shard_index, joint_name in zip(self.shard_indices, self.shard_joint_names):
            qpos = np.concatenate(
                [
                    FRAGILE_WALL_SHARD_STASH_POS
                    + np.array([0.0, -0.04 * shard_index, 0.0]),
                    FRAGILE_WALL_SHARD_QUATS[shard_index],
                ]
            )
            set_freejoint(self.model, data, joint_name, qpos)
        mujoco.mj_forward(self.model, data)

    def _contact_can_break_wall(self, contact) -> bool:
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        if geom1 in self.wall_geom_ids:
            other_geom = geom2
        elif geom2 in self.wall_geom_ids:
            other_geom = geom1
        else:
            return False
        other_body = int(self.model.geom_bodyid[other_geom])
        if self.box_body_id >= 0 and body_descends_from(
            self.model, other_body, self.box_body_id
        ):
            return True
        return any(
            gripper_body_id >= 0
            and body_descends_from(self.model, other_body, gripper_body_id)
            for gripper_body_id in self.gripper_body_ids
        )

    def update(self, data: mujoco.MjData) -> bool:
        if not self.enabled or self.broken:
            return False
        force = np.zeros(6, dtype=np.float64)
        for contact_id in range(data.ncon):
            contact = data.contact[contact_id]
            if not self._contact_can_break_wall(contact):
                continue
            mujoco.mj_contactForce(self.model, data, contact_id, force)
            contact_force = float(np.linalg.norm(force[:3]))
            if contact_force >= self.break_force_threshold:
                self.break_now(data, reason=f"contact force {contact_force:.2f} N")
                return True
        return False

    def break_now(self, data: mujoco.MjData, reason: str = "manual trigger") -> None:
        if not self.enabled or self.broken:
            return
        mujoco.mj_forward(self.model, data)
        wall_center = np.mean(data.geom_xpos[self.wall_geom_ids], axis=0)
        self.model.geom_rgba[self.wall_geom_ids, 3] = 0.0
        self.model.mat_rgba[self.wall_mat_ids, 3] = 0.0
        self.model.geom_contype[self.wall_geom_ids] = 0
        self.model.geom_conaffinity[self.wall_geom_ids] = 0
        self.model.geom_rgba[self.shard_geom_ids, 3] = 1.0
        self.model.mat_rgba[self.shard_mat_ids, 3] = 1.0
        self.model.geom_contype[self.shard_geom_ids] = self._initial_geom_contype[
            self.shard_geom_ids
        ]
        self.model.geom_conaffinity[self.shard_geom_ids] = (
            self._initial_geom_conaffinity[self.shard_geom_ids]
        )
        for shard_index, joint_name in zip(self.shard_indices, self.shard_joint_names):
            qpos = np.concatenate(
                [
                    wall_center + FRAGILE_WALL_SHARD_OFFSETS[shard_index],
                    FRAGILE_WALL_SHARD_QUATS[shard_index],
                ]
            )
            velocity = np.concatenate(
                [
                    2.0 * FRAGILE_WALL_SHARD_OFFSETS[shard_index]
                    + np.array([0.0, 0.0, 0.06]),
                    6.0 * FRAGILE_WALL_SHARD_OFFSETS[shard_index],
                ]
            )
            set_freejoint(self.model, data, joint_name, qpos, velocity)
        self.broken = True
        self.runtime_state["fragile_wall_broken"] = True
        mujoco.mj_forward(self.model, data)
        print(f"Fragile wall broke: {reason}", flush=True)
