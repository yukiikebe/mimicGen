from __future__ import annotations

from .common import MAX_RECORDED_CONTACTS, TACTILE_BOX_BODY, TACTILE_GRIPPER_BODIES, mujoco, np

def body_is_descendant(
    model: mujoco.MjModel, body_id: int, ancestor_body_ids: set[int]
) -> bool:
    while body_id > 0:
        if body_id in ancestor_body_ids:
            return True
        body_id = int(model.body_parentid[body_id])
    return body_id in ancestor_body_ids


def make_tactile_contact_filter(
    model: mujoco.MjModel,
    gripper_bodies: tuple[str, ...] = TACTILE_GRIPPER_BODIES,
) -> tuple[set[int], set[int]]:
    box_body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, TACTILE_BOX_BODY)
    if box_body_id < 0:
        raise ValueError(f"Missing tactile box body: {TACTILE_BOX_BODY}")

    gripper_body_ids = set()
    for name in gripper_bodies:
        body_id = mujoco.mj_name2id(model, mujoco.mjtObj.mjOBJ_BODY, name)
        if body_id < 0:
            raise ValueError(f"Missing tactile gripper body: {name}")
        gripper_body_ids.add(body_id)

    return {box_body_id}, gripper_body_ids


def geom_matches_body_set(
    model: mujoco.MjModel, geom_id: int, body_ids: set[int]
) -> bool:
    if geom_id < 0:
        return False
    body_id = int(model.geom_bodyid[geom_id])
    return body_is_descendant(model, body_id, body_ids)


def extract_tactile_contacts(
    model: mujoco.MjModel,
    data: mujoco.MjData,
    contact_filter: tuple[set[int], set[int]],
    max_contacts: int = MAX_RECORDED_CONTACTS,
) -> dict[str, np.ndarray | np.integer]:
    box_body_ids, gripper_body_ids = contact_filter
    contact_pos = np.full((max_contacts, 3), np.nan, dtype=np.float64)
    contact_frame = np.full((max_contacts, 3, 3), np.nan, dtype=np.float64)
    contact_dist = np.full(max_contacts, np.nan, dtype=np.float64)
    contact_geom1 = np.full(max_contacts, -1, dtype=np.int32)
    contact_geom2 = np.full(max_contacts, -1, dtype=np.int32)
    contact_force = np.full((max_contacts, 6), np.nan, dtype=np.float64)

    recorded_count = 0
    force = np.zeros(6, dtype=np.float64)
    for contact_id in range(data.ncon):
        contact = data.contact[contact_id]
        geom1 = int(contact.geom1)
        geom2 = int(contact.geom2)
        geom1_is_box = geom_matches_body_set(model, geom1, box_body_ids)
        geom2_is_box = geom_matches_body_set(model, geom2, box_body_ids)
        geom1_is_gripper = geom_matches_body_set(model, geom1, gripper_body_ids)
        geom2_is_gripper = geom_matches_body_set(model, geom2, gripper_body_ids)
        if not (
            (geom1_is_box and geom2_is_gripper) or (geom2_is_box and geom1_is_gripper)
        ):
            continue
        if recorded_count >= max_contacts:
            break

        mujoco.mj_contactForce(model, data, contact_id, force)
        contact_pos[recorded_count] = contact.pos
        contact_frame[recorded_count] = np.asarray(contact.frame).reshape(3, 3)
        contact_dist[recorded_count] = contact.dist
        contact_geom1[recorded_count] = geom1
        contact_geom2[recorded_count] = geom2
        contact_force[recorded_count] = force
        recorded_count += 1

    return {
        "contact_count": np.int32(recorded_count),
        "contact_pos": contact_pos,
        "contact_frame": contact_frame,
        "contact_dist": contact_dist,
        "contact_geom1": contact_geom1,
        "contact_geom2": contact_geom2,
        "contact_force": contact_force,
    }
