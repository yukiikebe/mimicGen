from __future__ import annotations

from pathlib import Path


CARDBOARD_BOX_CATEGORY = "cardboard_box"
CARDBOARD_BOX_REGISTRY = "aigen"
CARDBOARD_BOX_SCALE = 1.0


def cardboard_box_asset_dir(mujoco_root: Path) -> Path:
    return (
        mujoco_root
        / "robocasa_goal_definition"
        / "objects"
        / CARDBOARD_BOX_CATEGORY
    )


def register_cardboard_box(mujoco_root: Path) -> None:
    """Register the repo-local cardboard box with RoboCasa's object sampler."""
    from robocasa.models.objects import kitchen_object_utils as object_utils
    from robocasa.models.objects import kitchen_objects

    asset_dir = cardboard_box_asset_dir(mujoco_root).resolve()
    if not (asset_dir / "cardboard_box_0" / "model.xml").exists():
        raise FileNotFoundError(f"Missing cardboard_box object asset under {asset_dir}")

    category = {
        CARDBOARD_BOX_REGISTRY: object_utils.ObjCat(
            name=CARDBOARD_BOX_CATEGORY,
            types=("packaged_food", "graspable_box"),
            model_folders=[str(asset_dir)],
            graspable=True,
            washable=False,
            microwavable=False,
            cookable=False,
            fridgable=False,
            freezable=False,
            dishwashable=False,
            scale=CARDBOARD_BOX_SCALE,
            solimp=(0.998, 0.998, 0.001),
            solref=(0.004, 1),
            density=450,
            friction=(0.95, 0.02, 0.001),
            reg_type="aigen_objs",
        )
    }

    # kitchen_object_utils imports these dictionaries by reference. Mutating them
    # updates Kitchen, EnvUtils, and the sampler without changing the checkout.
    kitchen_objects.OBJ_CATEGORIES[CARDBOARD_BOX_CATEGORY] = category
    object_utils.OBJ_CATEGORIES[CARDBOARD_BOX_CATEGORY] = category

    groups = kitchen_objects.OBJ_GROUPS
    for group_name, categories in (
        (CARDBOARD_BOX_CATEGORY, [CARDBOARD_BOX_CATEGORY]),
        ("packaged_food", [CARDBOARD_BOX_CATEGORY]),
        ("graspable_box", [CARDBOARD_BOX_CATEGORY]),
    ):
        existing = groups.setdefault(group_name, [])
        for category_name in categories:
            if category_name not in existing:
                existing.append(category_name)
    if CARDBOARD_BOX_CATEGORY not in groups.setdefault("all", []):
        groups["all"].append(CARDBOARD_BOX_CATEGORY)

    object_utils.OBJ_GROUPS.update(groups)
