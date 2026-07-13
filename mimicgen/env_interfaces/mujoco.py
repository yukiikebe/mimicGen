# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
MimicGen environment interfaces for local custom MuJoCo environments.
"""
import numpy as np

from mimicgen.env_interfaces.base import MG_EnvInterface


class CustomMuJoCoInterface(MG_EnvInterface):
    """
    Base interface for the local MuJoCo EnvBase wrappers.
    """

    INTERFACE_TYPE = "mujoco"
    CLIP_ACTIONS = False

    def get_robot_eef_pose(self):
        robot_profile = getattr(self.env, "robot_profile", None)
        if robot_profile is not None:
            return self.env.get_body_pose(robot_profile.eef_body)
        return self.env.get_body_pose("ur10e_gripper_base")

    def target_pose_to_action(self, target_pose, relative=True):
        del relative
        return self.env.target_pose_to_action(target_pose)

    def action_to_target_pose(self, action, relative=True):
        del relative
        return self.env.action_to_target_pose(action)

    def action_to_gripper_action(self, action):
        return np.asarray(action[6:7], dtype=np.float64)


class MG_CustomCardboardBox(CustomMuJoCoInterface):
    """
    Interface for the custom cardboard-box fragile-wall lift scene.
    """

    def get_object_poses(self):
        return self.env.get_object_poses()

    def get_subtask_term_signals(self):
        return self.env.get_subtask_term_signals()
