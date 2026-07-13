# Copyright (c) 2024 NVIDIA CORPORATION & AFFILIATES. All rights reserved.
#
# Licensed under the NVIDIA Source Code License [see LICENSE for details].

"""
Task configs for local custom MuJoCo environments.
"""
from mimicgen.configs.config import MG_Config


class CustomCardboardBox_Config(MG_Config):
    """
    Custom PandaOmron cardboard-box fragile-wall lift scene.
    """

    NAME = "custom_cardboard_box"
    TYPE = "mujoco"

    def experiment_config(self):
        super().experiment_config()
        self.experiment.name = "custom_cardboard_box_demo"
        self.experiment.source.dataset_path = (
            "/tmp/custom_cardboard_box_source_success_demo0_only.hdf5"
        )
        self.experiment.generation.path = "/tmp/custom_cardboard_box_mimicgen"
        self.experiment.generation.num_trials = 10
        self.experiment.generation.keep_failed = True
        self.experiment.render_video = False
        self.experiment.task.env_meta_update_kwargs.env_kwargs = dict(
            goal_definition=dict(
                type="fragile_wall_lift",
                stable_steps=15,
                margin=0.0,
            ),
            randomize_on_reset=True,
        )

    def obs_config(self):
        super().obs_config()
        self.obs.collect_obs = True
        self.obs.camera_names = []

    def task_config(self):
        self.task.task_spec.subtask_1 = dict(
            object_ref="cardboard_box",
            subtask_term_signal=None,
            subtask_term_offset_range=None,
            selection_strategy="nearest_neighbor_object",
            selection_strategy_kwargs=dict(nn_k=3),
            action_noise=0.0,
            num_interpolation_steps=10,
            num_fixed_steps=0,
            apply_noise_during_interpolation=False,
        )
        self.task.task_spec.do_not_lock_keys()
