# PandaOmron Cardboard-Box MimicGen Example

This example contains the code used to generate MimicGen rollouts for a custom
MuJoCo/RoboCasa scene where a `PandaOmron` robot lifts a repo-local
`cardboard_box` above four fragile transparent walls without breaking them.

Generated artifacts are intentionally not included. Keep teleop recordings,
source HDF5 files, generated HDF5 files, and rendered MP4s outside git, for
example under `data/` or `/tmp`.

## Contents

- `run_custom_cardboard_box_mimicgen.py`: installs the local runtime patches and
  runs MimicGen generation for the PandaOmron fragile-wall lift task.
- `convert_npz_to_mimicgen_hdf5.py`: converts teleop `.npz` recordings into a
  MimicGen source HDF5 dataset.
- `render_mimicgen_hdf5_to_mp4.py`: renders generated state trajectories for
  visual inspection.
- `teleop_custom/` and `teleop_custom_scene_t265.py`: scene helpers, controller,
  fragile-wall success logic, and recording support used by the MimicGen wrapper.
- `robocasa_goal_definition/`: repo-local cardboard-box object definition.
- `third_party/robosuite_assets/teleop_scenes/`: the generated PandaOmron
  RoboCasa scene XML used by the successful run.

## Paths

When this directory lives at `mimicgen/examples/panda_omron_cardboard_box`, the
runner finds the MimicGen checkout automatically. Set these variables only if
you move files elsewhere:

```bash
export MIMICGEN_ROOT=/path/to/mimicgen
export CUSTOM_MUJOCO_REPO=/path/to/panda_omron_cardboard_box
```

The generated RoboCasa scene XML is included because it records the exact scene
used by the successful run. It may contain absolute asset references from the
local RoboSuite and RoboCasa installations used to generate it. Those asset
trees are not copied into this example; keep RoboSuite and RoboCasa installed in
the same environment, or regenerate the scene XML for your local asset paths.

## Convert Teleop Recordings

```bash
cd /path/to/mimicgen/examples/panda_omron_cardboard_box
conda run -n mujoco_t265 python convert_npz_to_mimicgen_hdf5.py \
    --inputs /path/to/teleop_npz_dir \
    --output /tmp/custom_cardboard_box_source_success_8_contact_stand_grasp_lift.hdf5 \
    --only-success \
    --subtask-scheme contact_stand_grasp_lift \
    --overwrite
```

For the most stable generation run, create a source file that keeps only
`demo_0` from the converted source dataset:

```bash
conda run -n mujoco_t265 python -c "import h5py; src='/tmp/custom_cardboard_box_source_success_8_contact_stand_grasp_lift.hdf5'; dst='/tmp/custom_cardboard_box_source_success_demo0_only.hdf5'; f=h5py.File(src, 'r'); g=h5py.File(dst, 'w'); gd=g.create_group('data'); [gd.attrs.__setitem__(k, v) for k, v in f['data'].attrs.items()]; f.copy('data/demo_0', gd, name='demo_0'); f.close(); g.close()"
```

## Generate Successful Rollouts

```bash
cd /path/to/mimicgen/examples/panda_omron_cardboard_box
conda run -n mujoco_t265 python run_custom_cardboard_box_mimicgen.py \
    --source /tmp/custom_cardboard_box_source_success_demo0_only.hdf5 \
    --folder /tmp/custom_cardboard_box_mimicgen_demo0only_latched_lift022_success10 \
    --num-demos 10 \
    --guarantee-success \
    --seed 221 \
    --contact-stand-grasp-lift-subtasks \
    --reset-from-source-placements \
    --freeze-eef-rotation-after-stood \
    --freeze-eef-rotation-after-close \
    --script-stood-point-local-axis-down z \
    --script-position-regrasp-after-stood \
    --script-retreat-steps 20 \
    --script-retreat-height 0.05 \
    --script-position-regrasp-steps 160 \
    --script-position-regrasp-side-offset 0.035 \
    --script-position-regrasp-z-offset 0.005 \
    --script-position-regrasp-xy-tolerance 0.04 \
    --script-position-regrasp-z-tolerance 0.04 \
    --script-lift-hold-steps 80
```

The key runtime choices are the `contact_stand_grasp_lift` subtask split, source
placement resets, fixed EEF rotation after the box stands, a latched
position-based regrasp, and a lift height of `0.22`.

## Render a Generated Demo

```bash
MUJOCO_GL=egl conda run -n mujoco_t265 python render_mimicgen_hdf5_to_mp4.py \
    /tmp/custom_cardboard_box_mimicgen_demo0only_latched_lift022_success10/custom_cardboard_box_demo/demo.hdf5 \
    --demo demo_0 \
    --camera robot_side_view \
    --output /tmp/mimicgen_demo0_success_robot_side_view.mp4 \
    --video-skip 8
```

Useful camera names are `agentview_left`, `agentview_right`,
`robot0_eye_in_hand`, and `robot_side_view`.
