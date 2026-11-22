# img2event

Scripts used to turn clear images into synthetic event streams and prepare the
event-conditioned dehazing datasets for EvDehaze.

## Layout

```text
img2event/
├── rpg_vid2e/              # upstream uzh-rpg/rpg_vid2e (git submodule, GPLv3)
├── simulate_events/        # 6-DoF TPR generator and ESIM event-stream launchers
├── convert_to_voxel/       # merge 4 motion streams into 8-channel voxels
├── utils/event_utils.py    # voxel helpers
└── split_nhhaze.py         # split official NH-HAZE into train/test folders
```

From the repository root, initialize the Vid2E submodule and build its CUDA
extension in an environment with PyTorch and a matching CUDA toolkit.

```bash
git submodule update --init --recursive
python -m pip install --no-build-isolation ./img2event/rpg_vid2e/esim_torch
```

Prepare the paired NH-HAZE train and test directories with the split script.

```bash
python img2event/split_nhhaze.py --src data/NH-HAZE --out datasets/NH-HAZE
```
