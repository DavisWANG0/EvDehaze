import argparse
import tempfile
from pathlib import Path

import cv2
import numpy as np
from tqdm import tqdm

from utils.event_voxel import events_to_temporal_pyramid


def camera_intrinsics(height, width, calibration):
    if calibration is None:
        focal_length = max(height, width)
        calibration = (focal_length, focal_length, (width - 1) / 2, (height - 1) / 2)
    focal_x, focal_y, center_x, center_y = calibration
    return np.array([[focal_x, 0, center_x], [0, focal_y, center_y], [0, 0, 1]], dtype=np.float64)


def render_frames(image, intrinsics, scene_depth, translation, rotation_deg, num_frames):
    """Render a centered camera trajectory viewing a fronto-parallel plane."""
    height, width = image.shape
    inverse_intrinsics = np.linalg.inv(intrinsics)
    corners = np.array([[0, width - 1, width - 1, 0], [0, 0, height - 1, height - 1], [1, 1, 1, 1]])
    corner_rays = inverse_intrinsics @ corners
    plane_translation = np.outer(np.asarray(translation) / scene_depth, [0, 0, 1])
    rotation_vector = np.deg2rad(rotation_deg)
    identity = np.eye(3)
    for progress in np.linspace(-1.0, 1.0, num_frames):
        camera_rotation, _ = cv2.Rodrigues(progress * rotation_vector)
        if np.any(camera_rotation[2] @ corner_rays <= 0):
            raise ValueError('The camera trajectory must keep the plane in front of the image')
        reference_to_camera = camera_rotation.T @ (identity - progress * plane_translation)
        homography = intrinsics @ reference_to_camera @ inverse_intrinsics
        yield cv2.warpPerspective(
            image, homography, (width, height),
            flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT_101,
        )


def simulate_tpr(simulator, frames, timestamps_ns, height, width):
    import torch

    simulator.reset()
    timestamps = torch.from_numpy(timestamps_ns).cuda()
    chunks = {field: [] for field in ('t', 'x', 'y', 'p')}
    for frame, timestamp in zip(frames, timestamps):
        log_image = np.log(frame.astype(np.float32) / 255 + 1e-5)
        events = simulator.forward(torch.from_numpy(log_image).cuda(), timestamp)
        if events is not None:
            for field in chunks:
                chunks[field].append(events[field].cpu().numpy())

    events = {
        field: np.concatenate(arrays) if arrays else np.empty(0, dtype=np.int64)
        for field, arrays in chunks.items()
    }
    return events_to_temporal_pyramid(
        events, height, width, start_time=timestamps_ns[0], end_time=timestamps_ns[-1],
    )


def parse_args():
    parser = argparse.ArgumentParser(description='Simulate a 6-DoF camera trajectory and save its event TPR')
    parser.add_argument('--input_dir', type=Path, required=True, help='Directory of clear or ground-truth images')
    parser.add_argument('--output_dir', type=Path, required=True, help='Directory for per-image TPR .npz files')
    parser.add_argument('--intrinsics', type=float, nargs=4, metavar=('FX', 'FY', 'CX', 'CY'),
                        help='Camera intrinsics in pixels. Defaults to a centered principal point and f=max(H, W).')
    parser.add_argument('--scene_depth', type=float, default=1.0, help='Plane depth in scene units')
    parser.add_argument('--translation', type=float, nargs=3, default=(0.005, 0.005, 0.025),
                        metavar=('TX', 'TY', 'TZ'), help='Camera translation from the center pose in scene units')
    parser.add_argument('--rotation_deg', type=float, nargs=3, default=(0.2, 0.2, 0.2),
                        metavar=('RX', 'RY', 'RZ'), help='Rotation-vector components from the center pose in degrees')
    parser.add_argument('--num_frames', type=int, default=21, help='Odd number of frames including the input view')
    parser.add_argument('--fps', type=float, default=240.0, help='Frame rate of the rendered trajectory')
    parser.add_argument('--contrast_threshold', type=float, default=0.2, help='ESIM threshold for both polarities')
    args = parser.parse_args()

    values = (args.scene_depth, args.fps, args.contrast_threshold,
              *args.translation, *args.rotation_deg, *(args.intrinsics or ()))
    if not np.isfinite(values).all():
        parser.error('Camera and simulation parameters must be finite')
    if args.scene_depth <= abs(args.translation[2]):
        parser.error('--scene_depth must exceed the magnitude of TZ')
    if args.intrinsics is not None and min(args.intrinsics[:2]) <= 0:
        parser.error('FX and FY must be positive')
    if args.num_frames < 3 or args.num_frames % 2 == 0:
        parser.error('--num_frames must be odd and at least 3')
    if not 0 < args.fps <= 1e9 or (args.num_frames - 1) * 1e9 / args.fps >= np.iinfo(np.int64).max:
        parser.error('--fps must yield distinct, representable nanosecond timestamps')
    if args.contrast_threshold <= 0:
        parser.error('--contrast_threshold must be positive')
    return args


def main():
    args = parse_args()
    image_extensions = {'.png', '.jpg', '.jpeg', '.bmp', '.tif', '.tiff'}
    images = sorted(path for path in args.input_dir.iterdir()
                    if path.is_file() and path.suffix.lower() in image_extensions)
    if not images:
        raise ValueError(f'No input images found in {args.input_dir}')
    if len({path.stem for path in images}) != len(images):
        raise ValueError(f'Images in {args.input_dir} must have unique filename stems')

    from esim_torch.esim_torch import EventSimulator_torch

    simulator = EventSimulator_torch(args.contrast_threshold, args.contrast_threshold)
    timestamps_ns = np.rint(np.arange(args.num_frames) * (1e9 / args.fps)).astype(np.int64)
    args.output_dir.mkdir(parents=True, exist_ok=True)
    with tempfile.TemporaryDirectory(dir=args.output_dir) as staging_dir:
        for path in tqdm(images, desc='Simulating 6-DoF events', unit='image'):
            image = cv2.imread(str(path), cv2.IMREAD_GRAYSCALE | cv2.IMREAD_IGNORE_ORIENTATION)
            if image is None:
                raise ValueError(f'Could not read image {path}')
            height, width = image.shape
            intrinsics = camera_intrinsics(height, width, args.intrinsics)
            frames = render_frames(image, intrinsics, args.scene_depth, args.translation,
                                   args.rotation_deg, args.num_frames)
            pyramid = simulate_tpr(simulator, frames, timestamps_ns, height, width)
            filename = f'{path.stem}.npz'
            staged_path = Path(staging_dir) / filename
            np.savez_compressed(staged_path, tpr=pyramid)
            staged_path.replace(args.output_dir / filename)


if __name__ == '__main__':
    main()
