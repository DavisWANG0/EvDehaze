import glob
import os
import shutil
from os.path import join, basename, splitext
import cv2
import numpy as np
import torch
import tqdm
from absl import app, flags
from absl.logging import info
import esim_torch
from concurrent.futures import ThreadPoolExecutor, as_completed

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", "", "Input directory containing PNG images")
flags.DEFINE_string("output_root", "./events_output", "Root directory for output events")
flags.DEFINE_float("max_translation", 3, "Max translation in pixels")
flags.DEFINE_float("max_rotation", 0.2, "Max rotation in degrees")
flags.DEFINE_float("max_scale", 0.05, "Max scale factor for radial motion")
flags.DEFINE_integer("num_frames", 21, "Number of generated frames per sequence")
flags.DEFINE_float("fps", 240.0, "Frames per second for event simulation")
flags.DEFINE_float("contrast_threshold_neg", 0.2, "")
flags.DEFINE_float("contrast_threshold_pos", 0.2, "")
flags.DEFINE_float("refractory_period", 0.0, "")
flags.DEFINE_integer("num_threads", 8, "Number of concurrent threads")

motion_types = ['translation_x', 'translation_y', 'rotation', 'radial']


def generate_sequence(img, motion, image_name):
    temp_dir = join(FLAGS.output_root, f"{motion}_temp", image_name)
    os.makedirs(temp_dir, exist_ok=True)
    h, w = img.shape[:2]
    center = (w//2, h//2)
    center_frame = FLAGS.num_frames // 2

    for i in range(FLAGS.num_frames):
        filename = join(temp_dir, f"{i:04d}.png")
        if motion == 'translation_x':
            offset = (i - center_frame) * (FLAGS.max_translation / center_frame)
            M = np.float32([[1, 0, offset], [0, 1, 0]])
        elif motion == 'translation_y':
            offset = (i - center_frame) * (FLAGS.max_translation / center_frame)
            M = np.float32([[1, 0, 0], [0, 1, offset]])
        elif motion == 'rotation':
            angle = (i - center_frame) * (FLAGS.max_rotation / center_frame)
            M = cv2.getRotationMatrix2D(center, angle, 1)
        elif motion == 'radial':
            scale = 1 + ((i - center_frame) * (FLAGS.max_scale / center_frame))
            M = cv2.getRotationMatrix2D(center, 0, scale)
        else:
            raise ValueError("Invalid motion type")
        warped = cv2.warpAffine(img, M, dsize=(w, h), flags=cv2.INTER_LINEAR, borderMode=cv2.BORDER_REFLECT)
        cv2.imwrite(filename, warped)
    return temp_dir


def simulate_events_for_image_motion(image_path, motion):
    name = splitext(basename(image_path))[0]
    base_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
    if base_img is None:
        return f"Skip: {image_path}"

    esim = esim_torch.esim_torch.EventSimulator_torch(
        FLAGS.contrast_threshold_neg,
        FLAGS.contrast_threshold_pos,
        FLAGS.refractory_period
    )
    temp_dir = generate_sequence(base_img, motion, name)

    img_files = sorted(glob.glob(join(temp_dir, "*.png")))
    timestamps_ns = torch.arange(
        0, len(img_files) * 1e9/FLAGS.fps, 1e9/FLAGS.fps, dtype=torch.int64
    ).cuda()

    t, x, y, p = [], [], [], []
    for img_path, ts in zip(img_files, timestamps_ns):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        log_img = torch.from_numpy(np.log(img.astype(np.float32)/255 + 1e-5)).cuda()
        events = esim.forward(log_img, ts)
        if events is None:
            continue
        t.append(events['t'])
        x.append(events['x'])
        y.append(events['y'])
        p.append(events['p'])

    t = torch.cat(t)
    x = torch.cat(x)
    y = torch.cat(y)
    p = torch.cat(p)

    save_npz_dir = join(FLAGS.output_root, motion)
    os.makedirs(save_npz_dir, exist_ok=True)
    np.savez_compressed(join(save_npz_dir, name + ".npz"),
                        t=t.cpu().numpy(), x=x.cpu().numpy(), y=y.cpu().numpy(), p=p.cpu().numpy())
                        # t=t.numpy(), x=x.numpy(), y=y.numpy(), p=p.numpy())

    H = y.max().item() + 1
    W = x.max().item() + 1
    
    y_np = y.cpu().numpy()
    x_np = x.cpu().numpy()
    p_np = p.cpu().numpy()
    
    event_img = np.zeros((H, W), dtype=np.uint8)
    # event_img[y.numpy(), x.numpy()] = (p.numpy() * 255).astype(np.uint8)
    event_img[y_np, x_np] = (p_np * 255).astype(np.uint8)

    vis_dir = join(FLAGS.output_root, f"{motion}_vis")
    os.makedirs(vis_dir, exist_ok=True)
    cv2.imwrite(join(vis_dir, name + "_events.png"), event_img)

    shutil.rmtree(temp_dir, ignore_errors=True)
    return f"Done: {name} [{motion}]"


def main(argv):
    all_images = sorted(glob.glob(join(FLAGS.input_dir, "*.png")))
    total_tasks = len(all_images) * len(motion_types)

    with ThreadPoolExecutor(max_workers=FLAGS.num_threads) as executor:
        futures = []
        for image_path in all_images:
            for motion in motion_types:
                futures.append(executor.submit(simulate_events_for_image_motion, image_path, motion))

        for f in tqdm.tqdm(as_completed(futures), total=total_tasks, desc="Total Progress"):
            try:
                result = f.result()
                info(result)
            except Exception as e:
                info(f"Error: {e}")

if __name__ == "__main__":
    app.run(main)
