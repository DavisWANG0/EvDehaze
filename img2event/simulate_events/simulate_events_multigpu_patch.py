import glob
import os
import shutil
from os.path import join, basename, splitext
import cv2
import numpy as np
import torch
from absl import app, flags
import esim_torch
from multiprocessing import Pool
from tqdm import tqdm

FLAGS = flags.FLAGS

flags.DEFINE_string("input_dir", "", "Input directory containing PNG images")
flags.DEFINE_string("output_root", "./events_output", "Root directory for output events")
flags.DEFINE_float("max_translation", 3, "")
flags.DEFINE_float("max_rotation", 0.2, "")
flags.DEFINE_float("max_scale", 0.05, "")
flags.DEFINE_integer("num_frames", 21, "")
flags.DEFINE_float("fps", 240.0, "")
flags.DEFINE_float("contrast_threshold_neg", 0.2, "")
flags.DEFINE_float("contrast_threshold_pos", 0.2, "")
flags.DEFINE_float("refractory_period", 0.0, "")
flags.DEFINE_string("gpu_ids", "0,1,2,3", "Comma-separated GPU device IDs to use")
flags.DEFINE_integer("batch_size", 10, "Number of images each process handles")

motion_types = ['translation_x', 'translation_y', 'rotation', 'radial']

def generate_sequence(img, motion, image_name, temp_root):
    temp_dir = join(temp_root, f"{motion}_temp", image_name)
    os.makedirs(temp_dir, exist_ok=True)
    h, w = img.shape[:2]
    center = (w // 2, h // 2)
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

def simulate_events(temp_dir, esim, motion, image_name, output_root):
    img_files = sorted(glob.glob(join(temp_dir, "*.png")))
    timestamps_ns = torch.arange(
        0, len(img_files) * 1e9 / FLAGS.fps, 1e9 / FLAGS.fps, dtype=torch.int64
    ).cuda()

    t, x, y, p = [], [], [], []
    for img_path, ts in zip(img_files, timestamps_ns):
        img = cv2.imread(img_path, cv2.IMREAD_GRAYSCALE)
        if img is None:
            continue
        log_img = torch.from_numpy(np.log(img.astype(np.float32) / 255 + 1e-5)).cuda()
        events = esim.forward(log_img, ts)
        if events is None:
            continue
        t.append(events['t'].cpu())
        x.append(events['x'].cpu())
        y.append(events['y'].cpu())
        p.append(events['p'].cpu())

    t = torch.cat(t)
    x = torch.cat(x)
    y = torch.cat(y)
    p = torch.cat(p)

    save_npz_dir = join(output_root, motion)
    os.makedirs(save_npz_dir, exist_ok=True)
    np.savez_compressed(join(save_npz_dir, image_name + ".npz"),
                        t=t.numpy(), x=x.numpy(), y=y.numpy(), p=p.numpy())

    H = y.max().item() + 1
    W = x.max().item() + 1
    vis_img = np.zeros((H, W), dtype=np.uint8)
    vis_img[y.numpy(), x.numpy()] = (p.numpy() * 255).astype(np.uint8)

    vis_dir = join(output_root, f"{motion}_vis")
    os.makedirs(vis_dir, exist_ok=True)
    vis_path = join(vis_dir, image_name + "_events.png")
    cv2.imwrite(vis_path, vis_img)

    shutil.rmtree(temp_dir)

def process_batch(args):
    image_batch, gpu_id = args
    os.environ["CUDA_VISIBLE_DEVICES"] = str(gpu_id)
    for image_path in image_batch:
        name = splitext(basename(image_path))[0]
        base_img = cv2.imread(image_path, cv2.IMREAD_GRAYSCALE)
        if base_img is None:
            continue
        for motion in motion_types:
            esim = esim_torch.esim_torch.EventSimulator_torch(
                FLAGS.contrast_threshold_neg,
                FLAGS.contrast_threshold_pos,
                FLAGS.refractory_period
            )
            temp_dir = generate_sequence(base_img, motion, name, FLAGS.output_root)
            simulate_events(temp_dir, esim, motion, name, FLAGS.output_root)

def batch_image_paths(image_paths, batch_size):
    for i in range(0, len(image_paths), batch_size):
        yield image_paths[i:i + batch_size]

def main(argv):
    image_paths = sorted(glob.glob(join(FLAGS.input_dir, "*.png")))
    gpu_id_list = [int(g.strip()) for g in FLAGS.gpu_ids.split(",") if g.strip()]
    batches = list(batch_image_paths(image_paths, FLAGS.batch_size))
    tasks = [(batch, gpu_id_list[i % len(gpu_id_list)]) for i, batch in enumerate(batches)]

    print(f"Launching {len(tasks)} batch-tasks using GPUs: {gpu_id_list}")
    with Pool(processes=len(gpu_id_list)) as pool:
        with tqdm(total=len(tasks), desc="Total Progress", ncols=100) as pbar:
            for _ in pool.imap_unordered(process_batch, tasks):
                pbar.update()

if __name__ == "__main__":
    app.run(main)
