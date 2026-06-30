import random
import numpy as np
from pathlib import Path
from scipy.io import loadmat

import cv2
import torch
from functools import partial
import torchvision as thv
from torch.utils.data import Dataset

from utils import util_sisr
from utils import util_image
from utils import util_common

from basicsr.data.transforms import augment
from basicsr.data.realesrgan_dataset import RealESRGANDataset
from .ffhq_degradation_dataset import FFHQDegradationDataset
from .degradation_bsrgan.bsrgan_light import degradation_bsrgan_variant, degradation_bsrgan
from .masks import MixedMaskGenerator

class LamaDistortionTransform:
    def __init__(self, kwargs):
        import albumentations as A
        from .aug import IAAAffine2, IAAPerspective2
        out_size = kwargs.get('pch_size', 256)
        self.transform = A.Compose([
            A.SmallestMaxSize(max_size=out_size),
            IAAPerspective2(scale=(0.0, 0.06)),
            IAAAffine2(scale=(0.7, 1.3),
                       rotate=(-40, 40),
                       shear=(-0.1, 0.1)),
            A.PadIfNeeded(min_height=out_size, min_width=out_size),
            A.OpticalDistortion(),
            A.RandomCrop(height=out_size, width=out_size),
            A.HorizontalFlip(),
            A.CLAHE(),
            A.RandomBrightnessContrast(brightness_limit=0.2, contrast_limit=0.2),
            A.HueSaturationValue(hue_shift_limit=5, sat_shift_limit=30, val_shift_limit=5),
            A.Normalize(mean=kwargs.mean, std=kwargs.std, max_pixel_value=kwargs.max_value),
        ])

    def __call__(self, im):
        '''
        im: numpy array, h x w x c, [0,1]

        '''
        return self.transform(image=im)['image']

def get_transforms(transform_type, kwargs):
    '''
    Accepted optins in kwargs.
        mean: scaler or sequence, for nornmalization
        std: scaler or sequence, for nornmalization
        crop_size: int or sequence, random or center cropping
        scale, out_shape: for Bicubic
        min_max: tuple or list with length 2, for cliping
    '''
    if transform_type == 'default':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None), out_shape=kwargs.get('out_shape', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'bicubic_back_norm':
        transform = thv.transforms.Compose([
            util_sisr.Bicubic(scale=kwargs.get('scale', None)),
            util_sisr.Bicubic(scale=1/kwargs.get('scale', None)),
            util_image.Clamper(min_max=kwargs.get('min_max', (0.0, 1.0))),
            thv.transforms.ToTensor(),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'resize_ccrop_norm':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),
            # max edge resize if crop_size is int
            thv.transforms.Resize(size=kwargs.get('size', None)),
            thv.transforms.CenterCrop(size=kwargs.get('size', None)),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'rcrop_aug_norm':
        transform = thv.transforms.Compose([
            util_image.RandomCrop(pch_size=kwargs.get('pch_size', 256)),
            util_image.SpatialAug(
                only_hflip=kwargs.get('only_hflip', False),
                only_vflip=kwargs.get('only_vflip', False),
                only_hvflip=kwargs.get('only_hvflip', False),
                ),
            util_image.ToTensor(max_value=kwargs.get('max_value')),  # (ndarray, hwc) --> (Tensor, chw)
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'aug_norm':
        transform = thv.transforms.Compose([
            util_image.SpatialAug(
                only_hflip=kwargs.get('only_hflip', False),
                only_vflip=kwargs.get('only_vflip', False),
                only_hvflip=kwargs.get('only_hvflip', False),
                ),
            util_image.ToTensor(),   # hwc --> chw
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    elif transform_type == 'lama_distortions':
        transform = thv.transforms.Compose([
                LamaDistortionTransform(kwargs),
                util_image.ToTensor(max_value=1.0),   # hwc --> chw
            ])
    elif transform_type == 'rgb2gray':
        transform = thv.transforms.Compose([
            thv.transforms.ToTensor(),   # c x h x w, [0,1]
            thv.transforms.Grayscale(num_output_channels=kwargs.get('num_output_channels', 3)),
            thv.transforms.Normalize(mean=kwargs.get('mean', 0.5), std=kwargs.get('std', 0.5)),
        ])
    else:
        raise ValueError(f'Unexpected transform_variant {transform_variant}')
    return transform

def create_dataset(dataset_config):
    if dataset_config['type'] == 'gfpgan':
        dataset = FFHQDegradationDataset(dataset_config['params'])
    elif dataset_config['type'] == 'base':
        dataset = BaseData(**dataset_config['params'])
    elif dataset_config['type'] == 'bsrgan':
        dataset = BSRGANLightDeg(**dataset_config['params'])
    elif dataset_config['type'] == 'bsrganimagenet':
        dataset = BSRGANLightDegImageNet(**dataset_config['params'])
    elif dataset_config['type'] == 'realesrgan':
        dataset = RealESRGANDataset(dataset_config['params'])
    elif dataset_config['type'] == 'siddval':
        dataset = SIDDValData(**dataset_config['params'])
    elif dataset_config['type'] == 'inpainting':
        dataset = InpaintingDataSet(**dataset_config['params'])
    elif dataset_config['type'] == 'inpainting_val':
        dataset = InpaintingDataSetVal(**dataset_config['params'])
    elif dataset_config['type'] == 'deg_from_source':
        dataset = DegradedDataFromSource(**dataset_config['params'])
    elif dataset_config['type'] == 'bicubic':
        dataset = BicubicFromSource(**dataset_config['params'])
    elif dataset_config['type'] == 'paired':
        dataset = PairedData(**dataset_config['params'])
    elif dataset_config['type'] == 'paired_aligned':
        dataset = PairedDataAligned(**dataset_config['params'])
    else:
        raise NotImplementedError(dataset_config['type'])

    return dataset

class BaseData(Dataset):
    def __init__(
            self,
            dir_path,
            txt_path=None,
            transform_type='default',
            transform_kwargs={'mean':0.0, 'std':1.0},
            extra_dir_path=None,
            extra_transform_type=None,
            extra_transform_kwargs=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super().__init__()

        file_paths_all = []
        if dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_path, im_exts, recursive))
        if txt_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_path))

        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)

        self.extra_dir_path = extra_dir_path
        if extra_dir_path is not None:
            assert extra_transform_type is not None
            self.extra_transform = get_transforms(extra_transform_type, extra_transform_kwargs)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path_base = self.file_paths[index]
        im_base = util_image.imread(im_path_base, chn='rgb', dtype='float32')

        im_target = self.transform(im_base)
        out = {'image':im_target, 'lq':im_target}

        if self.extra_dir_path is not None:
            im_path_extra = Path(self.extra_dir_path) / Path(im_path_base).name
            im_extra = util_image.imread(im_path_extra, chn='rgb', dtype='float32')
            im_extra = self.extra_transform(im_extra)
            out['gt'] = im_extra

        if self.need_path:
            out['path'] = im_path_base

        return out

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class PairedData(Dataset):
    def __init__(
            self,
            dir_path,
            dir_path_extra,
            transform_type='default',
            transform_kwargs={'mean':0.5, 'std':0.5},
            pch_size=256,
            im_exts='png',
            length=None,
            recursive=False,
            need_path=False,
            event_dirs=None,
            is_val=False, # ADD
            ):
        super().__init__()

        file_paths_all = []
        if dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_path, im_exts, recursive))

        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)

        self.dir_path_extra = dir_path_extra
        if event_dirs is None:
            if is_val:  # validation directories
                self.event_dirs = {
                    'rotation': 'datasets/ITS/val/event_pair/rotation',
                    'radial': 'datasets/ITS/val/event_pair/radial',
                    'translation_x': 'datasets/ITS/val/event_pair/translation_x',
                    'translation_y': 'datasets/ITS/val/event_pair/translation_y',
                }
            else:  # training directories
                self.event_dirs = {
                    'rotation': 'datasets/ITS/train/event_pair/rotation',
                    'radial': 'datasets/ITS/train/event_pair/radial',
                    'translation_x': 'datasets/ITS/train/event_pair/translation_x',
                    'translation_y': 'datasets/ITS/train/event_pair/translation_y',
                }
        else:
            self.event_dirs = event_dirs

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path_base = self.file_paths[index]
        im_base = util_image.imread(im_path_base, chn='rgb', dtype='uint8')
        im_path_extra = Path(self.dir_path_extra) / Path(im_path_base).name
        im_extra = util_image.imread(im_path_extra, chn='rgb', dtype='uint8')

        # Add shape
        h_min = min(im_base.shape[0], im_extra.shape[0])
        w_min = min(im_base.shape[1], im_extra.shape[1])
        im_base = im_base[:h_min, :w_min, ...]
        im_extra = im_extra[:h_min, :w_min, ...]
        im_all = np.concatenate([im_base, im_extra], -1)

        # im_all = np.concatenate([im_base, im_extra], -1) Change

        im_all = self.transform(im_all)
        im_lq, im_gt = torch.chunk(im_all, 2, dim=0)

        out = {'lq':im_lq, 'gt':im_gt}
        
        ### Event data
        file_name = Path(im_path_base).stem
        # event_streams = [] # 稀疏事件点流 [t, x, y, p]
        event_channels = [] # 静态特征图
        H, W = im_lq.shape[-2:]

        # ⭐ Event 数据使用固定的 256x256 尺寸（不跟随 RGB 的 crop 尺寸）
        EVENT_H, EVENT_W = 256, 256

        ###### 单帧 event 尝试
        # for key, dir_path in self.event_dirs.items():
        #     npz_path = Path(dir_path) / f"{file_name}.npz"
        #     try:
        #         npz_data = np.load(npz_path)
        #         # 标准点流格式：[N,] 的 t, x, y, p
        #         if all(k in npz_data for k in ['t', 'x', 'y', 'p']):
        #             events = np.stack([npz_data['t'], npz_data['x'], npz_data['y'], npz_data['p']], axis=-1)  # → (N, 4)
        #             events = self._select_events_in_resolution(events, H, W)
        #             voxel = self._event_stream_to_voxel_grid(events, moments=1, resolution=(H, W))
        #             event_channels.append(voxel[0])  # (H, W)
        #         else:
        #             print(f"[Warning] Missing keys in {npz_path.name}, skipping")
        #     except Exception as e:
        #         print(f"[Warning] Failed to load event {key} from {npz_path}: {e}")

        ###### pyramid event 尝试
        pyramid_level = 3
        pyramid_moments = 2
        
        # ⭐ 检查是否使用预处理的合并数据（单个文件包含所有4个event类型）
        if 'preprocessed' in self.event_dirs:
            # 使用预处理的合并数据：单个 .npz 文件包含所有 event 类型
            npz_path = Path(self.event_dirs['preprocessed']) / f"{file_name}.npz"
            try:
                npz_data = np.load(npz_path)
                if 'voxel' in npz_data:
                    # ⭐ 预处理格式：voxel shape = (4, 2, H, W)
                    # 4 = num_bins (时间bins), 2 = polarities (正负极性)
                    event_voxel = npz_data['voxel']  # (4, 2, H_orig, W_orig)
                    
                    # ⭐ Reshape 到 (8, H, W) - 与 train_multimodal.py 的逻辑一致
                    event_voxel = event_voxel.reshape(-1, *event_voxel.shape[2:])  # (8, H, W)
                    event_tensor = torch.from_numpy(event_voxel).float()
                    
                    # ⭐ 调整尺寸到 Event 专用尺寸（256x256）
                    if event_tensor.shape[1:] != (EVENT_H, EVENT_W):
                        event_tensor = torch.nn.functional.interpolate(
                            event_tensor.unsqueeze(0),  # (1, 8, H, W)
                            size=(EVENT_H, EVENT_W),
                            mode='bilinear',
                            align_corners=False
                        ).squeeze(0)  # (8, EVENT_H, EVENT_W)
                    
                    # print(f"[DEBUG] Loaded preprocessed voxel: {event_tensor.shape} from {npz_path.name}")
                else:
                    print(f"[Warning] No 'voxel' key in {npz_path.name}")
                    event_tensor = torch.zeros(8, EVENT_H, EVENT_W)  # 8 通道
            except Exception as e:
                import traceback
                print(f"[Warning] Failed to load event preprocessed from {npz_path}: {e}")
                traceback.print_exc()
                event_tensor = torch.zeros(8, EVENT_H, EVENT_W)  # 8 通道
        else:
            # 使用原始的4个独立 event 类型目录
            for key, dir_path in self.event_dirs.items():
                npz_path = Path(dir_path) / f"{file_name}.npz"
                try:
                    npz_data = np.load(npz_path)
                    
                    # 原始事件流格式
                    if all(k in npz_data for k in ['t', 'x', 'y', 'p']):
                        events = np.stack([npz_data['t'], npz_data['x'], npz_data['y'], npz_data['p']], axis=-1)
                        # ⭐ 使用 Event 专用尺寸（256x256）
                        events = self._select_events_in_resolution(events, EVENT_H, EVENT_W)
                        voxel = self._event_stream_to_temporal_pyramid_representation(
                            events, pyramid_level, pyramid_moments, reduction_factor=1.5, resolution=(EVENT_H, EVENT_W)
                        )  # → (L, M, EVENT_H, EVENT_W)
                        voxel = voxel.reshape(-1, EVENT_H, EVENT_W)  # → (L*M, EVENT_H, EVENT_W)
                        event_channels.append(voxel)
                    else:
                        print(f"[Warning] Missing keys in {npz_path.name}, skipping")
                except Exception as e:
                    print(f"[Warning] Failed to load event {key} from {npz_path}: {e}")

            # 拼接所有 event 通道
            if event_channels:
                event_tensor = torch.from_numpy(np.concatenate(event_channels, axis=0)).float()  # → (C, EVENT_H, EVENT_W)
            else:
                event_tensor = torch.zeros(8, EVENT_H, EVENT_W)

        out['event'] = event_tensor

        if self.need_path:
            out['path'] = im_path_base

        return out

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

    def _event_stream_to_voxel_grid(self, events_stream, moments, resolution):
        """
        :param events_stream: The events stream.
        :param moments: The moments.
        :param resolution: The resolution of the voxel grid.
        :param positive: The positive value.
        :param negative: The negative value.
        :return: A list of voxel grid images. [M, H, W]
        """
        # The voxel grid is a list of i
        voxel_grid = np.zeros((moments, resolution[0], resolution[1]), dtype=np.float32)

        # The voxel grid is a 3D image.
        start_time = events_stream[:, 0].min()
        end_time = events_stream[:, 0].max()
        voxel_grid_time_step = (end_time - start_time) / moments
        for i in range(moments):
            left_time = start_time + i * voxel_grid_time_step
            right_time = start_time + (i + 1) * voxel_grid_time_step
            # The voxel grid in a moment is a 2D image.
            left_index = np.searchsorted(events_stream[:, 0], left_time, side="left")
            right_index = np.searchsorted(events_stream[:, 0], right_time, side="right")
            li, ri = left_index, right_index
            x, y, p = events_stream[li:ri, 1], events_stream[li:ri, 2], events_stream[li:ri, 3]
            x = x.astype(np.int32)
            y = y.astype(np.int32)
            voxel_grid[i] = self._render(x=x, y=y, p=p, shape=resolution)
        return voxel_grid

    # def _select_events_in_resolution(self, events):
    #     H, W = self.original_resolution
    #     if events.shape[0] == 0:
    #         return events
    #     # Remove the events outside the image
    #     x_max = np.max(events[:, 1])
    #     x_min = np.min(events[:, 1])
    #     y_max = np.max(events[:, 2])
    #     y_min = np.min(events[:, 2])
    #     if x_max >= W or x_min < 0:
    #         events = events[events[:, 1] < W]
    #     if y_max >= H or y_min < 0:
    #         events = events[events[:, 2] < H]
    #     return events

    def _select_events_in_resolution(self, events, H, W):
        
        if events.shape[0] == 0:
            return events
        x, y = events[:, 1], events[:, 2]
        valid = (x >= 0) & (x < W) & (y >= 0) & (y < H)
        return events[valid]

    # def _render(x, y, p, shape):
    #     # info(f"render: x:max:{x.max()}, min:{x.min()}, y:max:{y.max()}, min:{y.min()}, p:max:{p.max()}, min:{p.min()}")
    #     events = np.zeros(shape=shape, dtype=np.float32)
    #     events[y, x] = p
    #     return events

    def _render(self, x, y, p, shape):
        event_img = np.zeros(shape, dtype=np.float32)
        valid = (x >= 0) & (x < shape[1]) & (y >= 0) & (y < shape[0])
        x = x[valid].astype(np.int32)
        y = y[valid].astype(np.int32)
        p = p[valid]
        np.add.at(event_img, (y, x), p)
        return event_img

    def _event_stream_to_temporal_pyramid_representation(self,events, pyramid_level, pyramid_moments, reduction_factor, resolution):
        """
        events: [t,x,y,p], p = 0 or 1
        return [L,M,H,W]
        """
        H, W = resolution

        #
        event_time_pyramid_voxel = np.zeros(shape=[pyramid_level, pyramid_moments, H, W], dtype=np.float32)

        if events.shape[0] == 0:
            return event_time_pyramid_voxel
        #
        begin_time = events[:, 0].min()
        end_time = events[:, 0].max()
        during_time = end_time - begin_time

        time_start_end_list = [[0, 1]]
        for i in range(pyramid_level):
            l, r = time_start_end_list[-1]
            during = r - l
            deta = (during - during / reduction_factor) / 2
            l = l + deta
            r = r - deta
            time_start_end_list.append([l, r])

        # if DEBUG:
        #     info(f"begin_time: {begin_time}")
        #     info(f"end_time: {end_time}")
        #     info(f"during_time: {during_time}")
        #     info(f"time_start_end_list: {time_start_end_list}")

        for i in range(pyramid_level):
            l, r = time_start_end_list[i]
            l = l * during_time + begin_time
            r = r * during_time + begin_time
            moment_during_time = r - l

            for j in range(pyramid_moments):
                m_t_l = l + moment_during_time * j / pyramid_moments
                m_t_r = l + moment_during_time * (j + 1) / pyramid_moments
                left_index = np.searchsorted(events[:, 0], m_t_l, side="left")
                right_index = np.searchsorted(events[:, 0], m_t_r, side="right")
                li, ri = left_index, right_index

                x, y, p = events[li:ri, 1], events[li:ri, 2], events[li:ri, 3]
                x = x.astype(np.int32)
                y = y.astype(np.int32)
                event_voxel_grid = self._render(x=x, y=y, p=p, shape=resolution)
                event_time_pyramid_voxel[i, j] = event_voxel_grid
        return event_time_pyramid_voxel


class PairedDataAligned(Dataset):
    """
    Aligned paired (hazy, clear, event) dataset.

    保证 event 与 RGB 的空间对齐，避免事件边缘与图像边缘错位：

      1) 同一裁剪区域:
          先把 event resize 到 RGB 原图尺寸 (H_orig, W_orig)，再用同一组
          (i, j, pch) 同时 crop hazy / clear / event。

      2) 同步增广:
          采样一次 SpatialAug flag，对 RGB 与 event 同步应用同一 flag
          （hflip / rot 一致）。

    用法（YAML）:
        data:
          train:
            type: paired_aligned
            params:
              dir_path: <hazy>
              dir_path_extra: <clear>
              pch_size: 128
              event_dirs:
                preprocessed: <event_npz_dir>

    备注:
      - 当前实现只支持 ITS_v2 主用的 preprocessed 单文件 voxel 格式（npz['voxel']
        shape (4, 2, He, We)）。原始 4-key 事件流 (t, x, y, p) 暂不实现，因为
        在 ITS_v2 训练里没用到；如以后需要可对照 PairedData._event_stream_to_*
        系列方法补回。
      - 默认对齐方式：先把 event bilinear resize 到 (H_orig, W_orig)，再 crop。
        这与 PromptIR 修复版本（先 resize 到 RGB orig，再 RandomCrop.get_params
        共享坐标）一致。
    """

    def __init__(
        self,
        dir_path,
        dir_path_extra,
        pch_size=128,
        pch_sizes=None,
        mean=0.5,
        std=0.5,
        max_value=255,
        pass_aug=False,
        only_hflip=False,
        only_vflip=False,
        only_hvflip=False,
        im_exts='png',
        length=None,
        recursive=False,
        need_path=False,
        event_dirs=None,
        is_val=False,
        event_resize_mode='bilinear',
        event_channels=8,
        its_v2_naming=False,
        cache=False,
        tile_val=False,
    ):
        super().__init__()
        # ITS_v2 命名：hazy=1400_1_0.7.png -> clear=1400.png, event=1400.npz
        # （取 stem 中第一个 '_' 之前的部分作为 clear_id；与 PromptIR 的 its_v2_naming 一致）
        self.its_v2_naming = bool(its_v2_naming)

        file_paths_all = []
        if dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_path, im_exts, recursive))

        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all
        self.length = length
        self.need_path = bool(need_path)

        self.dir_path_extra = dir_path_extra
        self.pch_size = int(pch_size)
        self.pch_sizes = [int(x) for x in pch_sizes] if pch_sizes else None
        self.is_val = bool(is_val)
        self.mean = float(mean)
        self.std = float(std)
        self.max_value = float(max_value)
        self.pass_aug = bool(pass_aug)
        self.only_hflip = bool(only_hflip)
        self.only_vflip = bool(only_vflip)
        self.only_hvflip = bool(only_hvflip)
        self.event_resize_mode = event_resize_mode
        self.event_channels = int(event_channels)

        if event_dirs is None:
            if is_val:
                self.event_dirs = {
                    'rotation': 'datasets/ITS/val/event_pair/rotation',
                    'radial': 'datasets/ITS/val/event_pair/radial',
                    'translation_x': 'datasets/ITS/val/event_pair/translation_x',
                    'translation_y': 'datasets/ITS/val/event_pair/translation_y',
                }
            else:
                self.event_dirs = {
                    'rotation': 'datasets/ITS/train/event_pair/rotation',
                    'radial': 'datasets/ITS/train/event_pair/radial',
                    'translation_x': 'datasets/ITS/train/event_pair/translation_x',
                    'translation_y': 'datasets/ITS/train/event_pair/translation_y',
                }
        else:
            self.event_dirs = event_dirs

        # In-RAM cache of decoded full-res (im_base, im_extra, event) per file index.
        self.use_cache = bool(cache)
        self._cache = {}

        # Deterministic tiled-128 evaluation: expand each image into all 128x128
        # grid tiles (stride=pch, last tile flush to the edge for full coverage),
        # so the validation loop computes per-tile PSNR/SSIM/LPIPS then averages.
        self.tile_val = bool(tile_val)
        self._tiles = []
        if self.tile_val:
            for fidx in range(len(self.file_paths)):
                im_base, _, _ = self._load_full(fidx)
                H, W = im_base.shape[:2]
                for top in self._tile_starts(H, self.pch_size):
                    for left in self._tile_starts(W, self.pch_size):
                        self._tiles.append((fidx, top, left))

    @staticmethod
    def _tile_starts(size, tile):
        if size <= tile:
            return [0]
        starts = list(range(0, size - tile + 1, tile))
        if starts[-1] != size - tile:
            starts.append(size - tile)
        return starts

    def _load_full(self, index):
        """Load full-res (im_base uint8 HxWx3, im_extra uint8 HxWx3, event (C,H,W)).
        Cropped to the common min size; optionally cached in RAM."""
        if self.use_cache and index in self._cache:
            return self._cache[index]
        im_path_base = self.file_paths[index]
        im_base = util_image.imread(im_path_base, chn='rgb', dtype='uint8')
        clear_id = self._clear_id(im_path_base)
        im_path_extra = Path(self.dir_path_extra) / f"{clear_id}{Path(im_path_base).suffix}"
        im_extra = util_image.imread(im_path_extra, chn='rgb', dtype='uint8')
        # hazy 与 clear 尺寸可能不同 (如 SOTS hazy 620x460 vs clear 640x480),
        # 且 hazy 对应 clear 的中心区域 -> 必须居中裁剪对齐, 否则 tiled 评测整体错位.
        h_min = min(im_base.shape[0], im_extra.shape[0])
        w_min = min(im_base.shape[1], im_extra.shape[1])

        def _cc(im):
            h, w = im.shape[:2]
            t = max((h - h_min) // 2, 0)
            l = max((w - w_min) // 2, 0)
            return im[t:t + h_min, l:l + w_min, :]

        im_base = _cc(im_base)
        im_extra = _cc(im_extra)
        event_tensor = self._load_event_voxel(clear_id, h_min, w_min)
        result = (im_base, im_extra, event_tensor)
        if self.use_cache:
            self._cache[index] = result
        return result

    def __len__(self):
        if self.tile_val:
            return len(self._tiles)
        return len(self.file_paths)

    def _clear_id(self, hazy_path):
        """hazy 文件 -> 对应 clear/event 的 id（stem）。ITS_v2 取首个 '_' 前的部分。"""
        stem = Path(hazy_path).stem
        if not self.its_v2_naming:
            return stem
        import re as _re
        m = _re.match(r'^([^_]+)', stem)
        return m.group(1) if m else stem

    def reset_dataset(self):
        if self.length is None:
            return
        self.file_paths = random.sample(self.file_paths_all, self.length)

    def _sample_aug_flag(self):
        """Sample a SpatialAug flag in [0, 7]; respects pass_aug / only_* options."""
        if self.pass_aug:
            return 0
        if self.only_hflip:
            return random.choice([0, 5])
        if self.only_vflip:
            return random.choice([0, 1])
        if self.only_hvflip:
            return random.choice([0, 1, 5])
        return random.randint(0, 7)

    def _load_event_voxel(self, file_name, target_h, target_w):
        """
        Load preprocessed event voxel and resize to (target_h, target_w).
        Returns: torch.FloatTensor of shape (event_channels, target_h, target_w).
        """
        C = self.event_channels
        if 'preprocessed' in self.event_dirs:
            npz_path = Path(self.event_dirs['preprocessed']) / f"{file_name}.npz"
            event_tensor = None
            try:
                npz_data = np.load(npz_path)
                if 'voxel' in npz_data:
                    voxel = npz_data['voxel']  # (Bins, Polarity, He, We) — 通常 (4, 2, He, We)
                    voxel = voxel.reshape(-1, *voxel.shape[2:])  # → (C, He, We)
                    event_tensor = torch.from_numpy(voxel).float()
            except Exception as e:
                print(f"[PairedDataAligned] Failed to load preprocessed event {npz_path}: {e}")
            if event_tensor is None:
                event_tensor = torch.zeros(C, target_h, target_w)
        else:
            # Other event modes (raw event streams) are not implemented in the
            # aligned dataset on purpose; fall back to zeros and warn once.
            event_tensor = torch.zeros(C, target_h, target_w)

        # Resize event to RGB original resolution so it shares the same pixel coordinate
        # frame; this is the prerequisite for shared random-crop sampling below.
        if event_tensor.shape[1:] != (target_h, target_w):
            mode = self.event_resize_mode
            align_corners = False if mode in {'bilinear', 'bicubic'} else None
            event_tensor = torch.nn.functional.interpolate(
                event_tensor.unsqueeze(0),
                size=(target_h, target_w),
                mode=mode,
                align_corners=align_corners,
            ).squeeze(0)
        return event_tensor

    def _sample_pch(self):
        if self.pch_sizes and not self.is_val:
            return int(random.choice(self.pch_sizes))
        return self.pch_size

    def __getitem__(self, index):
        # ---------------------------------------------------------------- 0. tiled-128 val path
        if self.tile_val:
            return self._getitem_tile(index)

        # ---------------------------------------------------------------- 1. read RGB (+event), cached
        im_base, im_extra, event_tensor = self._load_full(index)
        im_base = im_base.copy()
        im_extra = im_extra.copy()
        im_path_base = self.file_paths[index]

        # ---------------------------------------------------------------- 3. pad if smaller than pch
        H, W = im_base.shape[:2]
        pch = self._sample_pch()
        if H < pch or W < pch:
            pad_h = max(0, pch - H)
            pad_w = max(0, pch - W)
            im_base = cv2.copyMakeBorder(im_base, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            im_extra = cv2.copyMakeBorder(im_extra, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            event_tensor = torch.nn.functional.pad(
                event_tensor.unsqueeze(0),
                (0, pad_w, 0, pad_h),
                mode='reflect',
            ).squeeze(0)
            H, W = im_base.shape[:2]

        # ---------------------------------------------------------------- 4. shared random crop
        i = random.randint(0, H - pch) if H > pch else 0
        j = random.randint(0, W - pch) if W > pch else 0
        im_base_pch = im_base[i:i + pch, j:j + pch, :]                 # (pch, pch, 3)
        im_extra_pch = im_extra[i:i + pch, j:j + pch, :]               # (pch, pch, 3)
        event_pch = event_tensor[:, i:i + pch, j:j + pch].contiguous()  # (C, pch, pch)

        # ---------------------------------------------------------------- 5. shared spatial aug
        flag = self._sample_aug_flag()
        if flag != 0:
            im_base_pch = util_image.data_aug_np(im_base_pch, flag)
            im_extra_pch = util_image.data_aug_np(im_extra_pch, flag)
            event_np = event_pch.numpy().transpose(1, 2, 0)                # (pch, pch, C)
            event_np = util_image.data_aug_np(event_np, flag)
            event_pch = torch.from_numpy(event_np.transpose(2, 0, 1)).contiguous()

        # ---------------------------------------------------------------- 6. RGB → [-1, 1]
        im_all = np.concatenate([im_base_pch, im_extra_pch], axis=-1)  # (pch, pch, 6) uint8
        im_all = util_image.ToTensor(max_value=self.max_value)(im_all)  # (6, pch, pch) [0, 1]
        im_all = (im_all - self.mean) / self.std                        # (6, pch, pch) [-1, 1]
        im_lq, im_gt = torch.chunk(im_all, 2, dim=0)

        out = {'lq': im_lq, 'gt': im_gt, 'event': event_pch}
        if self.need_path:
            out['path'] = im_path_base
        return out

    def _getitem_tile(self, index):
        """Deterministic tiled-128 sample for evaluation (no aug, fixed grid)."""
        fidx, top, left = self._tiles[index]
        im_base, im_extra, event_tensor = self._load_full(fidx)
        im_base = im_base.copy()
        im_extra = im_extra.copy()
        pch = self.pch_size

        H, W = im_base.shape[:2]
        if H < pch or W < pch:
            pad_h = max(0, pch - H)
            pad_w = max(0, pch - W)
            im_base = cv2.copyMakeBorder(im_base, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            im_extra = cv2.copyMakeBorder(im_extra, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)
            event_tensor = torch.nn.functional.pad(
                event_tensor.unsqueeze(0), (0, pad_w, 0, pad_h), mode='reflect',
            ).squeeze(0)

        im_base_pch = im_base[top:top + pch, left:left + pch, :]
        im_extra_pch = im_extra[top:top + pch, left:left + pch, :]
        event_pch = event_tensor[:, top:top + pch, left:left + pch].contiguous()

        im_all = np.concatenate([im_base_pch, im_extra_pch], axis=-1)   # (pch, pch, 6) uint8
        im_all = util_image.ToTensor(max_value=self.max_value)(im_all)  # (6, pch, pch) [0, 1]
        im_all = (im_all - self.mean) / self.std                        # [-1, 1]
        im_lq, im_gt = torch.chunk(im_all, 2, dim=0)

        out = {'lq': im_lq, 'gt': im_gt, 'event': event_pch}
        if self.need_path:
            out['path'] = self.file_paths[fidx]
        return out


class BSRGANLightDegImageNet(Dataset):
    def __init__(self,
                 dir_paths=None,
                 txt_file_path=None,
                 sf=4,
                 gt_size=256,
                 length=None,
                 need_path=False,
                 im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
                 mean=0.5,
                 std=0.5,
                 recursive=True,
                 degradation='bsrgan_light',
                 use_sharp=False,
                 rescale_gt=True,
                 ):
        super().__init__()
        file_paths_all = []
        if dir_paths is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_paths, im_exts, recursive))
        if txt_file_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_file_path))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.sf = sf
        self.length = length
        self.need_path = need_path
        self.mean = mean
        self.std = std
        self.rescale_gt = rescale_gt
        if rescale_gt:
            from albumentations import SmallestMaxSize
            self.smallest_rescaler = SmallestMaxSize(max_size=gt_size)

        self.gt_size = gt_size
        self.LR_size = int(gt_size / sf)

        if degradation == "bsrgan":
            self.degradation_process = partial(degradation_bsrgan, sf=sf, use_sharp=use_sharp)
        elif degradation == "bsrgan_light":
            self.degradation_process = partial(degradation_bsrgan_variant, sf=sf, use_sharp=use_sharp)
        else:
            raise ValueError(f'Except bsrgan or bsrgan_light for degradation, now is {degradation}')

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_hq = util_image.imread(im_path, chn='rgb', dtype='float32')

        h, w = im_hq.shape[:2]
        if h < self.gt_size or w < self.gt_size:
            pad_h = max(0, self.gt_size - h)
            pad_w = max(0, self.gt_size - w)
            im_hq = cv2.copyMakeBorder(im_hq, 0, pad_h, 0, pad_w, cv2.BORDER_REFLECT_101)

        if self.rescale_gt:
            im_hq = self.smallest_rescaler(image=im_hq)['image']

        im_hq = util_image.random_crop(im_hq, self.gt_size)

        # augmentation
        im_hq = util_image.data_aug_np(im_hq, random.randint(0,7))

        im_lq, im_hq = self.degradation_process(image=im_hq)
        im_lq = np.clip(im_lq, 0.0, 1.0)

        im_hq = torch.from_numpy((im_hq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        im_lq = torch.from_numpy((im_lq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        out_dict = {'lq':im_lq, 'gt':im_hq}

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

class BSRGANLightDeg(Dataset):
    def __init__(self,
                 dir_paths,
                 txt_file_path=None,
                 sf=4,
                 gt_size=256,
                 length=None,
                 need_path=False,
                 im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
                 mean=0.5,
                 std=0.5,
                 recursive=False,
                 resize_back=False,
                 use_sharp=False,
                 ):
        super().__init__()
        file_paths_all = util_common.scan_files_from_folder(dir_paths, im_exts, recursive)
        if txt_file_path is not None:
            file_paths_all.extend(util_common.readline_txt(txt_file_path))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all
        self.resize_back = resize_back

        self.sf = sf
        self.length = length
        self.need_path = need_path
        self.gt_size = gt_size
        self.mean = mean
        self.std = std
        self.use_sharp=use_sharp

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_hq = util_image.imread(im_path, chn='rgb', dtype='float32')

        # random crop
        im_hq = util_image.random_crop(im_hq, self.gt_size)

        # augmentation
        im_hq = util_image.data_aug_np(im_hq, random.randint(0,7))

        # degradation
        im_lq, im_hq = degradation_bsrgan_variant(im_hq, self.sf, use_sharp=self.use_sharp)
        if self.resize_back:
            im_lq = cv2.resize(im_lq, dsize=(self.gt_size,)*2, interpolation=cv2.INTER_CUBIC)
            im_lq = np.clip(im_lq, 0.0, 1.0)

        im_hq = torch.from_numpy((im_hq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        im_lq = torch.from_numpy((im_lq - self.mean) / self.std).type(torch.float32).permute(2,0,1)
        out_dict = {'lq':im_lq, 'gt':im_hq}

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

class SIDDValData(Dataset):
    def __init__(self, noisy_path, gt_path, mean=0.5, std=0.5):
        super().__init__()
        self.im_noisy_all = loadmat(noisy_path)['ValidationNoisyBlocksSrgb']
        self.im_gt_all = loadmat(gt_path)['ValidationGtBlocksSrgb']

        h, w, c = self.im_noisy_all.shape[2:]
        self.im_noisy_all = self.im_noisy_all.reshape([-1, h, w, c])
        self.im_gt_all = self.im_gt_all.reshape([-1, h, w, c])
        self.mean, self.std = mean, std

    def __len__(self):
        return self.im_noisy_all.shape[0]

    def __getitem__(self, index):
        im_gt = self.im_gt_all[index].astype(np.float32) / 255.
        im_noisy = self.im_noisy_all[index].astype(np.float32) / 255.

        im_gt = (im_gt - self.mean) / self.std
        im_noisy = (im_noisy - self.mean) / self.std

        im_gt = torch.from_numpy(im_gt.transpose((2, 0, 1)))
        im_noisy = torch.from_numpy(im_noisy.transpose((2, 0, 1)))

        return {'lq': im_noisy, 'gt': im_gt}

class InpaintingDataSet(Dataset):
    def __init__(
            self,
            dir_path,
            transform_type,
            transform_kwargs,
            mask_kwargs,
            txt_file_path=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super().__init__()

        file_paths_all = [] if txt_file_path is None else util_common.readline_txt(txt_file_path)
        if dir_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(dir_path, im_exts, recursive))
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.file_paths_all = file_paths_all

        self.mean = transform_kwargs.mean
        self.std = transform_kwargs.std
        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)
        self.mask_generator = MixedMaskGenerator(**mask_kwargs)
        self.iter_i = 0

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im = util_image.imread(im_path, chn='rgb', dtype='uint8')
        im = self.transform(im)        # c x h x w
        out_dict = {'gt':im, }

        mask = self.mask_generator(im, iter_i=self.iter_i)             # c x h x w, [0,1]
        self.iter_i += 1
        im_masked = im *  (1 - mask) - mask * (self.mean / self.std)   # mask area: -1
        out_dict['lq'] = im_masked
        out_dict['mask'] = (mask - self.mean) / self.std               # c x h x w, [-1,1]

        if self.need_path:
            out_dict['path'] = im_path

        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class InpaintingDataSetVal(Dataset):
    def __init__(
            self,
            lq_path,
            gt_path=None,
            mask_path=None,
            transform_type=None,
            transform_kwargs=None,
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        super().__init__()

        file_paths_all = util_common.scan_files_from_folder(lq_path, im_exts, recursive)
        self.file_paths_all = file_paths_all

        # lq image path
        self.file_paths = file_paths_all if length is None else random.sample(file_paths_all, length)
        self.gt_path = gt_path
        self.mask_path = mask_path

        self.length = length
        self.need_path = need_path
        self.transform = get_transforms(transform_type, transform_kwargs)

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_lq = util_image.imread(im_path, chn='rgb', dtype='float32')
        im_lq = self.transform(im_lq)
        out_dict = {'lq':im_lq}

        if self.need_path:
            out_dict['path'] = im_path

        # ground truth images
        if self.gt_path is not None:
            im_path = Path(self.gt_path) / Path(im_path).name
            im_gt = util_image.imread(im_path, chn='rgb', dtype='float32')
            im_gt = self.transform(im_gt)
            out_dict['gt'] = im_gt

        # image mask
        im_path = Path(self.mask_path) / Path(im_path).name
        im_mask = util_image.imread(im_path, chn='gray', dtype='float32')
        im_mask = self.transform(im_mask)
        out_dict['mask'] = im_mask        # -1 and 1

        return out_dict

    def reset_dataset(self):
        self.file_paths = random.sample(self.file_paths_all, self.length)

class DegradedDataFromSource(Dataset):
    def __init__(
            self,
            source_path,
            source_txt_path=None,
            degrade_kwargs=None,
            transform_type='default',
            transform_kwargs={'mean':0.0, 'std':1.0},
            length=None,
            need_path=False,
            im_exts=['png', 'jpg', 'jpeg', 'JPEG', 'bmp'],
            recursive=False,
            ):
        file_paths_all = []
        if source_path is not None:
            file_paths_all.extend(util_common.scan_files_from_folder(source_path, im_exts, recursive))
        if source_txt_path is not None:
            file_paths_all.extend(util_common.readline_txt(source_txt_path))
        self.file_paths_all = file_paths_all

        if length is None:
            self.file_paths = file_paths_all
        else:
            assert len(file_paths_all) >= length
            self.file_paths = random.sample(file_paths_all, length)

        self.length = length
        self.need_path = need_path

        self.transform = get_transforms(transform_type, transform_kwargs)
        self.degrade_kwargs = degrade_kwargs

    def __len__(self):
        return len(self.file_paths)

    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_source = util_image.imread(im_path, chn='rgb', dtype='float32')
        out = {'gt':self.gt_transform(im_source), 'lq':self.lq_transform(im_source)}

        if self.need_path:
            out['path'] = im_path

        return out

class BicubicFromSource(DegradedDataFromSource):
    def __getitem__(self, index):
        im_path = self.file_paths[index]
        im_gt = util_image.imread(im_path, chn='rgb', dtype='float32')

        if not hasattr(self, 'smallmax_resizer'):
            self.smallmax_resizer= util_image.SmallestMaxSize(
                    max_size = self.degrade_kwargs.get('gt_size', 256),
                    )
        if not hasattr(self, 'bicubic_transform'):
            self.bicubic_transform = util_image.Bicubic(
                scale=self.degrade_kwargs.get('scale', None),
                out_shape=self.degrade_kwargs.get('out_shape', None),
                activate_matlab=self.degrade_kwargs.get('activate_matlab', True),
                resize_back=self.degrade_kwargs.get('resize_back', False),
                )
        if not hasattr(self, 'random_cropper'):
            self.random_cropper = util_image.RandomCrop(
                pch_size=self.degrade_kwargs.get('pch_size', None),
                pass_crop=self.degrade_kwargs.get('pass_crop', False),
                )
        if not hasattr(self, 'paired_aug'):
            self.paired_aug = util_image.SpatialAug(
                    pass_aug = self.degrade_kwargs.get('pass_aug', False)
                    )

        im_gt = self.smallmax_resizer(im_gt)
        im_gt = self.random_cropper(im_gt)
        im_lq = self.bicubic_transform(im_gt)
        im_lq, im_gt = self.paired_aug([im_lq, im_gt])

        out = {'gt':self.transform(im_gt), 'lq':self.transform(im_lq)}

        if self.need_path:
            out['path'] = im_path

        return out
