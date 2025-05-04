import os
import json
from dataclasses import dataclass
from functools import cached_property
from io import BytesIO
from pathlib import Path
from typing import Literal

import numpy as np
import torch
import torchvision.transforms as tf
from einops import rearrange, repeat
from jaxtyping import Float, UInt8
from PIL import Image
from torch import Tensor
from torch.utils.data import IterableDataset

from ..geometry.projection import get_fov
from .dataset import DatasetCfgCommon
from .shims.augmentation_shim import apply_augmentation_shim
from .shims.crop_shim import apply_crop_shim
from .types import Stage
from .view_sampler import ViewSampler
from ..misc.cam_utils import camera_normalization

from .dataset_re10k import DatasetRE10kCfg

class DatasetScanNetpp(IterableDataset):
    cfg: DatasetRE10kCfg
    stage: Stage
    view_sampler: ViewSampler

    to_tensor: tf.ToTensor
    chunks: list[Path]
    near: float = 0.1
    far: float = 100.0

    def __init__(
        self,
        cfg: DatasetRE10kCfg,
        stage: Stage,
        view_sampler: ViewSampler,
    ) -> None:
        super().__init__()
        self.cfg = cfg
        self.stage = stage
        self.view_sampler = view_sampler
        self.to_tensor = tf.ToTensor()

        self.root = cfg.roots[0]
        if self.stage == 'train':
            split_file = 'splits/nvs_sem_train_v1.txt'
        else:
            split_file = 'splits/nvs_sem_val.txt'
        with open(self.root / split_file, 'r') as f:
            self.chunks = [self.root / line.strip() for line in f.readlines()]

    def shuffle(self, lst: list) -> list:
        indices = torch.randperm(len(lst))
        return [lst[x] for x in indices]

    def __iter__(self):
        # Chunks must be shuffled here (not inside __init__) for validation to show
        # random chunks.
        if self.stage in ("train", "val"):
            self.chunks = self.shuffle(self.chunks)

        # When testing, the data loaders alternate chunks.
        worker_info = torch.utils.data.get_worker_info()
        if self.stage == "test" and worker_info is not None:
            self.chunks = [
                chunk
                for chunk_index, chunk in enumerate(self.chunks)
                if chunk_index % worker_info.num_workers == worker_info.id
            ]

        for chunk_path in self.chunks:
            assert self.cfg.overfit_to_scene is None
            chunk = os.listdir(chunk_path / 'dslr' / 'rgb_resized_undistorted')

            if self.stage in ("train", "val"):
                chunk = self.shuffle(chunk)

            extrinsics = []
            intrinsics = []
            for image in chunk:
                basename = image.split('.')[0]
                meta = np.load(chunk_path / 'dslr' / 'camera' / f'{basename}.npz')
                extrinsics.append(meta['extrinsic'])
                intrinsics.append(meta['intrinsic'])

            extrinsics = torch.from_numpy(np.stack(extrinsics))
            intrinsics = torch.from_numpy(np.stack(intrinsics))

            try:
                context_indices, target_indices, overlap = self.view_sampler.sample(
                    chunk_path.name,
                    extrinsics,
                    intrinsics,
                )
            except ValueError:
                continue

            # Skip the example if the field of view is too wide.
            if (get_fov(intrinsics).rad2deg() > self.cfg.max_fov).any():
                continue

            # Load the images.
            try:
                context_images = [
                    chunk_path / 'dslr' / 'rgb_resized_undistorted' / chunk[index.item()] for index in context_indices
                ]
                context_images = self.convert_images(context_images)
                target_images = [
                    chunk_path / 'dslr' / 'rgb_resized_undistorted' / chunk[index.item()] for index in target_indices
                ]
                target_images = self.convert_images(target_images)
            except IndexError:
                continue
            except OSError:
                print(f"Skipped bad example {chunk_path.name}.")  # DL3DV-Full have some bad images
                continue

            # Resize the world to make the baseline 1.
            context_extrinsics = extrinsics[context_indices]
            if self.cfg.make_baseline_1:
                a, b = context_extrinsics[0, :3, 3], context_extrinsics[-1, :3, 3]
                scale = (a - b).norm()
                if scale < self.cfg.baseline_min or scale > self.cfg.baseline_max:
                    print(
                        f"Skipped {chunk_path.name} because of baseline out of range: "
                        f"{scale:.6f}"
                    )
                    continue
                extrinsics[:, :3, 3] /= scale
            else:
                scale = 1

            if self.cfg.relative_pose:
                extrinsics = camera_normalization(extrinsics[context_indices][0:1], extrinsics)

            example = {
                "context": {
                    "extrinsics": extrinsics[context_indices],
                    "intrinsics": intrinsics[context_indices],
                    "image": context_images,
                    "near": self.get_bound("near", len(context_indices)) / scale,
                    "far": self.get_bound("far", len(context_indices)) / scale,
                    "index": context_indices,
                    "overlap": overlap,
                },
                "target": {
                    "extrinsics": extrinsics[target_indices],
                    "intrinsics": intrinsics[target_indices],
                    "image": target_images,
                    "near": self.get_bound("near", len(target_indices)) / scale,
                    "far": self.get_bound("far", len(target_indices)) / scale,
                    "index": target_indices,
                },
                "scene": chunk_path.name,
            }
            if self.stage == "train" and self.cfg.augment:
                example = apply_augmentation_shim(example)
            yield apply_crop_shim(example, tuple(self.cfg.input_image_shape))

    def convert_poses(
        self,
        poses: Float[Tensor, "batch 18"],
    ) -> tuple[
        Float[Tensor, "batch 4 4"],  # extrinsics
        Float[Tensor, "batch 3 3"],  # intrinsics
    ]:
        b, _ = poses.shape

        # Convert the intrinsics to a 3x3 normalized K matrix.
        intrinsics = torch.eye(3, dtype=torch.float32)
        intrinsics = repeat(intrinsics, "h w -> b h w", b=b).clone()
        fx, fy, cx, cy = poses[:, :4].T
        intrinsics[:, 0, 0] = fx
        intrinsics[:, 1, 1] = fy
        intrinsics[:, 0, 2] = cx
        intrinsics[:, 1, 2] = cy

        # Convert the extrinsics to a 4x4 OpenCV-style W2C matrix.
        w2c = repeat(torch.eye(4, dtype=torch.float32), "h w -> b h w", b=b).clone()
        w2c[:, :3] = rearrange(poses[:, 6:], "b (h w) -> b h w", h=3, w=4)
        return w2c.inverse(), intrinsics

    def convert_images(
        self,
        images: list[UInt8[Tensor, "..."]],
    ) -> Float[Tensor, "batch 3 height width"]:
        torch_images = []
        for image in images:
            image = Image.open(image)
            torch_images.append(self.to_tensor(image))
        return torch.stack(torch_images)

    def get_bound(
        self,
        bound: Literal["near", "far"],
        num_views: int,
    ) -> Float[Tensor, " view"]:
        value = torch.tensor(getattr(self, bound), dtype=torch.float32)
        return repeat(value, "-> v", v=num_views)

    @property
    def data_stage(self) -> Stage:
        if self.cfg.overfit_to_scene is not None:
            return "test"
        if self.stage == "val":
            return "test"
        return self.stage

    @cached_property
    def index(self) -> dict[str, Path]:
        merged_index = {}
        data_stages = [self.data_stage]
        if self.cfg.overfit_to_scene is not None:
            data_stages = ("test", "train")
        for data_stage in data_stages:
            for root in self.cfg.roots:
                # Load the root's index.
                with (root / data_stage / "index.json").open("r") as f:
                    index = json.load(f)
                index = {k: Path(root / data_stage / v) for k, v in index.items()}

                # The constituent datasets should have unique keys.
                assert not (set(merged_index.keys()) & set(index.keys()))

                # Merge the root's index into the main index.
                merged_index = {**merged_index, **index}
        return merged_index

    # def __len__(self) -> int:
    #     return len(self.index.keys())
