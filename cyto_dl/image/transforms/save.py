from pathlib import Path
from typing import Sequence, Union

import numpy as np
import tifffile
import torch
from monai.data.meta_tensor import MetaTensor
from monai.transforms import Transform
from omegaconf import ListConfig


class Save(Transform):
    def __init__(
        self,
        save_path: str = "./",
    ):
        """
        Parameters
        ----------
        keys: Sequence[str]
            keys to save
        save_path: str
            path to save images
        """
        super().__init__()
        self.save_path = Path(save_path)
        self.count = 0

    def __call__(self, img, name="img"):
        tifffile.imwrite(
            self.save_path / f"{name}_{self.count}.tif",
            img if not isinstance(img, (torch.Tensor, MetaTensor)) else img.numpy(),
        )
        self.count += 1
        return img


class Saved(Transform):
    """Save a batch of images to disk for debugging."""

    def __init__(
        self,
        keys: Sequence[str],
        save_path: str = "./",
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys: Sequence[str]
            keys to save
        save_path: str
            path to save images
        allow_missing_keys: bool
            allow missing keys in batch
        """
        super().__init__()
        self.keys = keys if isinstance(keys, (list, ListConfig)) else [keys]
        self.allow_missing_keys = allow_missing_keys
        self.saver = Save(save_path)

    def __call__(self, img_dict):
        for key in self.keys:
            if key in img_dict:
                self.saver(img_dict[key], key)
            elif not self.allow_missing_keys:
                raise ValueError(f"key {key} found in data. Available keys are {img_dict.keys()}")
        return img_dict
