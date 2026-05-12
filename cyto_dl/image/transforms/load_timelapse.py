"""Custom transform for loading timelapse images with timepoint support."""

from typing import Dict, List, Optional, Union

import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform
from monai.utils import ensure_tuple
from omegaconf import ListConfig

from cyto_dl.image.io import MonaiBioReader


class LoadTimelapseImaged(MapTransform):
    """Load timelapse images with dynamic timepoint selection from data dict.

    Unlike MONAI's LoadImaged, this transform passes the full data dict to the reader so it can
    extract timepoint and other metadata.
    """

    def __init__(
        self,
        keys: KeysCollection,
        timepoint_key: str = "timepoint",
        channel: int = 0,
        dimension_order_out: str = "ZYX",
        scene_key: Optional[str] = None,
        path_key: Optional[str] = None,
        allow_missing_keys: bool = False,
    ):
        """
        Parameters
        ----------
        keys : KeysCollection
            Keys of the data dict to load images from.
        timepoint_key : str
            Key in data dict containing the timepoint index.
        channel : int
            Channel index to load.
        dimension_order_out : str
            Output dimension order (e.g., "ZYX", "YX").
        scene_key : Optional[str]
            Key in data dict containing the scene identifier.
        path_key : Optional[str]
            Key in data dict containing the file path to load from. If None,
            uses the same key as specified in `keys`. Useful when loading
            multiple channels from the same file into different output keys.
        allow_missing_keys : bool
            Whether to allow missing keys.
        """
        super().__init__(keys, allow_missing_keys)
        self.timepoint_key = timepoint_key
        self.channel = channel
        self.dimension_order_out = dimension_order_out
        self.scene_key = scene_key
        self.path_key = path_key

    def __call__(self, data: Dict) -> Dict:
        d = dict(data)

        # Get timepoint from data dict
        timepoint = d.get(self.timepoint_key, 0)
        if hasattr(timepoint, "item"):
            timepoint = timepoint.item()
        elif isinstance(timepoint, (list, tuple)):
            timepoint = timepoint[0]
        timepoint = int(timepoint)

        # Get scene if specified
        scene = None
        if self.scene_key and self.scene_key in d:
            scene = d[self.scene_key]

        # When path_key is specified, we load from path_key but store to the keys
        # This allows loading multiple channels from the same file into different output keys
        if self.path_key is not None:
            keys_to_process = list(ensure_tuple(self.keys))
        else:
            keys_to_process = list(self.key_iterator(d))

        for key in keys_to_process:
            # Use path_key if specified, otherwise use the same key
            source_key = self.path_key if self.path_key is not None else key
            filepath = d.get(source_key)

            if filepath is None:
                if self.allow_missing_keys:
                    continue
                raise KeyError(f"Path key '{source_key}' not found in data dict")

            # Handle if filepath is already loaded or is a list
            if isinstance(filepath, (list, tuple)):
                filepath = filepath[0]

            if not isinstance(filepath, str):
                # Already loaded, skip
                continue

            # Create reader with the specific timepoint
            reader = MonaiBioReader(
                dask_load=True,
                T=timepoint,
                C=self.channel,
                dimension_order_out=self.dimension_order_out,
            )

            # Read and get data
            img = reader.read(filepath)

            # Set scene if needed
            if scene is not None:
                img.set_scene(scene)
            elif img.scenes:
                img.set_scene(img.scenes[0])

            # Get the image data
            img_data, _ = reader.get_data(img)

            d[key] = img_data

        return d
