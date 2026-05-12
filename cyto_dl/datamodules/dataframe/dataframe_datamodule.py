from typing import Dict, Optional, Sequence, Union

import numpy as np
import torch
from lightning import LightningDataModule
from lightning.pytorch.utilities.exceptions import MisconfigurationException
from monai.data import DataLoader
from upath import UPath as Path

from .utils import (
    get_canonical_split_name,
    make_multiple_dataframe_splits,
    make_single_dataframe_splits,
    parse_transforms,
)
from .worker_init import cloud_worker_init_fn


class DataframeDatamodule(LightningDataModule):
    """A pytorch lightning datamodule based on dataframes. It can either use a single dataframe
    file, which contains a column based on which a train- val- test split can be made; or it can
    use three dataframe files, one for each fold (train, val, test).

    Additionally, if it is only going to be used for prediction/testing, a flag
    `just_inference` can be set to True so the splits are ignored and the whole
    dataset is used.

    The `predict_datamodule` is simply constructed from the whole dataset,
    regardless of the value of `just_inference`.
    """

    def __init__(
        self,
        path: Union[Path, str],
        transforms: Dict,
        split_column: Optional[str] = None,
        columns: Optional[Sequence[str]] = None,
        split_map: Optional[Dict] = None,
        just_inference: bool = False,
        cache_dir: Optional[Union[Path, str]] = None,
        subsample: Optional[Dict] = None,
        refresh_subsample: bool = False,
        seed: int = 42,
        smartcache_args: Optional[Dict] = None,
        **dataloader_kwargs,
    ):
        """
        Parameters
        ----------
        path: Union[Path, str]
            Path to a dataframe file

        transforms: Dict
            Transforms specifications for each given split.

        split_column: Optional[str] = None
            Name of a column in the dataset which can be used to create train, val, test
            splits.

        columns: Optional[Sequence[str]] = None
            List of columns to load from the dataset, in case it's a parquet file.
            If None, load everything.

        split_map: Optional[Dict] = None
            TODO: document this argument

        just_inference: bool = False
            Whether this datamodule will be used for just inference
            (testing/prediction).
            If so, the splits are ignored and the whole dataset is used.

        cache_dir: Optional[Union[Path, str]] = None
            Path to a directory in which to store cached transformed inputs, to
            accelerate batch loading.

        subsample: Optional[Dict] = None
            Dictionary with a key per split ("train", "val", "test"), and the
            number of samples of each split to use per epoch. If `None` (default),
            use all the samples in each split per epoch.

        refresh_subsample: bool = False
            Whether to refresh subsample each time dataloader is called

        seed: int = 42
            random seed

        smartcache_args: Optional[Dict] = None
            Arguments to pass to SmartcacheDataset

        dataloader_kwargs:
            Additional keyword arguments are passed to the
            torch.utils.data.DataLoader class when instantiating it (aside from
            `shuffle` which is only used for the train dataloader).
            Among these args are `num_workers`, `batch_size`, `shuffle`, etc.
            See the PyTorch docs for more info on these args:
            https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader
        """

        super().__init__()
        torch.manual_seed(seed)
        self.seed = seed
        self.path = path
        self.cache_dir = cache_dir
        # if only one loader is specified, the same transforms are
        # used for all  folds
        transforms = parse_transforms(transforms)

        path = Path(path)

        # if path is a directory, we assume it is a directory containing multiple
        # dataframes - one per split. otherwise, we assume it is the path to a single
        # dataframe file, which is expected to have a
        if path.is_dir():
            self.datasets = make_multiple_dataframe_splits(
                path,
                transforms,
                columns,
                just_inference,
                cache_dir,
                smartcache_args=smartcache_args,
            )
        elif path.is_file():
            if split_column is None and not just_inference:
                raise MisconfigurationException(
                    "When using a single dataframe file, it must have a "
                    "split column, to use for train-val-test splitting."
                )

            self.datasets = make_single_dataframe_splits(
                path,
                transforms,
                split_column,
                columns,
                just_inference,
                split_map,
                cache_dir,
                smartcache_args=smartcache_args,
            )
        else:
            raise FileNotFoundError(f"Could not find specified dataframe path {path}")

        self.just_inference = just_inference
        self.dataloader_kwargs = dataloader_kwargs
        self.dataloaders = {}
        self.rng = np.random.default_rng(seed=seed)
        self.subsample = subsample or {}
        self.refresh_subsample = refresh_subsample

        self.batch_size = dataloader_kwargs.get("batch_size", 1)
        # init size is used to check if the batch size has changed (used for Automatic batch size finder)
        self._init_size = self.batch_size

        for key in list(self.subsample.keys()):
            self.subsample[get_canonical_split_name(key)] = self.subsample[key]

    def get_dataset(self, split):
        sample_size = self.subsample.get(split, -1)
        # always return a Subset
        sample = range(len(self.datasets[split]))
        if sample_size != -1:
            sample = self.rng.integers(len(self.datasets[split]), size=sample_size).tolist()
        # this doesn't affect performance because it returns a Subset,
        # which loads from the underlying dataset lazily
        return self.datasets[split][sample]

    def make_dataloader(self, split):
        kwargs = {**self.dataloader_kwargs}
        kwargs["shuffle"] = kwargs.get("shuffle", True) and split == "train"
        kwargs["batch_size"] = self.batch_size

        # Inject worker_init_fn to reset fsspec state for cloud data access
        if kwargs.get("num_workers", 0) > 0 and "worker_init_fn" not in kwargs:
            kwargs["worker_init_fn"] = cloud_worker_init_fn

        subset = self.get_dataset(split)
        return DataLoader(dataset=subset, **kwargs)

    def get_dataloader(self, split):
        sample_size = self.subsample.get(split, -1)

        if (
            (split not in self.dataloaders)
            or (sample_size != -1 and self.refresh_subsample)
            # check if batch size has changed (used for Automatic batch size finder)
            or (self._init_size != self.batch_size)
        ):
            # if we want to use a subsample per epoch, we need to remake the
            # dataloader, to refresh the sample
            self.dataloaders[split] = self.make_dataloader(split)
            # reset the init size to the current batch size so dataloader isn't recreated every epoch
            self._init_size = self.batch_size

        return self.dataloaders[split]

    def train_dataloader(self):
        if self.just_inference:
            raise TypeError(
                "This datamodule was configured with `just_inference=True`, "
                "so it doesn't have a train_dataloader and can't be "
                "used for training."
            )
        return self.get_dataloader("train")

    def val_dataloader(self):
        if self.just_inference:
            raise TypeError(
                "This datamodule was configured with `just_inference=True`, "
                "so it doesn't have a val_dataloader and can't be "
                "used for training."
            )

        return self.get_dataloader("val")

    def test_dataloader(self):
        split = "predict" if self.just_inference else "test"
        return self.get_dataloader(split)

    def predict_dataloader(self):
        return self.get_dataloader("predict")
