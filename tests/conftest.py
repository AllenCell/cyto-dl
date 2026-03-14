from typing import Generator

import pyrootutils
import pytest
import torch
from hydra import compose, initialize
from hydra.core.global_hydra import GlobalHydra
from monai.utils.enums import TraceKeys
from omegaconf import DictConfig, OmegaConf, open_dict

torch.serialization.add_safe_globals([TraceKeys])

from cyto_dl.utils.config import kv_to_dict
from scripts.download_test_data import delete_test_data, download_test_data

OmegaConf.register_new_resolver("kv_to_dict", kv_to_dict)
OmegaConf.register_new_resolver("eval", eval)

# Experiment configs to test
experiment_types = [
    "hiera",
    "mae",
    "ijepa",
    "iwm",
    "segmentation",
    "labelfree",
    "gan",
    "instance_seg",
]


@pytest.fixture(scope="package", params=experiment_types)
def cfg_train_global(request) -> DictConfig:
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="train.yaml",
            return_hydra_config=True,
            overrides=[
                f"experiment=im2im/{request.param}.yaml",
                "trainer=cpu.yaml",
            ],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_train_batches = 2
            cfg.trainer.limit_val_batches = 2
            cfg.trainer.limit_test_batches = 2
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None

    return cfg


@pytest.fixture(scope="package", params=experiment_types)
def cfg_eval_global(request) -> DictConfig:
    with initialize(version_base="1.2", config_path="../configs"):
        cfg = compose(
            config_name="eval.yaml",
            return_hydra_config=True,
            overrides=[
                "checkpoint.ckpt_path=.",
                f"experiment=im2im/{request.param}.yaml",
                "trainer=cpu.yaml",
            ],
        )

        # set defaults for all tests
        with open_dict(cfg):
            cfg.paths.root_dir = str(pyrootutils.find_root())
            cfg.trainer.max_epochs = 1
            cfg.trainer.limit_test_batches = 2
            cfg.trainer.accelerator = "cpu"
            cfg.trainer.devices = 1
            cfg.data.num_workers = 0
            cfg.data.pin_memory = False
            cfg.extras.print_config = False
            cfg.extras.enforce_tags = False
            cfg.logger = None
    return cfg


# this is called by each test which uses `cfg_train` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_train(cfg_train_global, tmp_path) -> Generator[DictConfig, None, None]:
    cfg = cfg_train_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.data.cache_dir = str(tmp_path / "cache")

    yield cfg

    GlobalHydra.instance().clear()


# this is called by each test which uses `cfg_eval` arg
# each test generates its own temporary logging path
@pytest.fixture(scope="function")
def cfg_eval(cfg_eval_global, tmp_path) -> Generator[DictConfig, None, None]:
    cfg = cfg_eval_global.copy()

    with open_dict(cfg):
        cfg.paths.output_dir = str(tmp_path)
        cfg.paths.log_dir = str(tmp_path)
        cfg.data.cache_dir = str(tmp_path / "cache")

    yield cfg

    GlobalHydra.instance().clear()


def pytest_sessionstart(session):
    download_test_data()


def pytest_sessionfinish(session, exitstatus):
    delete_test_data()
    return exitstatus
