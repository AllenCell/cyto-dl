import os
import warnings
import tempfile
from pathlib import Path
from typing import Optional, Dict, Any, Union
from argparse import Namespace
import wandb
from omegaconf import OmegaConf
from pytorch_lightning.loggers import WandbLogger as Logger
from pytorch_lightning.callbacks import ModelCheckpoint
from pytorch_lightning.utilities.rank_zero import rank_zero_only
from cyto_dl import utils
import re 

log = utils.get_pylogger(__name__)

class WandBLogger(Logger):
    def __init__(self,
                 experiment_name: str = "lightning_logs",
                 run_name: Optional[str] = None,
                 config: Optional[Dict[str, Any]] = None,
                 tags: Optional[Dict[str, Any]] = None,
                 fault_tolerant: bool = False):
        super().__init__(project=experiment_name, name=run_name, config=config, resume="must", reinit=True)
        self.tags = tags
        self.fault_tolerant = fault_tolerant

        # Apply tags if provided
        if tags:
            self.experiment.tags = tags

    @property
    def experiment(self):
        return super().experiment

    @rank_zero_only
    def log_hyperparams(self, params: Union[Dict[str, Any], Namespace], mode="train") -> None:
        if hasattr(self.experiment, 'config'):
            self.experiment.config.update(params)

        # Save parameters to a YAML file in temp directory
        with tempfile.TemporaryDirectory() as tmp_dir:
            conf_path = Path(tmp_dir) / f"{mode}.yaml"
            config = OmegaConf.create(params)
            OmegaConf.save(config=config, f=conf_path)
            
            # Log YAML config as artifact
            self.experiment.save(str(conf_path))

    @rank_zero_only
    def log_metrics(self, metrics: Dict[str, Any], step: Optional[int] = None):
        # Ensure correct log step if required
        self.experiment.log({**metrics, 'step': step} if step is not None else metrics)

    @rank_zero_only
    def after_save_checkpoint(self, ckpt_callback: ModelCheckpoint):
        try:
            self._after_save_checkpoint(ckpt_callback)
        except Exception as e:
            if self.fault_tolerant:
                warnings.warn(
                    f"`WandBLogger.after_save_checkpoint` failed with exception {e}\n\nContinuing..."
                )
            else:
                raise e

    @rank_zero_only
    def _after_save_checkpoint(self, ckpt_callback: ModelCheckpoint) -> None:
        run = self.experiment
        artifact_path = "checkpoints"

        if ckpt_callback.monitor:
            artifact_path = f"checkpoints/{ckpt_callback.monitor}"

            try:
                artifact = run.use_artifact(f'{artifact_path}:latest', type='model')
                existing_ckpts = {artifact_file.name for artifact_file in artifact.manifest.entries.values()}
            except Exception:
                existing_ckpts = set()

            top_k_ckpts = {os.path.basename(path) for path in ckpt_callback.best_k_models.keys()}

            to_delete = existing_ckpts - top_k_ckpts
            to_upload = top_k_ckpts - existing_ckpts

            for ckpt in to_delete:
                checkpoint_path = os.path.join(ckpt_callback.dirpath, ckpt)
                if os.path.exists(checkpoint_path):
                    os.remove(checkpoint_path)
                    log.info(f"Deleted old checkpoint: {checkpoint_path}")
                else:
                    warnings.warn(f"Checkpoint {checkpoint_path} not found for deletion.")

            for ckpt in to_upload:
                local_checkpoint_path = os.path.join(ckpt_callback.dirpath, ckpt)
                log.info(f"Saving {ckpt} locally at: {local_checkpoint_path}")

                if os.path.exists(local_checkpoint_path):
                    log.info(f"Checkpoint {ckpt} saved locally at: {local_checkpoint_path}")

                self.log_metrics({"checkpoint_saved": os.path.basename(ckpt), "status": "uploaded locally"}, step=None)

            filepath = ckpt_callback.best_model_path
            local_best_path = Path(ckpt_callback.dirpath) / "best.ckpt"
            os.link(filepath, local_best_path)
            log.info(f"Best model checkpoint saved locally at: {local_best_path}")
            self.log_metrics({"best_model_checkpoint": str(local_best_path), "status": "saved locally"}, step=None)
            local_best_path.unlink()

        else:
            filepath = ckpt_callback.best_model_path
            if ckpt_callback.save_top_k == 1:
                last_path = Path(filepath).with_name("last.ckpt")
                os.link(filepath, last_path)

                log.info(f"Last model checkpoint saved locally at: {last_path}")

                self.log_metrics({"last_model_checkpoint": str(last_path), "status": "saved locally"}, step=None)
                last_path.unlink()
            else:
                log.info(f"Saving other checkpoints locally without W&B upload: {filepath}")

    @rank_zero_only
    def finalize(self, status: str):
        # Finish the WandB run
        self.experiment.finish()
