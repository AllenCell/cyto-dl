from pathlib import Path
from typing import List, Union

import tifffile
from lightning.pytorch.callbacks import Callback

VALID_STAGES = ("train", "val", "test", "predict")


class ImageSaver(Callback):
    def __init__(
        self,
        save_dir: Union[str, Path],
        save_every_n_epochs: int = 1,
        stages: List[str] = ["train", "val"],
        save_input: bool = False,
    ):
        """Callback for saving images after postprocessing by eads.

        Parameters
        ----------
        save_dir: Union[str, Path]
            Directory to save images
        save_every_n_epochs:int=1
            Frequency to save images
        stages:List[str]=["train", "val"]
            Stages to save images
        save_input:bool =False
            Whether to save input images
        """
        self.save_dir = Path(save_dir)
        for stage in stages:
            assert stage in VALID_STAGES, f"Invalid stage {stage}, must be one of {VALID_STAGES}"
        self.save_every_n_epochs = save_every_n_epochs
        self.stages = stages
        self.save_keys = ["pred", "target"]
        if save_input:
            self.save_keys.append("input")

    def _save(self, fn, data):
        fn.parent.mkdir(exist_ok=True, parents=True)
        tifffile.imwrite(fn, data)

    def on_predict_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if "predict" in self.stages:
            # Handle case where outputs is None
            if outputs is None:
                return

            # Handle case where outputs is a tuple (io_map, outputs)
            io_map = None
            if isinstance(outputs, tuple) and len(outputs) == 2:
                io_map, outputs = outputs

            # If outputs is None after unpacking, nothing to save
            if outputs is None:
                return

            # If io_map exists and is valid, use the original save path logic
            if io_map is not None:
                # If timepoint is available in the batch, inject it into save paths
                # so each timepoint gets its own file instead of being overwritten
                tp_suffix = None
                if isinstance(batch, dict) and "timepoint" in batch:
                    tp = batch["timepoint"]
                    if hasattr(tp, "item"):
                        tp = tp.item()
                    elif isinstance(tp, (list, tuple)):
                        tp = tp[0]
                    tp_suffix = f"_T{int(tp):03d}"

                for i, head_io_map in enumerate(io_map.values()):
                    for k, save_path in head_io_map.items():
                        if tp_suffix is not None:
                            save_path = Path(save_path)
                            save_path = (
                                save_path.parent / f"{save_path.stem}{tp_suffix}{save_path.suffix}"
                            )
                        self._save(save_path, outputs[k]["pred"][i])
            else:
                # io_map is None - use custom filename from batch metadata
                base_filename = self._extract_filename_from_batch(batch)

                suffix = None
                if isinstance(batch, dict) and "timepoint" in batch:
                    tp = batch["timepoint"]
                    if hasattr(tp, "item"):
                        tp = tp.item()
                    elif isinstance(tp, (list, tuple)):
                        tp = tp[0]
                    suffix = f"T{int(tp):03d}"

                # Add scene to suffix if available
                if isinstance(batch, dict) and "scene" in batch:
                    scene = batch["scene"]
                    if isinstance(scene, (list, tuple)):
                        scene = scene[0]
                    if suffix:
                        suffix = f"{scene}_{suffix}"
                    else:
                        suffix = str(scene)

                # Transform outputs from {head: {pred: data}} to {pred: {head: data}}
                # predict_step returns: {head_name: {"pred": data, "target": data, ...}}
                # save expects: {"pred": {head_name: data}, "target": {head_name: data}}
                transformed_outputs = {}
                for head_name, head_data in outputs.items():
                    if isinstance(head_data, dict):
                        for key, value in head_data.items():
                            if key not in transformed_outputs:
                                transformed_outputs[key] = {}
                            transformed_outputs[key][head_name] = value

                self.save(
                    transformed_outputs,
                    "predict",
                    batch_idx,
                    suffix=suffix,
                    base_filename=base_filename,
                )

    # train/test/val
    def save(self, outputs, stage=None, step=None, suffix=None, base_filename=None):
        for k in self.save_keys:
            # Skip if key doesn't exist in outputs
            if k not in outputs:
                continue
            for head in outputs[k]:
                # Skip if data is None (e.g., no target during prediction)
                data = outputs[k][head]
                if data is None:
                    continue

                if base_filename and suffix:
                    # Use original filename + timepoint
                    filename = f"{stage}_images/{base_filename}_{suffix}_{head}_{k}.tif"
                elif base_filename:
                    filename = f"{stage}_images/{base_filename}_{head}_{k}.tif"
                elif suffix:
                    filename = f"{stage}_images/{step}_{head}_{k}_{suffix}.tif"
                else:
                    filename = f"{stage}_images/{step}_{head}_{k}.tif"
                self._save(self.save_dir / filename, data)

    def _extract_filename_from_batch(self, batch):
        """Extract original filename from batch metadata."""
        if not isinstance(batch, dict):
            return None

        # Check for path column directly in batch (added via CSV columns)
        for key in ("path", "file", "filepath", "raw_path", "image_path", "filename"):
            if key in batch:
                val = batch[key]
                if isinstance(val, str):
                    return Path(val).stem
                elif isinstance(val, (list, tuple)) and len(val) > 0:
                    if isinstance(val[0], str):
                        return Path(val[0]).stem

        # Try to find from MONAI metadata if available
        for key in ("raw", "seg", "source", "input", "image"):
            meta_key = f"{key}_meta_dict"
            if meta_key in batch:
                meta = batch[meta_key]
                if isinstance(meta, dict):
                    for fname_key in ("filename_or_obj", "filename", "path"):
                        if fname_key in meta:
                            filepath = meta[fname_key]
                            if isinstance(filepath, (list, tuple)):
                                filepath = filepath[0]
                            if filepath:
                                return Path(str(filepath)).stem

        return None

    def on_test_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if "test" in self.stages:
            # Extract original filename from batch
            base_filename = self._extract_filename_from_batch(batch)

            # Extract timepoint from batch if available
            suffix = None
            if isinstance(batch, dict) and "timepoint" in batch:
                tp = batch["timepoint"]
                # Handle tensor or list of timepoints
                if hasattr(tp, "item"):
                    tp = tp.item()
                elif isinstance(tp, (list, tuple)):
                    tp = tp[0]
                suffix = f"T{int(tp):03d}"

            # Use original filename if available, otherwise fall back to batch_idx
            self.save(outputs, "test", batch_idx, suffix=suffix, base_filename=base_filename)

    def _should_save(self, batch_idx, epoch):
        return batch_idx == (epoch + 1) % self.save_every_n_epochs == 0

    def on_train_batch_end(self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0):
        if "train" in self.stages and self._should_save(batch_idx, trainer.current_epoch):
            self.save(outputs, "train", trainer.global_step)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if "val" in self.stages and self._should_save(batch_idx, trainer.current_epoch):
            self.save(outputs, "val", trainer.global_step)
