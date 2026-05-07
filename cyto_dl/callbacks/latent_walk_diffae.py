from typing import Optional
from warnings import warn

import cv2
import numpy as np
import tifffile
import torch
from lightning.pytorch.callbacks import Callback
from sklearn.decomposition import PCA
from sklearn.pipeline import Pipeline
from sklearn.preprocessing import StandardScaler

from cyto_dl.models.im2im.utils.postprocessing import detach


class DiffAELatentWalk(Callback):
    def __init__(
        self,
        num_pcs: int = 8,
        n_steps: int = 10,
        sigma_range: Optional[int] = None,
        every_n_epoch: int = 1,
        n_noise_samples: int = 1,
        average: bool = True,
        batch_size: int = 3,
    ):
        """
        Parameters
        ----------
        num_pcs: int=8
            Number of principal components to use for latent walk
        n_steps: int=10
            Number of steps to traverse each PC in the latent walk
        sigma_range: Optional[int]=None
            Range to traverse each PC in the latent walk. If None, the min and max of the PC are used.
        every_n_epoch:int=1
            Frequency to perform latent walk
        n_noise_samples: int=1
            Number of noise samples to generate for each latent walk step
        average: bool=True
            Whether to average the generated images
        batch_size: int=3
            Batch size for generating images to prevent GPU OOM
        """
        self.num_pcs = num_pcs
        self.n_steps = n_steps
        self.sigma_range = int(sigma_range) if sigma_range is not None else None
        self.every_n_epoch = every_n_epoch
        self.n_noise_samples = n_noise_samples
        self.average = average
        self.batch_size = batch_size

        self.pca = Pipeline([("pca", PCA(n_components=num_pcs)), ("scaler", StandardScaler())])

        self.val_feats = []

    def _write_text(self, img, text):
        font = cv2.FONT_HERSHEY_SIMPLEX
        font_scale = 0.5
        color = tuple([img.max()] * 3)
        thickness = 1
        text_size = cv2.getTextSize(text, font, font_scale, thickness)[0]
        text_x = img.shape[1] - text_size[0] - 3  # 3 pixels from the right edge
        text_y = text_size[1] + 3  # 3 pixels from the top edge
        cv2.putText(img, text, (text_x, text_y), font, font_scale, color, thickness)
        return img

    def _write_pc_vals(self, walk_img, ranges):
        """Write PC index and value on image."""
        idx = 0
        for i, range_ in enumerate(ranges):
            for val in range_:
                walk_img[idx] = self._write_text(walk_img[idx], f"PC{i+1}:{val:.1f}")
                idx += 1
        return walk_img

    def _latent_walk(self, feats, model, save_path):
        # catch if only one batch for validation
        if len(feats.shape) == 1 or feats.shape[0] < self.num_pcs:
            warn(f"Insufficient data for latent walk with {self.num_pcs} PCs. Skipping...")
            return
        pca_data = self.pca.fit_transform(feats)
        print(f"Explained variance ratio: {self.pca['pca'].explained_variance_ratio_}")
        walk = []
        ranges = []
        for pc in np.arange(self.num_pcs):
            std = pca_data[:, pc].std()
            if self.sigma_range is None:
                min = pca_data[:, pc].min() / std
                max = pca_data[:, pc].max() / std
                range_ = np.linspace(min, max, self.n_steps)
            else:
                range_ = np.arange(-self.sigma_range, self.sigma_range + 0.01)
            print(f"PC{pc} range: {range_}")
            for i in range_:
                array = np.zeros(self.num_pcs)
                array[pc] = i * std
                walk.append(array)
            ranges.append(range_)
        walk = np.stack(walk).squeeze()
        walk = self.pca.inverse_transform(walk)
        walk = torch.from_numpy(walk).float().to(model.device)
        walk_img = model.generate_from_latent(
            walk,
            n_noise_samples=self.n_noise_samples,
            average=self.average,
            save=False,
            batch_size=self.batch_size,
        )
        # if vertically stack multi-channel generations
        walk_img = walk_img.reshape(walk_img.shape[0], -1, walk_img.shape[-1])
        walk_img = self._write_pc_vals(walk_img, ranges)
        tifffile.imwrite(save_path, walk_img)

    def on_validation_batch_end(
        self, trainer, pl_module, outputs, batch, batch_idx, dataloader_idx=0
    ):
        if (trainer.current_epoch + 1) % self.every_n_epoch == 0:
            latent_feat = detach(outputs[1].squeeze(-1))
            self.val_feats.append(latent_feat)

    def on_validation_epoch_end(self, trainer, pl_module):
        if (trainer.current_epoch + 1) % self.every_n_epoch == 0:
            # aggregate all latent features for PCA
            feats = np.concatenate(self.val_feats)
            self._latent_walk(
                feats,
                trainer.model,
                f"{pl_module.hparams.save_dir}/{trainer.current_epoch+1}_latent_walk.tiff",
            )

    def on_predict_epoch_end(self, trainer, pl_module):
        feats = np.concatenate([x[0] for x in trainer.predict_loop.predictions])
        self._latent_walk(feats, trainer.model, f"{pl_module.hparams.save_dir}/latent_walk.tiff")
