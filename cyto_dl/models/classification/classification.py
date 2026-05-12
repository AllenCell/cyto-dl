import sys
from copy import deepcopy
from pathlib import Path

import numpy as np
import tifffile
import torch
import torch.nn as nn
from PIL import Image, ImageDraw, ImageFont
from skimage.exposure import rescale_intensity
from torchmetrics import MeanMetric
from torchmetrics.classification import MulticlassF1Score

from cyto_dl.models.base_model import BaseModel


class Classification(BaseModel):
    def __init__(
        self,
        *,
        model: nn.Module,
        x_key: str,
        num_classes: int,
        y_key: str = "label",
        save_dir="./",
        save_images_every_n_epochs=10,
        compile=False,
        write_batch_predictions=False,
        **base_kwargs,
    ):
        """
        Parameters
        ----------
        model: nn.Module
            model network, parameters are shared between task heads
        x_key: str
            key of input image in batch
        save_dir="./"
            directory to save images during training and validation
        save_images_every_n_epochs=1
            Frequency to save out images during training
        compile: False
            Whether to compile the model using torch.compile
        **base_kwargs:
            Additional arguments passed to BaseModel
        """
        _DEFAULT_METRICS = {
            "train/loss": MeanMetric(),
            "val/loss": MeanMetric(),
            "test/loss": MeanMetric(),
            "train/f1": MulticlassF1Score(num_classes=num_classes),
            "val/f1": MulticlassF1Score(num_classes=num_classes),
        }
        metrics = base_kwargs.pop("metrics", _DEFAULT_METRICS)
        super().__init__(metrics=metrics, **base_kwargs)

        self.automatic_optimization = True
        if compile is True and not sys.platform.startswith("win"):
            self.model = torch.compile(model)
        else:
            self.model = model

        self.loss_fn = nn.CrossEntropyLoss()

    def forward(self, x):
        return self.model(x)

    def should_save_image(self, batch_idx, stage):
        return (
            stage in ("test", "predict")
            or (self.current_epoch + 1) % self.hparams.save_images_every_n_epochs == batch_idx == 0
        )

    def save_images(self, batch, stage, logits, name=None):
        """Create image with prediction and label text overlaid on each image in batch."""
        pred = torch.argmax(logits, dim=1)

        if stage == "predict":
            label = torch.zeros_like(pred)
        else:
            label = batch[self.hparams.y_key].squeeze(0)
        # save labeled movie of prediction

        font = ImageFont.load_default()
        # Define text positions
        movie = []
        for p, l, frame in zip(pred, label, batch[self.hparams.x_key].cpu().numpy().squeeze()):
            # Convert the NumPy array to a PIL Image
            image = Image.fromarray(frame.astype(float))

            # Create a drawing context
            draw = ImageDraw.Draw(image)
            # Draw text on the image
            draw.text((0, 0), f"P:{p}", fill=2 * frame.max(), font=font)
            if stage != "predict":
                draw.text((0, 20), f"L:{int(l)}", fill=2 * frame.max(), font=font)

            # Convert the image back to a NumPy array
            movie.append(np.array(image))

        if name is None:
            name = self.current_epoch

        movie = rescale_intensity(np.stack(movie), out_range=(0, 255)).astype(np.uint8)
        save_path = Path(self.hparams.save_dir) / f"{stage}_images" / f"{name}.ome.tiff"
        save_path.parent.mkdir(exist_ok=True, parents=True)
        tifffile.imwrite(save_path, movie)

    def model_step(self, stage, batch, batch_idx):
        logits = self(batch[self.hparams.x_key]).squeeze(0)
        labels = batch[self.hparams.y_key].squeeze(0)
        if self.should_save_image(batch_idx, stage):
            self.save_images(batch, stage, logits)
        loss = self.loss_fn(logits, labels.long())
        return loss, logits.argmax(dim=1), labels

    def predict_step(self, batch, batch_idx):
        x = batch[self.hparams.anchor_key]
        logits = self(x).squeeze(0)
        preds = torch.argmax(logits, dim=1).cpu().numpy()
        return preds, deepcopy(x.meta)
