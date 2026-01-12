<div align="center">

<!-- <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/AllenCellModeling/cyto-dl/blob/b73e6f357727e3b42adea8540c86f2475ea60379/docs/CytoDL-logo-1C-onDark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/AllenCellModeling/cyto-dl/blob/b73e6f357727e3b42adea8540c86f2475ea60379/docs/CytoDL-logo-1C-onLight.png">
  <img src="https://github.com/AllenCellModeling/cyto-dl/blob/b73e6f357727e3b42adea8540c86f2475ea60379/docs/CytoDL-logo-1C-onLight.png">
</picture> -->

<h1>CytoDL</h1>

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/AllenCellModeling/cyto-dl/blob/acf7dad69f492c417b0e486f8f08c19f25575927/docs/CytoDL-overview_dark_1.png">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/AllenCellModeling/cyto-dl/blob/acf7dad69f492c417b0e486f8f08c19f25575927/docs/CytoDL-overview_light_1.png">
    <img src="https://github.com/AllenCellModeling/cyto-dl/blob/acf7dad69f492c417b0e486f8f08c19f25575927/docs/CytoDL-overview_light_1.png">
  </picture>
</p>

## Description

As part of the [Allen Institute for Cell Science's](https://allencell.org) mission to understand the principles by which human induced pluripotent stem cells establish and maintain robust dynamic localization of cellular structure, `CytoDL` aims to unify deep learning approaches for understanding 2D and 3D biological data as images, point clouds, and tabular data.

The bulk of `CytoDL`'s underlying structure bases the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) organization - we highly recommend that you familiarize yourself with their (short) docs for detailed instructions on running training, overrides, etc.

Our currently available code is roughly split into two domains: image-to-image transformations and representation learning. The image-to-image code (denoted im2im) contains configuration files detailing how to train and predict using models for resolution enhancement using conditional GANs (e.g. predicting 100x images from 20x images), semantic and instance segmentation, and label-free prediction. We also provide configs for Masked Autoencoder (MAE) and Joint Embedding Prediction Architecture ([JEPA](https://github.com/facebookresearch/jepa)) pretraining on 2D and 3D images using a Vision Transformer (ViT) backbone and for training segmentation decoders from these pretrained features. Representation learning code includes a wide variety of Variational Auto Encoder (VAE) architectures and contrastive learning methods such as [VICReg](https://github.com/facebookresearch/vicreg). Due to dependency issues, equivariant autoencoders are not currently supported on Windows.

As we rely on recent versions of pytorch, users wishing to train and run models on GPU hardware will need up-to-date NVIDIA drivers. Users with older GPUs should not expect code to work out of the box. Similarly, we do not currently support training/predicting on Mac GPUs. In most cases, cpu-based training should work when GPU training fails.

For im2im models, we provide a handful of example 3D images for training the basic image-to-image tranformation-type models and default model configuration files for users to become comfortable with the framework and prepare them for training and applying these models on their own data. Note that these default models are very small and train on heavily downsampled data in order to make tests run efficiently - for best performance, the model size should be increased and downsampling removed from the data configuration.

## Installation

### Prerequisites

- **Python**: 3.11 or 3.12
- **GPU (Optional but recommended)**: NVIDIA GPU with CUDA support
- **Package Manager**: We recommend [uv](https://github.com/astral-sh/uv) for faster installations

### Step 1: Clone the Repository

```bash
git clone https://github.com/AllenCellModeling/cyto-dl
cd cyto-dl
```

### Step 2: Create a Virtual Environment

**Option A: Using conda (recommended for Fortran dependencies)**

```bash
# Create environment with Python 3.11 and Fortran compiler
conda create -n cyto-dl python=3.11 fortran-compiler blas-devel -c conda-forge
conda activate cyto-dl
```

**Option B: Using venv**

```bash
python -m venv cyto-dl-env
# On Windows
cyto-dl-env\Scripts\activate
# On Linux/macOS
source cyto-dl-env/bin/activate
```

### Step 3: Install PyTorch

Choose the appropriate installation based on your hardware:

**GPU with CUDA 13.0 (Windows/Linux) - Recommended**

```bash
# Using uv (faster)
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130

# Or using pip
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
```

**GPU with CUDA 12.4 (Windows/Linux)**

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
```

**CPU Only (All platforms including macOS)**

```bash
uv pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
```

**macOS with Apple Silicon**

```bash
pip install torch torchvision
```

### Step 4: Install CytoDL and Dependencies

**Base installation:**

```bash
# Install remaining dependencies
pip install --no-deps -r requirements/requirements.txt

# Install CytoDL in editable mode
pip install -e .
```

**With all extras (includes equivariance, spherical harmonics, testing, docs):**

```bash
pip install --no-deps -r requirements/all-requirements.txt
pip install -e .
```

**With specific extras:**

```bash
# For equivariant models (not supported on Windows)
pip install --no-deps -r requirements/equiv-requirements.txt

# For spherical harmonics
pip install --no-deps -r requirements/spharm-requirements.txt

# For development and testing
pip install --no-deps -r requirements/test-requirements.txt
```

### Step 5: Download Example Data (Optional)

```bash
python scripts/download_test_data.py
```

### Verification

Verify your installation:

```bash
python -c "import cyto_dl; print(f'CytoDL version: {cyto_dl.__version__}')"
python -c "import torch; print(f'PyTorch version: {torch.__version__}'); print(f'CUDA available: {torch.cuda.is_available()}')"
```

### Troubleshooting

**Issue: `ImportError` for numpy or other packages**
- Try installing without `--no-deps`: `pip install -r requirements/requirements.txt`

**Issue: CUDA out of memory**
- Reduce batch size in your config: `datamodule.batch_size=8`
- Use mixed precision: `trainer.precision=16-mixed`

**Issue: Fortran compiler not found (for equivariant models)**
- Install via conda: `conda install -c conda-forge fortran-compiler`

### API

```python
from cyto_dl.api import CytoDLModel

model = CytoDLModel()
model.download_example_data()
model.load_default_experiment("segmentation", output_dir="./output", overrides=["trainer=cpu"])
model.print_config()
model.train()

# [OPTIONAL] async training
await model.train(run_async=True)
```

Most models work by passing data paths in the data config. For training or predicting on datasets that are already in memory, you can pass the data directly to the model. Note that this use case is primarily for programmatic use (e.g. in a workflow or a jupyter notebook), not through the normal CLI. An experiment showing a possible config setup for this use case is demonstrated with the [im2im/segmentation_array](configs/experiment/im2im/segmentation_array.yaml) experiment. For training, data must be passed as a dictionary with keys "train" and "val" containing lists of dictionaries with keys corresponding to the data config.

```python
from cyto_dl.api import CytoDLModel
import numpy as np

model = CytoDLModel()
model.load_default_experiment("segmentation_array", output_dir="./output")
model.print_config()

# create CZYX dummy data
data = {
    "train": [{"raw": np.random.randn(1, 40, 256, 256), "seg": np.ones((1, 40, 256, 256))}],
    "val": [{"raw": np.random.randn(1, 40, 256, 256), "seg": np.ones((1, 40, 256, 256))}],
}
model.train(data=data)
```

For predicting, data must be passed as a list of numpy arrays. The resulting predictions will be processed in a dictionary with one key for each task head in the model config and corresponding values in BC(Z)YX order.

```python
from cyto_dl.api import CytoDLModel
import numpy as np
from cyto_dl.utils import extract_array_predictions

model = CytoDLModel()
model.load_default_experiment(
    "segmentation_array", output_dir="./output", overrides=["data=im2im/numpy_dataloader_predict"]
)
model.print_config()

# create CZYX dummy data
data = [np.random.rand(1, 32, 64, 64), np.random.rand(1, 32, 64, 64)]

_, _, output = model.predict(data=data)
preds = extract_array_predictions(output)
```

## Workflow Examples

### Example 1: Training a Segmentation Model (CLI)

Train a 3D cell segmentation model on example data:

```bash
# Download example data first
python scripts/download_test_data.py

# Train on GPU with default config
python cyto_dl/train.py experiment=im2im/segmentation trainer=gpu

# Train on CPU with custom batch size
python cyto_dl/train.py experiment=im2im/segmentation trainer=cpu datamodule.batch_size=4

# Train with custom number of epochs and learning rate
python cyto_dl/train.py experiment=im2im/segmentation trainer=gpu trainer.max_epochs=50 model.optimizer.lr=0.0001
```

**What this does:**
- Loads the segmentation experiment configuration
- Trains a U-Net model for semantic segmentation
- Saves checkpoints to `logs/train/runs/<timestamp>/`
- Logs metrics to TensorBoard

**View training progress:**
```bash
tensorboard --logdir logs/train/runs/
```

### Example 2: Running Inference with a Trained Model (CLI)

Use a trained model to predict on new data:

```bash
# Predict using a trained checkpoint
python cyto_dl/predict.py \
  experiment=im2im/segmentation \
  trainer=gpu \
  ckpt_path=logs/train/runs/<timestamp>/checkpoints/best.ckpt \
  data.path=/path/to/your/images

# Predict on CPU
python cyto_dl/predict.py \
  experiment=im2im/segmentation \
  trainer=cpu \
  ckpt_path=logs/train/runs/<timestamp>/checkpoints/best.ckpt
```

**Output:**
- Predictions saved to `logs/predict/runs/<timestamp>/predictions/`

### Example 3: Complete Workflow with Python API

Full workflow from training to prediction using the Python API:

```python
from cyto_dl.api import CytoDLModel
from pathlib import Path

# 1. Initialize and download example data
model = CytoDLModel()
model.download_example_data()

# 2. Load experiment and configure
model.load_default_experiment(
    "segmentation",
    output_dir="./my_segmentation_project",
    overrides=[
        "trainer=gpu",
        "trainer.max_epochs=30",
        "datamodule.batch_size=8",
        "model.optimizer.lr=0.001"
    ]
)

# 3. Inspect configuration
model.print_config()

# 4. Train the model
model.train()

# 5. Get the best checkpoint path
checkpoint_dir = Path("my_segmentation_project/checkpoints")
best_ckpt = list(checkpoint_dir.glob("*best*.ckpt"))[0]

# 6. Run prediction on new data
model.load_default_experiment(
    "segmentation",
    output_dir="./predictions",
    overrides=[
        "trainer=gpu",
        f"ckpt_path={best_ckpt}",
        "data.path=/path/to/new/images"
    ]
)
predictions = model.predict()
```

### Example 4: Training on Custom Data (In-Memory)

Train directly on numpy arrays without saving to disk:

```python
from cyto_dl.api import CytoDLModel
import numpy as np
from pathlib import Path

# Prepare your data (CZYX format for 3D, CYX for 2D)
train_images = [
    np.random.randn(1, 40, 256, 256) for _ in range(10)  # 10 training images
]
train_masks = [
    (np.random.rand(1, 40, 256, 256) > 0.5).astype(float) for _ in range(10)
]

val_images = [
    np.random.randn(1, 40, 256, 256) for _ in range(3)  # 3 validation images
]
val_masks = [
    (np.random.rand(1, 40, 256, 256) > 0.5).astype(float) for _ in range(3)
]

# Format data for CytoDL
data = {
    "train": [
        {"raw": img, "seg": mask}
        for img, mask in zip(train_images, train_masks)
    ],
    "val": [
        {"raw": img, "seg": mask}
        for img, mask in zip(val_images, val_masks)
    ]
}

# Train model
model = CytoDLModel()
model.load_default_experiment(
    "segmentation_array",
    output_dir="./output",
    overrides=["trainer=gpu", "trainer.max_epochs=20"]
)
model.train(data=data)
```

### Example 5: Label-Free Prediction

Train a model to predict fluorescent labels from brightfield images:

```bash
# Train label-free prediction model
python cyto_dl/train.py \
  experiment=im2im/label_free \
  trainer=gpu \
  datamodule.batch_size=16 \
  trainer.max_epochs=100

# Predict on new brightfield images
python cyto_dl/predict.py \
  experiment=im2im/label_free \
  trainer=gpu \
  ckpt_path=logs/train/runs/<timestamp>/checkpoints/best.ckpt \
  data.path=/path/to/brightfield/images
```

### Example 6: Self-Supervised Pre-training with MAE

Pre-train a Vision Transformer backbone using Masked Autoencoder:

```bash
# Pre-train MAE on unlabeled 3D images
python cyto_dl/train.py \
  experiment=vit/mae_pretraining \
  trainer=gpu \
  trainer.max_epochs=200 \
  datamodule.batch_size=32

# Fine-tune on segmentation task
python cyto_dl/train.py \
  experiment=im2im/segmentation_with_mae \
  trainer=gpu \
  model.backbone.pretrained_path=logs/train/runs/<mae_timestamp>/checkpoints/best.ckpt
```

### Example 7: Working with Point Clouds

Process cell data as 3D point clouds:

```python
from cyto_dl.api import CytoDLModel

model = CytoDLModel()
model.load_default_experiment(
    "pcloud/autoencoder",
    output_dir="./pcloud_output",
    overrides=[
        "trainer=gpu",
        "trainer.max_epochs=100",
        "datamodule.batch_size=8"
    ]
)

# Train point cloud autoencoder
model.train()
```

### Example 8: Hyperparameter Tuning with Hydra

Use Hydra's multirun feature for hyperparameter search:

```bash
# Grid search over learning rates and batch sizes
python cyto_dl/train.py -m \
  experiment=im2im/segmentation \
  trainer=gpu \
  model.optimizer.lr=0.0001,0.001,0.01 \
  datamodule.batch_size=8,16,32
```

**This runs 9 experiments** (3 learning rates × 3 batch sizes)

### Example 9: Resume Training from Checkpoint

Continue training from a saved checkpoint:

```bash
# Resume training
python cyto_dl/train.py \
  experiment=im2im/segmentation \
  trainer=gpu \
  ckpt_path=logs/train/runs/<timestamp>/checkpoints/last.ckpt \
  trainer.max_epochs=100
```

### Example 10: Custom Data Loading Configuration

Override data paths and augmentation settings:

```bash
python cyto_dl/train.py \
  experiment=im2im/segmentation \
  trainer=gpu \
  data.path=/path/to/your/data \
  data.train_split=0.8 \
  data.augmentation.rotation=True \
  data.augmentation.flip=True \
  data.augmentation.noise=0.1
```

## Command Line Interface (CLI)

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
# Train on GPU
python cyto_dl/train.py experiment=im2im/experiment_name trainer=gpu

# Train on CPU
python cyto_dl/train.py experiment=im2im/experiment_name trainer=cpu
```

You can override any parameter from command line:

```bash
python cyto_dl/train.py trainer.max_epochs=20 datamodule.batch_size=64
```
