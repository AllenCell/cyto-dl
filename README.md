<div align="center">

<!-- <picture>
  <source media="(prefers-color-scheme: dark)" srcset="https://github.com/AllenCell/cyto-dl/blob/b73e6f357727e3b42adea8540c86f2475ea60379/docs/CytoDL-logo-1C-onDark.png">
  <source media="(prefers-color-scheme: light)" srcset="https://github.com/AllenCell/cyto-dl/blob/b73e6f357727e3b42adea8540c86f2475ea60379/docs/CytoDL-logo-1C-onLight.png">
  <img src="https://github.com/AllenCell/cyto-dl/blob/b73e6f357727e3b42adea8540c86f2475ea60379/docs/CytoDL-logo-1C-onLight.png">
</picture> -->

<h1>CytoDL</h1>

<a href="https://pytorch.org/get-started/locally/"><img alt="PyTorch" src="https://img.shields.io/badge/PyTorch-ee4c2c?logo=pytorch&logoColor=white"></a>
<a href="https://pytorchlightning.ai/"><img alt="Lightning" src="https://img.shields.io/badge/-Lightning-792ee5?logo=pytorchlightning&logoColor=white"></a>
<a href="https://hydra.cc/"><img alt="Config: Hydra" src="https://img.shields.io/badge/Config-Hydra-89b8cd"></a>
<a href="https://github.com/ashleve/lightning-hydra-template"><img alt="Template" src="https://img.shields.io/badge/-Lightning--Hydra--Template-017F2F?style=flat&logo=github&labelColor=gray"></a><br>

</div>

<p align="center">
  <picture>
    <source media="(prefers-color-scheme: dark)" srcset="https://github.com/AllenCell/cyto-dl/blob/acf7dad69f492c417b0e486f8f08c19f25575927/docs/CytoDL-overview_dark_1.png">
    <source media="(prefers-color-scheme: light)" srcset="https://github.com/AllenCell/cyto-dl/blob/acf7dad69f492c417b0e486f8f08c19f25575927/docs/CytoDL-overview_light_1.png">
    <img src="https://github.com/AllenCell/cyto-dl/blob/acf7dad69f492c417b0e486f8f08c19f25575927/docs/CytoDL-overview_light_1.png">
  </picture>
</p>

## Description

As part of the [Allen Institute for Cell Science's](https://allencell.org) mission to understand the principles by which human induced pluripotent stem cells establish and maintain robust dynamic localization of cellular structure, `CytoDL` aims to unify deep learning approaches for understanding 2D and 3D biological data as images, point clouds, and tabular data.

The bulk of `CytoDL`'s underlying structure bases the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template) organization - we highly recommend that you familiarize yourself with their (short) docs for detailed instructions on running training, overrides, etc.

Our currently available code is roughly split into two domains: image-to-image transformations and representation learning. The image-to-image code (denoted im2im) contains configuration files detailing how to train and predict using models for resolution enhancement using conditional GANs (e.g. predicting 100x images from 20x images), semantic and instance segmentation, and label-free prediction. We also provide configs for Masked Autoencoder (MAE) and Joint Embedding Prediction Architecture ([JEPA](https://github.com/facebookresearch/jepa)) pretraining on 2D and 3D images using a Vision Transformer (ViT) backbone and for training segmentation decoders from these pretrained features. Representation learning code includes a wide variety of Variational Auto Encoder (VAE) architectures and contrastive learning methods such as [VICReg](https://github.com/facebookresearch/vicreg). Due to dependency issues, equivariant autoencoders are not currently supported on Windows.

As we rely on recent versions of pytorch, users wishing to train and run models on GPU hardware will need up-to-date NVIDIA drivers. Users with older GPUs should not expect code to work out of the box. Similarly, we do not currently support training/predicting on Mac GPUs. In most cases, cpu-based training should work when GPU training fails.

For im2im models, we provide a handful of example 3D images for training the basic image-to-image tranformation-type models and default model configuration files for users to become comfortable with the framework and prepare them for training and applying these models on their own data. Note that these default models are very small and train on heavily downsampled data in order to make tests run efficiently - for best performance, the model size should be increased and downsampling removed from the data configuration.

## How to run

Install dependencies.

```bash
# clone project
git clone https://github.com/AllenCell/cyto-dl
cd cyto-dl

# [OPTIONAL] create conda environment
conda create -n cyto-dl python=3.12
conda activate cyto-dl

# Install dependencies (see CUDA Installation section below for GPU support)
pip install -r requirements/requirements.txt

# [OPTIONAL] install extra dependencies - equivariance related
pip install -r requirements/equiv-requirements.txt

pip install -e .


#[OPTIONAL] if you want to use default experiments on example data
python scripts/download_test_data.py
```

### CUDA Installation

CytoDL supports GPU acceleration via PyTorch with CUDA. Choose the installation command that matches your CUDA version:

#### CUDA 13.0 (Latest)

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu130
pip install -r requirements/requirements.txt
pip install -e .
```

#### CUDA 12.4

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124
pip install -r requirements/requirements.txt
pip install -e .
```

#### CUDA 12.1

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu121
pip install -r requirements/requirements.txt
pip install -e .
```

#### CUDA 11.8

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu118
pip install -r requirements/requirements.txt
pip install -e .
```

#### CPU Only

```bash
pip install torch torchvision --index-url https://download.pytorch.org/whl/cpu
pip install -r requirements/requirements.txt
pip install -e .
```

> **Note:** Install PyTorch with the desired CUDA version *before* installing the requirements to ensure compatibility. You can check your CUDA version with `nvidia-smi` or `nvcc --version`.

> **Note:** No code changes are required in CytoDL for different CUDA versions. All CUDA interaction is handled through PyTorch's abstraction layer. Simply install the appropriate PyTorch build for your CUDA version.

### Windows Installation

Some packages (like `edt`) require C++ compilation and may fail to build on Windows. Use conda to install these packages before installing the requirements:

```bash
# Create and activate conda environment
conda create -n cyto-dl python=3.12
conda activate cyto-dl

# Install packages that require compilation via conda
conda install -c conda-forge edt

# Install PyTorch with CUDA support (adjust cu124 to your CUDA version)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Install remaining dependencies
pip install -r requirements/requirements.txt
pip install -e .
```

> **Note:** Due to dependency issues, equivariant autoencoders (`equiv` extras) and spherical harmonics (`spharm` extras) are not supported on Windows.

### Corporate Proxy / PyPI Mirror Issues

If you're behind a corporate proxy or using an internal PyPI mirror (e.g., Artifactory), you may encounter errors when installing PyTorch because CUDA-specific wheels are hosted on PyTorch's own index, not on pypi.org.

**Workaround:** Install PyTorch directly from the official source first, then install the remaining dependencies:

```bash
# Install PyTorch directly from pytorch.org (bypassing corporate mirror)
pip install torch torchvision --index-url https://download.pytorch.org/whl/cu124

# Then install the rest (will use your configured mirror)
pip install -r requirements/requirements.txt
pip install -e .
```

Alternatively, bypass the mirror entirely for the full installation:

```bash
pip install -r requirements/requirements.txt \
    --index-url https://pypi.org/simple/ \
    --extra-index-url https://download.pytorch.org/whl/cu124
```

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

Train model with chosen experiment configuration from [configs/experiment/](configs/experiment/)

```bash
#gpu
python cyto_dl/train.py experiment=im2im/experiment_name.yaml trainer=gpu

#cpu
python cyto_dl/train.py experiment=im2im/experiment_name.yaml trainer=cpu

```

You can override any parameter from command line like this

```bash
python cyto_dl/train.py trainer.max_epochs=20 datamodule.batch_size=64
```
