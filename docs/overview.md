# Overview

`cyto_dl` aims to implement common image-to-image transformations in a manner that is easy to use for beginners and flexible for those wanting to change parameters and customize the configs for their specific needs. To do so, `CytoDl` is based on the [lightning-hydra-template](https://github.com/ashleve/lightning-hydra-template), which combines the config-based flexibility of [hydra](https://hydra.cc/) and the convenience and multiple levels of abstraction provided by [pytorch-lightning](https://www.pytorchlightning.ai/). Training, testing, and prediction configs and syntax are identical to those in the lightning-hydra-template repo; as such their docs are the best way to become familiar with the mechanics of using this repo. Below are documented the unique features of this repo. If a type of config or topic is not covered here, it is because it is already documented in the `lightning-hydra-template` repo.
We provide simple example configs to help users get started. These configs rely heavily on OmegaConf's [variable interpolation](https://omegaconf.readthedocs.io/en/2.3_branch/usage.html#variable-interpolation) to keep parameters consistent across configs and limit the complexity exposed to new users.
An `_aux` section can be provided in any config. This can be used to store reference values for interpolation. Any parameters in this section will not be instantiated.

## data

The provided datamodule is the serotiny [DataFrame Datamodule](https://github.com/AllenCell/serotiny/blob/32b3811fc1ef013a191e34181b0add1ca145663d/serotiny/datamodules/dataframe/dataframe_datamodule.py) wraps MONAI's [persistent dataset](https://docs.monai.io/en/stable/data.html#persistentdataset).
The data config specifies how to construct `train`, `valid`, `test`, and `predict` dataloaders.
Some important arguments are :

- `path`: references a folder with `train.csv`, `test.csv`, and `valid.csv` or to a single `.csv` file with a `split` column
- `cache_dir`: references a location to save cached images to speed up training
- `split_column`: if a single `.csv` is passed to `path`, `split_column` is used to divide the manifest into `train`/`test`/`valid` splits.
- `transforms`: sequence of transforms that load, preprocess, and augment images for training, testing, and validation. [Monai](https://docs.monai.io/en/stable/transforms.html) offers a wide array of useful transforms. Often, the selected transforms are the primary differentiators of different tasks. For example, `labelfree` and `segmentation` differ in part due to how the target is normalized (z-score normalized for `labelfree` vs binarized for `segmentation`). When creating your own configs, the transforms are the first thing to change/
- All additional arguments are assumed to be arguments for the dataloaders, e.g. `num_worker`, `batch_size`, `pin_memory`, etc. See [here](https://pytorch.org/docs/stable/data.html#torch.utils.data.DataLoader) for dataloader arguments.

## experiment

This is the workhorse of `cyto_dl` training. These files compile and override defaults for the `data`, `model`, `callbacks`, `trainer`, and `logger` configs.
If you are using the example configs, `source_col` and `target_col` are expected to refer to the column names of your input and ground truth images in the provided `.csv` files. `experiment_name` and `run_name` are used to organize the saved log files and log to mlflow. `path` and `cache_dir` arguments passed to the `datamodule` config are also required.

## model

The model arrangement that we provide consists of two basic parts:

1. backbone
   The backbone is fed the input images and produces an $n$ channel output. This output is then passed to each of the `task` heads.
2. `task` heads
   `task` heads transform the backbone output into task-specific outputs. For each `task`, `head` and `loss` keys must be provided.

- The `head` function determines how the backbone output is transformed into the task-specific output. If used in a single task case, this can be the [identity head](https://github.com/AllenCell/cyto-dl/blob/b19dd56da4adfbaca658dd7ad6128a1bfe42b721/cyto_dl/models/components/aux_head.py#L12), which simply passes through the backbone output. In multi task cases, these can be more complex, doing [convolution or further upsampling](https://github.com/AllenCell/cyto-dl/blob/b19dd56da4adfbaca658dd7ad6128a1bfe42b721/cyto_dl/models/components/aux_head.py#L66), [3d to 2d projection](https://github.com/AllenCell/cyto-dl/blob/b19dd56da4adfbaca658dd7ad6128a1bfe42b721/cyto_dl/models/components/aux_head.py#L31), or any other task that you can implement in a custom head.
- The `loss` head determines which loss function is run on the output of a given task head. [MONAI](https://docs.monai.io/en/stable/losses.html#loss-functions) and [pytorch](https://docs.monai.io/en/stable/losses.html#loss-functions) provide many common loss functions, and we implement custom loss functions.

The `generator` `optimizer` and `lr_scheduler` must always be passed, and GANs requires a `discriminator` `optimizer` as well.

`postprocessing` specifies how images are saved out for visualization during training. For an example segmentation task where the `raw` columns is passed to the model and `seg` head predicts logits, we get

```
postprocessing:
  input:
    raw:
      _target_: cyto_dl.model.utils.postprocessing.rescale
      _partial_: True
    seg:
      _target_: cyto_dl.model.utils.postprocessing.rescale
      _partial_: True
  prediction:
    seg:
      _target_: cyto_dl.model.utils.postprocessing.sigmoid_rescale
      _partial_: True
```

This tells the model to save out the `raw` and `seg` images passed in by rescaling them to 8-bit range, as well as applying `sigmoid` to the segmentation prediction and rescaling it to 8-bit. Custom postprocessing code is encouraged for your use case, but a few defaults can be found [here](https://github.com/AllenCell/cyto-dl/blob/b19dd56da4adfbaca658dd7ad6128a1bfe42b721/cyto_dl/utils/postprocessing.py)
