__version__ = "0.6.3"


# silence bio packages warnings
import logging
import warnings

logging.getLogger("ome_zarr").setLevel(logging.WARNING)
logging.getLogger("ome_zarr.reader").setLevel(logging.WARNING)
logging.getLogger("bfio.init").setLevel(logging.ERROR)
logging.getLogger("bfio.backends").setLevel(logging.ERROR)
logging.getLogger("xmlschema").setLevel(logging.ERROR)
warnings.filterwarnings("ignore")


# Default torch.load's `weights_only` to False for cyto-dl-shaped checkpoints.
#
# PyTorch >= 2.6 flipped the default of `weights_only` in `torch.load` to
# True, which means the safe-unpickler rejects any non-allowlisted class.
# Lightning checkpoints saved by cyto-dl embed Hydra-instantiable hparams
# (OmegaConf containers, typing.Any, and arbitrary user-targeted classes),
# so resuming training (`Trainer.fit(ckpt_path=...)`), evaluating
# (`Trainer.test(ckpt_path=...)`), or calling
# `LightningModule.load_from_checkpoint(...)` would otherwise fail with:
#
#     UnpicklingError: Weights only load failed. ... Unsupported global:
#     GLOBAL <something> was not an allowed global.
#
# Allowlisting class-by-class via `torch.serialization.add_safe_globals` is
# whack-a-mole because Hydra configs can target arbitrary user code. Cyto-dl
# users load their own checkpoints, so we honor the pre-2.6 default by
# wrapping `torch.load` to force `weights_only=False` whenever the caller
# either omits the argument or passes the sentinel `None` (e.g. Lightning's
# `lightning.fabric.utilities.cloud_io._load` forwards `weights_only=None`
# unconditionally on torch >= 2.6). Pass `weights_only=True` explicitly at
# the call site to opt back in.
import functools as _functools

import torch as _torch

if not getattr(_torch.load, "_cyto_dl_patched", False):
    _orig_torch_load = _torch.load

    @_functools.wraps(_orig_torch_load)
    def _cyto_dl_torch_load(*args, **kwargs):
        if kwargs.get("weights_only") is None:
            kwargs["weights_only"] = False
        return _orig_torch_load(*args, **kwargs)

    _cyto_dl_torch_load._cyto_dl_patched = True  # type: ignore[attr-defined]
    _torch.load = _cyto_dl_torch_load  # type: ignore[assignment]
