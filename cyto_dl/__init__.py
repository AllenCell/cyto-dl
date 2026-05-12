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


# Register OmegaConf classes as safe globals for torch.load.
#
# PyTorch >= 2.6 flipped the default of `weights_only` in `torch.load` to
# True, which means the safe-unpickler rejects any non-allowlisted class.
# Lightning checkpoints saved by cyto-dl embed Hydra/OmegaConf hparams
# (DictConfig, ListConfig, ContainerMetadata, AnyNode, ...), so resuming
# training (`Trainer.fit(ckpt_path=...)`) or running
# `LightningModule.load_from_checkpoint(...)` would otherwise fail with:
#
#     UnpicklingError: Weights only load failed. ... Unsupported global:
#     GLOBAL omegaconf.listconfig.ListConfig was not an allowed global.
#
# OmegaConf containers are pure config holders, so allowlisting them is
# safe.
try:  # pragma: no cover - defensive: torch may be absent in some tooling
    import torch as _torch

    if hasattr(_torch.serialization, "add_safe_globals"):
        from omegaconf import DictConfig as _DictConfig
        from omegaconf import ListConfig as _ListConfig
        from omegaconf.base import ContainerMetadata as _ContainerMetadata
        from omegaconf.base import Metadata as _Metadata
        from omegaconf.nodes import AnyNode as _AnyNode
        from omegaconf.nodes import BooleanNode as _BooleanNode
        from omegaconf.nodes import FloatNode as _FloatNode
        from omegaconf.nodes import IntegerNode as _IntegerNode
        from omegaconf.nodes import StringNode as _StringNode

        _torch.serialization.add_safe_globals(
            [
                _DictConfig,
                _ListConfig,
                _ContainerMetadata,
                _Metadata,
                _AnyNode,
                _BooleanNode,
                _FloatNode,
                _IntegerNode,
                _StringNode,
            ]
        )
except ImportError:
    pass
