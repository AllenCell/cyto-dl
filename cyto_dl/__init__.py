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
