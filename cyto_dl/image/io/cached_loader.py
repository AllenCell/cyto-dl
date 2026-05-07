"""Cached S3/remote image loader for MONAI transform pipelines.

Drop-in replacement for `monai.transforms.LoadImaged` + `MonaiBioReader` that
caches downloaded volumes as .npy files on disk. Subsequent loads read from the
local cache (via mmap), eliminating redundant S3 reads across DataLoader workers
and epochs.

For zarr files (local or S3), loads directly via zarr+fsspec, bypassing BioImage.
For other formats, uses BioImage.
"""

import hashlib
import logging
import re
import time
from contextlib import suppress
from pathlib import Path
from typing import Dict, Hashable, Mapping, Optional, Union

import numpy as np
from monai.config import KeysCollection
from monai.transforms import MapTransform

logger = logging.getLogger(__name__)

# Pattern: https://<bucket>.s3.<region>.amazonaws.com/<key>
_S3_HTTPS_RE = re.compile(r"^https?://([^.]+)\.s3[.\-]([^.]+)?\.?amazonaws\.com/(.+)$")


def _https_to_s3(url: str) -> str:
    """Convert an HTTPS S3 URL to s3:// protocol.

    https://bucket.s3.region.amazonaws.com/key  ->  s3://bucket/key
    """
    m = _S3_HTTPS_RE.match(url)
    if m:
        bucket, _, key = m.groups()
        return f"s3://{bucket}/{key}"
    return url


def _is_zarr_path(path: str) -> bool:
    return path.rstrip("/").endswith(".zarr")


class CachedLoadImaged(MapTransform):
    """Load images with disk-based caching and retry logic.

    On first access for a given URL, the volume is downloaded and saved as a
    .npy file in `cache_dir`. On subsequent accesses (including from other
    DataLoader workers or future epochs), the cached .npy is loaded directly
    via numpy mmap -- no network I/O.

    For .zarr files (S3 or local), reads directly via zarr+fsspec.
    For other formats, uses BioImage.

    Channel selection is controlled by `channel` parameter or per-sample via
    a `C` column in the data dict (from CSV). When set, only that channel
    index is extracted, producing a single-channel (1, Z, Y, X) output.

    Parameters
    ----------
    keys : KeysCollection
        Keys in the input dict whose values are file paths / URLs to load.
    cache_dir : str or Path
        Directory for cached .npy files. Created if it does not exist.
    dimension_order_out : str
        Dimension order passed to ``BioImage.get_image_dask_data`` (non-zarr only).
    channel : int, optional
        Channel index to extract (0-based). If None, looks for a ``C`` column
        in the data dict. If neither is set, all channels are returned.
    max_retries : int
        Number of retry attempts for failed downloads.
    retry_delay : float
        Base delay in seconds between retries (doubled each attempt).
    allow_missing_keys : bool
        If True, skip keys that are not in the input dict.
    """

    def __init__(
        self,
        keys: KeysCollection,
        cache_dir: str,
        dimension_order_out: str = "CZYX",
        channel: Optional[int] = None,
        max_retries: int = 3,
        retry_delay: float = 1.0,
        allow_missing_keys: bool = False,
    ):
        super().__init__(keys, allow_missing_keys)
        self.cache_dir = Path(cache_dir)
        self.cache_dir.mkdir(parents=True, exist_ok=True)
        self.dimension_order_out = dimension_order_out
        self.channel = channel
        self.max_retries = max_retries
        self.retry_delay = retry_delay

    # ------------------------------------------------------------------
    # Cache helpers
    # ------------------------------------------------------------------

    @staticmethod
    def _cache_key(file_path: str, channel: Optional[int] = None) -> str:
        key_str = file_path if channel is None else f"{file_path}_c{channel}"
        # md5 used solely as a short, fast cache-key digest; not security-relevant.
        return hashlib.md5(key_str.encode(), usedforsecurity=False).hexdigest()[:16]

    def _cache_file(self, key: str) -> Path:
        return self.cache_dir / f"{key}.npy"

    def _lock_file(self, key: str) -> Path:
        return self.cache_dir / f"{key}.lock"

    # ------------------------------------------------------------------
    # Loading
    # ------------------------------------------------------------------

    def _load_zarr(self, file_path: str, channel: Optional[int] = None) -> np.ndarray:
        """Load volume from a zarr store (S3 or local) via zarr+fsspec."""
        import fsspec
        import zarr

        path = _https_to_s3(file_path) if file_path.startswith("http") else file_path

        if path.startswith("s3://"):
            store = fsspec.get_mapper(path, anon=True)
        else:
            store = str(path)

        z = zarr.open(store, mode="r")
        # OME-Zarr multiscale: use highest resolution array at key '0'
        arr = z["0"]
        # Expected shape: (T, C, Z, Y, X) — index directly to avoid loading all channels
        if channel is not None:
            data = arr[0, channel : channel + 1]  # (1, Z, Y, X) — only downloads 1 channel
        else:
            data = arr[0]  # (C, Z, Y, X)

        return np.ascontiguousarray(data)

    def _load_bioimage(self, file_path: str, channel: Optional[int] = None) -> np.ndarray:
        """Load volume via BioImage (non-zarr formats)."""
        from bioio import BioImage

        data = BioImage(file_path).get_image_dask_data(self.dimension_order_out).compute()
        if channel is not None:
            data = data[channel : channel + 1]
        return np.ascontiguousarray(data)

    def _load_from_source(self, file_path: str, channel: Optional[int] = None) -> np.ndarray:
        """Download volume from source with retries."""
        last_exc: Optional[Exception] = None
        delay = self.retry_delay
        use_zarr = _is_zarr_path(file_path)

        for attempt in range(1, self.max_retries + 1):
            try:
                if use_zarr:
                    return self._load_zarr(file_path, channel)
                else:
                    return self._load_bioimage(file_path, channel)
            except Exception as exc:
                last_exc = exc
                logger.warning(
                    "[CachedLoadImaged] Attempt %d/%d failed for %s: %s",
                    attempt,
                    self.max_retries,
                    file_path,
                    exc,
                )
                if attempt < self.max_retries:
                    time.sleep(delay)
                    delay *= 2

        raise RuntimeError(
            f"Failed to load {file_path} after {self.max_retries} attempts"
        ) from last_exc

    def _load_cached(self, file_path: str, channel: Optional[int] = None) -> np.ndarray:
        """Load volume, using disk cache when available."""
        key = self._cache_key(file_path, channel)
        cache_file = self._cache_file(key)

        # Fast path: already cached
        if cache_file.exists():
            try:
                return np.load(cache_file, mmap_mode="r").copy()
            except Exception as exc:
                logger.warning(
                    "[CachedLoadImaged] Corrupt cache for %s, re-downloading: %s",
                    file_path,
                    exc,
                )
                with suppress(OSError):
                    cache_file.unlink()

        # Download from source
        volume = self._load_from_source(file_path, channel)

        # Write to cache with file-based locking
        lock_file = self._lock_file(key)
        try:
            lock_file.touch(exist_ok=False)  # atomic lock acquisition
        except FileExistsError:
            # Another worker is writing — just return the downloaded volume
            return volume

        try:
            if not cache_file.exists():
                tmp_file = self.cache_dir / f"{key}.tmp.npy"
                np.save(tmp_file, volume)
                tmp_file.rename(cache_file)
        except Exception as exc:
            logger.warning("[CachedLoadImaged] Failed to cache %s: %s", file_path, exc)
        finally:
            with suppress(OSError):
                lock_file.unlink()

        return volume

    def _resolve_channel(self, data: Mapping) -> Optional[int]:
        """Resolve channel index from config or per-sample CSV column."""
        if self.channel is not None:
            return self.channel
        if "C" in data:
            try:
                return int(data["C"])
            except (ValueError, TypeError):
                return None
        return None

    # ------------------------------------------------------------------
    # MONAI MapTransform interface
    # ------------------------------------------------------------------

    def __call__(self, data: Mapping[Hashable, str]) -> Dict[Hashable, Union[np.ndarray, str]]:
        d = dict(data)
        channel = self._resolve_channel(d)
        for key in self.key_iterator(d):
            file_path = d[key]
            d[key] = self._load_cached(str(file_path), channel)
        return d
