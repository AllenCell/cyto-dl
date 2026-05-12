"""Worker init function to reset fsspec async state for safe S3 access in forked workers."""

import logging

logger = logging.getLogger(__name__)


def cloud_worker_init_fn(worker_id):
    """Reset fsspec async event loop state so each DataLoader worker gets fresh S3 connections.

    When PyTorch forks workers, the parent's asyncio event loop and thread references are copied
    but dead in the child. This function clears that state so fsspec/s3fs create new connections
    per worker.
    """
    try:
        import fsspec.asyn

        # Kill references to the parent's (now-dead) event loop and IO thread
        fsspec.asyn.iothread[0] = None
        fsspec.asyn.loop[0] = None
        fsspec.asyn.reset_lock()

        # Clear cached filesystem instances that carry stale state
        fsspec.filesystem("clear")  # noqa: safe to call even if no cache
    except Exception as exc:  # nosec B110 - best-effort fsspec reset; failures are non-fatal
        logger.debug("Worker %d: fsspec.asyn reset skipped (%s)", worker_id, exc)

    try:
        from fsspec.implementations.caching import CachingFileSystem

        CachingFileSystem._cache.clear()
    except Exception as exc:  # nosec B110 - best-effort cache reset; failures are non-fatal
        logger.debug("Worker %d: CachingFileSystem reset skipped (%s)", worker_id, exc)

    logger.debug("Worker %d: fsspec state reset", worker_id)
