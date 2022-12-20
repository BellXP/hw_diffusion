import atexit
import builtins
import functools
import logging
import sys


@functools.lru_cache(maxsize=None)
def _cached_log_stream(filename):
    io = open(filename, "a", buffering=8192)
    atexit.register(io.close)
    return io


def get_logger(name):
    """
    Retrieve the logger with the specified name or, if name is None, return a
    logger which is the root logger of the hierarchy.
    Args:
        name (string): name of the logger.
    """
    return logging.getLogger(name)


def setup_logging(save_path):
    """
    Sets up the logging for multiple processes. Only enable the logging for the
    master process, and suppress logging for the non-master processes.
    """
    logging.root.handlers = []
 
    logger = logging.getLogger()
    logger.setLevel(logging.DEBUG)
    logger.propagate = False
    print_plain_formatter = logging.Formatter(
        "[%(asctime)s]: %(message)s",
        datefmt="%m/%d %H:%M:%S",
    )
    fh_plain_formatter = logging.Formatter("%(message)s")

    ch = logging.StreamHandler(stream=sys.stdout)
    ch.setLevel(logging.DEBUG)
    ch.setFormatter(print_plain_formatter)
    logger.addHandler(ch)

    if save_path is not None:
        fh = logging.StreamHandler(_cached_log_stream(save_path))
        fh.setLevel(logging.DEBUG)
        fh.setFormatter(fh_plain_formatter)
        logger.addHandler(fh)