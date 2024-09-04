import logging
import os
import sys
import torch


def init_logging(log_file, stdout=False, loglevel=logging.INFO):
    formatter = logging.Formatter('%(asctime)s [%(levelname)s] %(module)s: %(message)s',
                                  datefmt='%m/%d/%Y %H:%M:%S')

    print('Making log output file: %s' % log_file)
    print(log_file[: log_file.rfind(os.sep)])
    if not os.path.exists(log_file[: log_file.rfind(os.sep)]):
        os.makedirs(log_file[: log_file.rfind(os.sep)])

    fh = logging.FileHandler(log_file)
    fh.setFormatter(formatter)
    fh.setLevel(loglevel)

    logger = logging.getLogger()
    logger.addHandler(fh)
    logger.setLevel(loglevel)

    if stdout:
        ch = logging.StreamHandler(sys.stdout)
        ch.setFormatter(formatter)
        ch.setLevel(logging.INFO)
        logger.addHandler(ch)

    return logger


def set_rnd(obj, seed: int) -> int:
    """
    Set seed or random state for all randomizable properties of obj.

    Args:
        obj: object to set seed or random state for.
        seed: set the random state with an integer seed.
    """
    if not hasattr(obj, "__dict__"):
        return seed  # no attribute
    if hasattr(obj, "set_random_state"):
        obj.set_random_state(seed=seed)
        return seed + 1  # a different seed for the next component
    for key in obj.__dict__:
        if key.startswith("__"):  # skip the private methods
            continue
        seed = set_rnd(obj.__dict__[key], seed=seed)
    return seed


def worker_init_fn(worker_id: int) -> None:
    """
    Callback function for PyTorch DataLoader `worker_init_fn`.
    It can set different random seed for the transforms in different workers.

    """
    worker_info = torch.utils.data.get_worker_info()
    set_rnd(worker_info.dataset, seed=worker_info.seed)
