import random
from typing import Dict, List, Tuple

import numpy
import torch


def set_random_seed(seed: int):
    """ set random state"""
    torch.manual_seed(seed)
    numpy.random.seed(seed)
    random.seed(seed)
    if torch.cuda.is_available():
        # noinspection PyUnresolvedReferences
        torch.cuda.manual_seed(seed)


def move_to_device(data, device: str):
    """ move data to device """
    if isinstance(data, torch.Tensor):
        return data.to(device)
    elif isinstance(data, List):
        return [move_to_device(d, device) for d in data]
    elif isinstance(data, Tuple):
        return tuple(move_to_device(d, device) for d in data)
    elif isinstance(data, Dict):
        return {key: move_to_device(value, device) for key, value in data.items()}
    else:
        raise ValueError("receives {}".format(type(data)))
