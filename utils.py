import random
from typing import List

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


def similar(token1: str, token2: str):
    token1, token2 = token1.lower(), token2.lower()
    return token1 in token2 or token2 in token1


def find_value_position(question_tokens: List[str], value_tokens: List[str]):
    length = len(value_tokens)
    for index in (index for index, t in enumerate(question_tokens) if similar(t, value_tokens[0])):
        if all(similar(q, v) for q, v in zip(question_tokens[index : index + length], value_tokens)):
            return index, index + length - 1
    raise ValueError(question_tokens + "\n" + value_tokens)
