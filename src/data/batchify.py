""" batchify """
from typing import Callable, Dict, List, NamedTuple, Tuple, TypeVar, Union

import torch
from torch import Tensor
from torch.nn.utils import rnn

T = TypeVar("T", Tuple, Dict, List)
Batchifier = Callable[[Union[T, List[Tensor]]], Union[T, Tensor]]


def skip() -> Batchifier:
    """ skip batchify """

    def _batchify(data: List):
        return data

    return _batchify


def arrays() -> Batchifier:
    """ simple stack tensors together """

    def _batchify(arrays_: List[Tensor]) -> Tensor:
        return torch.stack(arrays_)

    return _batchify


def sequences(padding_value: int = 0) -> Batchifier:
    """ pad sequences"""

    def _batchify(sequences_: List[Tensor]):
        return rnn.pad_sequence(sequences_, batch_first=True, padding_value=padding_value)

    return _batchify


def tuples(batchifier: Batchifier, *batchifiers: Batchifier) -> Batchifier:
    """ batchify tensors in tuples """
    if isinstance(batchifier, (List, Tuple)):
        assert len(batchifiers) == 0
        tuple_batchifiers = batchifier
    else:
        tuple_batchifiers = (batchifier, *batchifiers)

    def _batchify(tuple_list: List[Tuple[Tensor, ...]]):
        assert len(tuple_list[0]) == len(tuple_batchifiers)
        ret = []
        for index, func in enumerate(tuple_batchifiers):
            ret.append(func([t[index] for t in tuple_list]))
        return tuple(ret)

    return _batchify


def dicts(batchifiers: Dict[str, Batchifier], **named_batchifiers: Batchifier):
    """ batchify tensors in dicts """
    dict_batchifiers = {**batchifiers, **named_batchifiers}

    def _batchify(dicts_: List[Dict[str, Tensor]]):
        assert len(dicts_) == len(dict_batchifiers)
        return {key: batchifier(d[key] for d in dicts_) for key, batchifier in dict_batchifiers.items()}


def namedtuples(container, batchifiers: Dict[str, Batchifier]):
    """ batchify tensors in namedtuple """
    assert sorted(container._fields) == sorted(batchifiers.keys())

    def _batchify(named_tuples_: List[NamedTuple]):
        return container(
            **{key: batchifier(getattr(d, key) for d in named_tuples_) for key, batchifier in batchifiers.items()}
        )
