import logging
from os import path
from typing import Dict

import torch
from torch import nn
from torch.optim.optimizer import Optimizer
from typing_extensions import Protocol, runtime_checkable

logger = logging.getLogger(__name__)


class Engine:
    """ engine """

    CKPT_ATTRS = ["model", "optimizer"]

    def __init__(self, model: nn.Module, optimizer: Optimizer, criterion: nn.Module, device: str):
        self.model = model
        self.optimizer = optimizer
        self.criterion = criterion
        self.device = device

        self.model.to(self.device)
        self.__check_ckpt_attributes()

    def __check_ckpt_attributes(self):
        """ check ckpt attributes has load_state_dict state_dict """
        invalid_attrs = []
        for name in self.CKPT_ATTRS:
            attr = getattr(self, name, None)
            if attr is None:
                logger.warning(f"Redundant checkpoint attribute: {name}")
            elif isinstance(attr, PytorchStateMixin) or (
                hasattr(attr, "__getstate__") and hasattr(attr, "__setstate__"),
            ):
                continue
            else:
                invalid_attrs.append(name)
        if invalid_attrs:
            raise AttributeError(f"Attributes: {invalid_attrs} does not support `state_dict` or `load_state_dict`")

    def feed(self, input_batch, target_batch, update: bool = True):
        """ feed a batch of data to model """
        raise NotImplementedError()

    def predict(self, data):
        """ run inference for model """
        msg = f"method predict is not implemented but called"
        raise NotImplementedError(msg)

    def save_checkpoint(self, checkpoint_file: str):
        """ save checkpoint """
        checkpoint = {}
        for ckpt_attr in self.CKPT_ATTRS:
            state = getattr(self, ckpt_attr)
            if isinstance(state, PytorchStateMixin):
                state = state.state_dict()
                checkpoint[ckpt_attr] = state

        logger.info(f"Saving checkpoint to {path.abspath(checkpoint_file)}")
        torch.save(checkpoint, checkpoint_file)

    # @classmethod
    # def from_checkpoint(cls, checkpoint_file: str):
    #     """ load checkpoint to engine
    #     NOTICE this method requires
    #     """
    #     logger.info(f"Loading checkpoint from: {path.abspath(checkpoint_file)}")
    #     checkpoint = torch.load(checkpoint_file, lambda storage, location: storage)
    #     for ckpt_attr in cls.CKPT_ATTRS:
    #         setattr(self, ckpt_attr, checkpoint[ckpt_attr])


@runtime_checkable
class PytorchStateMixin(Protocol):
    """ pytorch state dict protocol"""

    def state_dict(self):
        """ return state dict of current object """
        raise NotImplementedError()

    def load_state_dict(self, state_dict: Dict):
        """ update state with given """
        raise NotImplementedError()
