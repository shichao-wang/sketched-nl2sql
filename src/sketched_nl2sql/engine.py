import logging
from argparse import Namespace
from operator import attrgetter
from typing import Dict, List, Tuple

import torch
from torch import optim

from sketched_nl2sql.data.vocab import Vocab
from sketched_nl2sql.model import SketchedTextToSql
from sketched_nl2sql.modules.loss import QueryLoss

logger = logging.getLogger(__name__)


def move_to_device(data, device: str):
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


class Engine:
    """ engine manipulate model
    inspired by stanza's Trainer
    """

    def __init__(self, args: Namespace, vocab: Vocab, device: str = None):
        self.device = device or "cuda" if torch.cuda.is_available() else "cpu"
        self.model = SketchedTextToSql.from_args(args).to(self.device)
        self.args = args
        self.vocab = vocab
        self.optimizer = optim.Adam(filter(attrgetter("requires_grad"), self.model.parameters()))
        self.criterion = QueryLoss()

    def feed(self, batch_data, update: bool = True):
        """ train model on this dataset """
        if update:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        batch_data = move_to_device(batch_data, self.device)
        input_data, target = batch_data

        logits = self.model(*input_data)
        loss = self.criterion(logits, target)

        if update:
            loss.backward()
            # gradient clip
            self.optimizer.step()

        return loss.item()

    @classmethod
    def from_checkpoint(cls, checkpoint_file: str):
        """ reload to checkpoint """
        try:
            state_dict = torch.load(checkpoint_file, lambda storage, location: storage)
            engine = cls(state_dict["args"], state_dict["vocab"])
            engine.model.load_state_dict(state_dict["model"])
            return engine
        except BaseException as e:
            logger.error("cannot load model", e)
            exit(1)

    def save_checkpoint(self, checkpoint_file: str):
        """ save """
        state_dict = {"model": self.model.state_dict(), "vocab": self.vocab, "args": self.args}
        try:
            torch.save(state_dict, checkpoint_file)
        except BaseException as e:
            logger.warning("save failed", e)
