import logging
from argparse import Namespace
from operator import attrgetter

import torch
from torch import optim

from sketched_nl2sql.data.vocab import Vocab
from sketched_nl2sql.modules.loss import QueryLoss
from sketched_nl2sql.model import SketchedTextToSql

logger = logging.getLogger(__name__)


class Engine:
    """ engine manipulate model
    inspired by stanza's Trainer
    """

    def __init__(self, args: Namespace, vocab: Vocab):
        self.model = SketchedTextToSql.from_args(args)
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
