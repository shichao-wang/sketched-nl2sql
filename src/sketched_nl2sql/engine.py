import logging
from operator import attrgetter
from typing import List

import torch
from torch import optim
from torch.nn.utils import rnn

from sketched_nl2sql.dataset import find_value_position
from sketched_nl2sql.model import SketchedTextToSql
from sketched_nl2sql.modules.loss import QueryLoss
from torchnlp.config import Config
from torchnlp.engine import Engine as BaseEngine
from torchnlp.utils import move_to_device
from torchnlp.vocab import Vocab

logger = logging.getLogger(__name__)


class Engine(BaseEngine):
    """ engine manipulate model
    inspired by stanza's Trainer
    """

    CKPT_ATTRS = ["model", "optimizer", "vocab", "tokenizer"]

    def __init__(
        self, config: Config, tokenizer, vocab: Vocab, device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.vocab = vocab
        model = SketchedTextToSql.from_config(self.vocab, config)
        optimizer = optim.Adam(filter(attrgetter("requires_grad"), model.parameters()))
        criterion = QueryLoss()
        super().__init__(model, optimizer, criterion, device)

    def prepare_batch(self, batch_data):
        """ pre_process """
        # unpack
        (question, table_id, headers), (sel_id, agg_id, num_cond, cond_cols, cond_ops, cond_values) = batch_data
        # clean
        pass
        # tokenize
        question_tokens: List[List[str]] = [self.tokenizer.tokenize(q) for q in question]
        headers_tokens: List[List[List[str]]] = [[self.tokenizer.tokenize(h) for h in header] for header in headers]
        # find value position
        cond_value_starts, cond_value_ends = [], []
        values_tokens = [[self.tokenizer.tokenize(value) for value in values] for values in cond_values]

        for q_tokens, value_tokens in zip(question_tokens, values_tokens):
            value_starts, value_ends = [], []
            for v_tokens in value_tokens:
                if v_tokens:
                    start, end = find_value_position(q_tokens, v_tokens)
                    value_starts.append(start)
                    value_ends.append(end)
            cond_value_starts.append(value_starts)
            cond_value_ends.append(value_ends)

        # make input tensor
        question_tokens = rnn.pad_sequence(
            [torch.as_tensor(self.vocab.map2index(q_tokens), device=self.device) for q_tokens in question_tokens],
            batch_first=True,
            padding_value=self.vocab.pad_index,
        )
        num_headers = [len(headers) for headers in headers_tokens]  # used to split
        headers_tokens = rnn.pad_sequence(
            [
                torch.as_tensor(self.vocab.map2index(tokens), device=self.device)
                for h_tokens in headers_tokens
                for tokens in h_tokens
            ],
            batch_first=True,
            padding_value=self.vocab.pad_index,
        )
        # make target tensor
        agg_id = torch.as_tensor(agg_id, device=self.device)
        sel_id = torch.as_tensor(sel_id, device=self.device)
        num_cond = torch.as_tensor(num_cond, device=self.device)
        cond_cols = [torch.as_tensor(cols, device=self.device) for cols in cond_cols]
        cond_ops = [torch.as_tensor(ops, device=self.device) for ops in cond_ops if ops]
        cond_value_starts = [torch.as_tensor(starts, device=self.device) for starts in cond_value_starts if starts]
        cond_value_ends = [torch.as_tensor(ends, device=self.device) for ends in cond_value_ends if ends]

        return (
            (question_tokens, headers_tokens, num_headers),
            (sel_id, agg_id, num_cond, cond_cols, cond_ops, cond_value_starts, cond_value_ends),
            (table_id,),
        )

    def feed(self, batch_data, update: bool = True):
        """ train model on this dataset """
        (inputs, targets, table_id) = self.prepare_batch(batch_data)
        (question_tokens, headers_tokens, num_headers) = inputs
        (sel_id, agg_id, num_cond, cond_cols, cond_ops, cond_value_starts, cond_value_ends) = targets

        if update:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        logits = self.model(question_tokens, headers_tokens, num_headers)
        loss = self.criterion(logits, targets)

        if update:
            loss.backward()
            # gradient clip
            self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def predict(self, batch_data):
        self.model.eval()

        batch_data = move_to_device(batch_data, self.device)
        input_data, target = batch_data

        logits = self.model(*input_data)

    @classmethod
    def from_checkpoint(cls, config: Config, checkpoint_file: str):
        """ reload to checkpoint """
        try:
            state_dict = torch.load(checkpoint_file, lambda storage, location: storage)
            engine = cls(config, state_dict["tokenizer"], state_dict["vocab"])
            engine.model.load_state_dict(state_dict["model"])
            engine.optimizer.load_state_dict(state_dict["optimizer"])
            return engine
        except BaseException as e:
            logger.error("cannot load model", e)
            exit(1)
