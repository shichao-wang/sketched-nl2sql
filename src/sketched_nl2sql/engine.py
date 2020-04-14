""" skectched nl2sql engine """
import logging
import sys
from operator import attrgetter
from typing import Tuple

import torch
from torch import Tensor, optim

from sketched_nl2sql.dataset import AGG_OPS, COND_OPS, Cond, Example, Query
from sketched_nl2sql.model import SketchedTextToSql
from sketched_nl2sql.modules.loss import QueryLoss
from torchnlp.config import Config
from torchnlp.engine import Engine as BaseEngine
from torchnlp.vocab import Vocab

logger = logging.getLogger(__name__)


class Engine(BaseEngine):
    """ engine manipulate model
    inspired by stanza's Trainer
    """

    CKPT_ATTRS = ["model", "optimizer", "vocab", "tokenizer"]

    def __init__(
        self,
        config: Config,
        tokenizer,
        vocab: Vocab,
        device: str = "cuda" if torch.cuda.is_available() else "cpu",
    ):
        self.config = config
        self.tokenizer = tokenizer
        self.vocab = vocab
        model = SketchedTextToSql.from_config(self.vocab, config)
        optimizer = optim.Adam(
            filter(attrgetter("requires_grad"), model.parameters()),
            lr=config.get("learning_rate", 1e-3),
        )
        criterion = QueryLoss()
        super().__init__(model, optimizer, criterion, device)

    def feed(self, input_batch, target_batch, update: bool = True):
        """ train model on this dataset """
        if update:
            self.model.train()
            self.optimizer.zero_grad()
        else:
            self.model.eval()

        logits = self.model(*input_batch)
        loss = self.criterion(logits, target_batch)

        if update:
            loss.backward()
            # gradient clip
            self.optimizer.step()

        return loss.item()

    @torch.no_grad()
    def predict(self, input_batch):
        """ predict """
        self.model.eval()
        header_lengths = input_batch[2]
        logits = self.model(*input_batch)  # type: Tuple[Tensor, ...]
        (
            sel_logits,
            agg_logits,
            where_num_logits,
            where_col_logits,
            where_op_logits,
            where_start_logits,  # (batch_size, num_headers, question_length)
            where_end_logits,
        ) = logits
        batch_size = len(header_lengths)
        pred_num = where_num_logits.argmax(dim=-1).tolist()

        pred_queries = []
        for b in range(batch_size):
            sel_col_id = sel_logits[b].argmax().item()
            cond_cols = where_col_logits[b].topk(pred_num[b])[1].tolist()
            conds = set()
            if cond_cols:
                cond_ops = where_op_logits[b, cond_cols].argmax(dim=1).tolist()
                cond_starts = (
                    where_start_logits[b, cond_cols].argmax(dim=1).tolist()
                )
                cond_ends = (
                    where_end_logits[b, cond_cols].argmax(dim=1).tolist()
                )
                conds = {
                    Cond(col, op, start, end)
                    for col, op, start, end in zip(
                        cond_cols, cond_ops, cond_starts, cond_ends
                    )
                }

            query = Query(
                sel_col_id, agg_logits[b, sel_col_id].argmax().item(), conds
            )
            pred_queries.append(query)

        return pred_queries

    def build_query_string(self, example: Example):
        """ build query string from example
            this method require tokenizer's decode method so I place it here
        """

        def column_name(col_id: int) -> str:
            """ column name """
            return f"col{col_id}"

        def table_name(table_id: str) -> str:
            """ table name """
            return f"table_{table_id.replace('-', '_')}"

        def build_condition(cond: Cond):
            """ where clause """
            value_string = self.tokenizer.convert_tokens_to_string(
                example.question_tokens[cond.value_start : cond.value_end + 1]
            )
            column_type = example.header.types[cond.col_id]
            if column_type == "text":
                value = f"'{value_string}'"
            elif column_type == "real":
                try:
                    value = float(value_string)
                except ValueError:
                    value = "''"
            else:
                raise ValueError()
            return "{col} {op} {value}".format(
                col=column_name(cond.col_id),
                op=COND_OPS[cond.op_id],
                value=value,
            )

        query_string = "SELECT {agg_op}({sel}) FROM {table_name}".format(
            agg_op=AGG_OPS[example.query.agg_id],
            sel=example.query.sel_col_id,
            table_name=table_name(example.header.table_id),
        )
        if example.query.conds:
            conds_string = [
                build_condition(cond) for cond in example.query.conds
            ]
            query_string += " WHERE " + " AND ".join(conds_string)
        return query_string

    def state_dict(self):
        """ state dict """
        return {
            "model": self.model.state_dict(),
            "optimizer": self.optimizer.state_dict(),
            "vocab": self.vocab.state_dict(),
            "tokenizer": self.tokenizer,
        }

    def save_checkpoint(self, checkpoint_file: str):
        """ save check point """
        torch.save(self.state_dict(), checkpoint_file)

    @classmethod
    def from_checkpoint(cls, config: Config, checkpoint_file: str):
        """ reload to checkpoint """
        try:
            state_dict = torch.load(
                checkpoint_file, lambda storage, location: storage
            )
            vocab = Vocab().load_state_dict(state_dict["vocab"])
            engine = cls(config, state_dict["tokenizer"], vocab)
            engine.model.load_state_dict(state_dict["model"])
            engine.optimizer.load_state_dict(state_dict["optimizer"])
            return engine
        except BaseException as ex:
            logger.error(f"Cannot Load Model", exec_info=ex)
            sys.exit(1)
