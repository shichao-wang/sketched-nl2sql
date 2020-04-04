""" model """
from argparse import Namespace
from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils import rnn
from transformers import AutoModel

from modules.query_predictor import (
    AggregatorPredictor,
    SelectPredictor,
    WhereColumnPredictor,
    WhereNumPredictor,
    WhereOperatorPredictor,
    WhereValuePredictor,
)


class SketchedTextToSql(nn.Module):
    """ sketched text to sql model """

    def __init__(self, embedder: nn.Module, hidden_dim: int, *, num_agg_op: int, num_conds: int, num_op: int):
        super().__init__()
        self.embedder = embedder
        self.select_predictor = SelectPredictor(embedder.config.hidden_size, hidden_dim)
        self.aggregator_predictor = AggregatorPredictor(embedder.config.hidden_size, hidden_dim, num_agg_op)
        self.where_number_predictor = WhereNumPredictor(embedder.config.hidden_size, hidden_dim, num_conds)
        self.where_column_predictor = WhereColumnPredictor(embedder.config.hidden_size, hidden_dim)
        self.where_operator_predictor = WhereOperatorPredictor(embedder.config.hidden_size, hidden_dim, num_op)
        self.where_value_position_predictor = WhereValuePredictor(embedder.config.hidden_size, hidden_dim, True)

    @classmethod
    def from_args(cls, args: Namespace):
        """ create model from args """
        embedder = AutoModel.from_pretrained(args.pretrained_model_name)
        return cls(embedder, args.hidden_dim, num_agg_op=6, num_conds=4, num_op=4)

    def forward(self, input_sequence: Tensor, input_segment: Tensor) -> Tuple[Tensor, ...]:
        """ forward """
        bert_embedding: Tensor
        bert_embedding, *_ = self.embedder(input_sequence)
        with torch.no_grad():
            question_embedding, headers_embeddings = unpack_encoder_output(bert_embedding, input_segment)

        select_logits = self.select_predictor(question_embedding, headers_embeddings)
        agg_logits = self.aggregator_predictor(question_embedding, headers_embeddings)
        where_num_logits = self.where_number_predictor(question_embedding)
        where_col_logits = self.where_column_predictor(question_embedding, headers_embeddings)
        where_op_logits = self.where_operator_predictor(question_embedding, headers_embeddings)
        value_start_logits, value_end_logits = self.where_value_position_predictor(
            question_embedding, headers_embeddings
        )
        return (
            agg_logits,
            select_logits,
            where_num_logits,
            where_col_logits,
            where_op_logits,
            value_start_logits,
            value_end_logits,
        )


def unpack_encoder_output(encoder_output: torch.Tensor, segment: torch.Tensor,) -> Tuple[Tensor, List[Tensor]]:
    """
    :param encoder_output: (batch_size, length, hidden_dim)
    :param segment: (batch_size, length)
        0: pad
        1: type
        2: header
        3: question
    :return: headers_hiddens, headers_type_hidden, question_encoding
    """
    question_hidden = rnn.pad_sequence([out[s == 2] for out, s in zip(encoder_output, segment)], batch_first=True)
    batch_size = encoder_output.size(0)
    headers_hiddens: List[Tensor] = []
    header_num_list = []
    for b in range(batch_size):
        hidden_list = []
        i = 0
        skip = True  # mode flag, denote if it is in skip or collect
        for j, seg in enumerate(segment[b]):
            if skip and seg == 1:
                # see a header segment, start collect mode
                i = j
                skip = False
            if not skip and seg != 1:
                # not a header segment, stop collect
                hidden_list.append(encoder_output[b, i:j])
                i = j
                skip = True
        header_num_list.append(len(hidden_list))
        headers_hiddens.extend(hidden_list)
    padded_hiddens: Tensor = rnn.pad_sequence(headers_hiddens, batch_first=True)
    return question_hidden, padded_hiddens.split_with_sizes(header_num_list)
