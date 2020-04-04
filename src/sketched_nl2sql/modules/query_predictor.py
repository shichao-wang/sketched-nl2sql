from __future__ import annotations

from abc import ABCMeta
from typing import List, Optional, Tuple

import torch
from torch import nn, Tensor

from sketched_nl2sql.modules import nn_utils
from sketched_nl2sql.modules.attention import ColumnAttention
from sketched_nl2sql.modules import HeadersEncoder, LSTMEncoder


class SketchModule(nn.Module, metaclass=ABCMeta):
    """ basic sketch module """

    def __init__(self, input_dim: int, hidden_dim: int, use_header_attention: bool = True):
        super().__init__()
        self._headers_encoder = HeadersEncoder(input_dim, hidden_dim)
        self._question_encoder = LSTMEncoder(input_dim, hidden_dim)
        if use_header_attention:
            self._column_attention = ColumnAttention(hidden_dim)
        else:
            # noinspection PyTypeChecker
            self.register_parameter("_column_attention", None)

    def encode_question_and_headers(
        self, question_embedding: Tensor, headers_embeddings: Tuple[Tensor, ...]
    ) -> Tuple[Tensor, ...]:
        """ encoder """
        headers_encoding: Tensor = self._headers_encoder(headers_embeddings)

        question_mask = nn_utils.compute_mask(question_embedding)
        question_encoding, (question_hidden, _) = self._question_encoder(
            question_embedding, question_mask
        )  # type: Tensor, (Tensor, Tensor)

        if self._column_attention is None:
            # expand question_hidden
            question_hiddens = question_hidden.repeat(1, headers_encoding.size(1), 1)
        else:
            # Shape: (batch_size, num_header, sequence_length)
            attention_weight = self._column_attention(question_encoding, headers_encoding, question_mask)
            question_hiddens = attention_weight @ question_encoding
        return question_hiddens, headers_encoding, question_encoding


class SelectPredictor(nn.Module):
    """ lstm """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.question_encoder = LSTMEncoder(input_dim, hidden_dim)
        self.headers_encoder = HeadersEncoder(input_dim, hidden_dim)
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim, 1)

    def forward(self, question_embedding: Tensor, headers_embeddings: Tuple[Tensor, ...]):
        """
        :param question_embedding: (batch_size, sequence_length, embedding_dim)
        :param headers_embeddings: [batch_size, (num_headers, header_tokens, embedding_dim)]
        :return: (batch_size, num_columns)
            probability of each column to appear in SELECT clause
        """
        # Shape: (batch_size, sequence_length, hidden_dim), (batch_size, 1, hidden_dim)
        question_encoding, (question_hidden, _) = self.question_encoder(
            question_embedding, nn_utils.compute_mask(question_embedding)
        )
        # Shape: (batch_size, num_headers, hidden_dim)
        headers_hidden = self.headers_encoder(headers_embeddings)
        x = self.question_projector(question_hidden) + self.headers_projector(headers_hidden)
        output = self.output_projector(x.tanh())
        return output.squeeze()


class AggregatorPredictor(nn.Module):
    """ aggregator predictor """

    def __init__(self, input_dim: int, hidden_dim: int, num_agg_op: int):
        super().__init__()
        self.question_encoder = LSTMEncoder(input_dim, hidden_dim)
        self.headers_encoder = HeadersEncoder(input_dim, hidden_dim)
        self.column_attention = ColumnAttention(hidden_dim)
        self.output_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, num_agg_op)
        )

    def forward(self, question_embedding: Tensor, headers_embeddings: List[Tensor]):
        """
        :param question_embedding: (batch_size, sequence_length, embedding_dim)
        :param headers_embeddings: [batch_size, (num_headers, header_tokens, embedding_dim)]
        :return: (batch_size, num_column, num_aggregator)
            probability of aggregator for each columns
        """
        question_mask = nn_utils.compute_mask(question_embedding)
        question_encoding, (_, _) = self.question_encoder.forward(question_embedding, question_mask)
        # Shape: (batch_size, num_headers, hidden_dim)
        headers_hidden = self.headers_encoder(headers_embeddings)
        # Shape: (num_header, sequence_length)
        attention_weight = self.column_attention.forward(question_encoding, headers_hidden, question_mask)
        conditioned_question_encoding = attention_weight @ question_encoding
        x = self.output_projector(conditioned_question_encoding)
        return x


class WhereNumPredictor(nn.Module):
    """ where num predictor """

    def __init__(self, input_dim: int, hidden_dim: int, num_conds: int):
        super().__init__()
        self.question_encoder = LSTMEncoder(input_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim, num_conds)

    def forward(self, question_embedding: Tensor):
        """
        :param question_embedding: (batch_size, sequence_length, embedding_dim)
        :return:
        """
        # Shape: (batch_size, 1, hidden_dim)
        _, (question_hidden, _) = self.question_encoder.forward(
            question_embedding, nn_utils.compute_mask(question_embedding)
        )
        x = self.output_projector(question_hidden)
        return x.squeeze()


class WhereColumnPredictor(nn.Module):
    """ predict where column """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.question_encoder = LSTMEncoder(input_dim, hidden_dim)
        self.headers_encoder = HeadersEncoder(input_dim, hidden_dim)
        self.column_attention = ColumnAttention(hidden_dim)

        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim * 2, 1)

    def forward(self, question_embedding: Tensor, headers_embeddings: List[Tensor]):
        r"""
        P(col|Q) = \sigma(u_c^T @ E_col + u_q^T @ E_q)
        :param headers_embeddings: (batch_size, num_headers, num_header_tokens, embedding_dim)
        :param question_embedding: (batch_size, sequence_length, embedding_dim)
        :return: (batch_size, num_headers)
        """
        question_mask = nn_utils.compute_mask(question_embedding)
        # Shape: (batch_size, sequence_length, hidden_dim), (batch_size, 1, hidden_dim)
        question_encoding, (question_hidden, _) = self.question_encoder(question_embedding, question_mask)
        # Shape: (batch_size, num_headers, hidden_dim)
        headers_hidden = self.headers_encoder(headers_embeddings)
        # Shape: (num_header, sequence_length)
        attention_weight = self.column_attention.forward(question_encoding, headers_hidden, question_mask)
        # Shape: (batch_size, num_headers, hidden_dim)
        conditioned_question_encoding = attention_weight @ question_encoding
        x = torch.cat(
            [self.headers_projector(headers_hidden), self.question_projector(conditioned_question_encoding)], dim=-1
        )
        x = self.output_projector(x.tanh())
        return x.squeeze()


class WhereOperatorPredictor(nn.Module):
    """ operator predictor """

    def __init__(self, input_dim: int, hidden_dim: int, num_op: int):
        super().__init__()
        self.question_encoder = LSTMEncoder(input_dim, hidden_dim)
        self.headers_encoder = HeadersEncoder(input_dim, hidden_dim)
        self.column_attention = ColumnAttention(hidden_dim)

        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim * 2, num_op)

    def forward(self, question_embedding: Tensor, headers_embeddings: Tuple[Tensor, ...]):
        """ forward """
        question_encoding, headers_encoding, attention_weight = module_encode(
            question_embedding, headers_embeddings, self.question_encoder, self.headers_encoder, self.column_attention
        )
        x = torch.cat([self.question_projector(question_encoding), self.headers_projector(headers_encoding)], dim=-1)
        logits = self.output_projector(x.tanh())
        return logits  # softmax


def module_encode(
    question_embedding: Tensor,
    headers_embeddings: Tuple[Tensor, ...],
    question_encoder: LSTMEncoder,
    headers_encoder: HeadersEncoder,
    column_attention: ColumnAttention = None,
) -> Tuple[Tensor, Tensor, Optional[Tensor]]:
    """ helper function that used in module """
    question_mask = nn_utils.compute_mask(question_embedding)
    # Shape: (batch_size, sequence_length, hidden_dim), (batch_size, 1, hidden_dim)
    question_encoding, (question_hidden, _) = question_encoder(question_embedding, question_mask)
    # Shape: (batch_size, num_headers, hidden_dim)
    headers_hidden = headers_encoder(headers_embeddings)
    attention_weight = None
    if column_attention:
        # Shape: (batch_size, num_header, sequence_length)
        attention_weight = column_attention(question_encoding, headers_hidden, question_mask)
        # Shape: (batch_size, num_headers, hidden_dim)
        question_encoding = attention_weight @ question_encoding

    return question_encoding, headers_hidden, attention_weight


class WhereValuePredictor(SketchModule):
    """ value """

    def __init__(self, input_dim, hidden_dim, use_header_attention: bool):
        super().__init__(input_dim, hidden_dim, use_header_attention=use_header_attention)
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim * 2, 2)

    def forward(self, question_embedding: Tensor, headers_embeddings: Tuple[Tensor, ...]):
        """
        :param question_embedding: (batch_size, sequence_length, embedding_dim)
        :param headers_embeddings: (batch_size, num_headers, embed_dim)
        :return (batch_size, num_headers, sequence_length), (batch_size, num_headers, sequence_length)
        """
        # Shape: (batch_size, num_headers, hidden_dim), (batch_size, num_headers, hidden_dim)
        question_hiddens, headers_hidden, question_encoding = self.encode_question_and_headers(
            question_embedding, headers_embeddings
        )

        # Shape: (batch_size, num_headers, sequence_length, hidden_dim)
        expanded_question_hiddens = question_hiddens.unsqueeze(2).repeat(1, 1, question_encoding.size(1), 1)
        expanded_question_encoding = question_encoding.unsqueeze(1).repeat(1, headers_hidden.size(1), 1, 1)
        x = torch.cat(
            [self.headers_projector(expanded_question_encoding), self.question_projector(expanded_question_hiddens)],
            dim=-1,
        )
        # IMPROVE padding
        pos_logits: Tensor = self.output_projector(x.tanh())
        return pos_logits.unbind(-1)
