from abc import ABCMeta
from typing import List

import torch
from torch import nn, Tensor
from torch.nn.utils import rnn

from sketched_nl2sql.modules import nn_utils
from sketched_nl2sql.modules.attention import ColumnAttention
from sketched_nl2sql.modules.encoder import LSTMEncoder


class SketchModule(nn.Module, metaclass=ABCMeta):
    """ basic sketch module """

    def __init__(self, input_dim: int, hidden_dim: int, *, use_header_attention: bool):
        super().__init__()
        self._header_encoder = LSTMEncoder(input_dim, hidden_dim)
        self._question_encoder = LSTMEncoder(input_dim, hidden_dim)
        if use_header_attention:
            self._header_attention = ColumnAttention(hidden_dim)
        else:
            # noinspection PyTypeChecker
            self.register_parameter("_header_attention", None)

    def encode_question(self, question_embedding: Tensor, question_mask: Tensor):
        """
        :param question_embedding: (batch_size, max_sequence_length, embed_dim)
        :param question_mask: (batch_size, max_sequence_length)
        :return:
        """
        question_encoding, (question_hidden, _) = self._question_encoder(question_embedding, question_mask)
        return question_hidden, question_encoding

    def encode_and_unpack_headers(self, headers_embedding: Tensor, header_mask: Tensor, num_headers: List[int]):
        """
        :param headers_embedding: (batch_size * num_headers, num_tokens, embed_dim)
        :param header_mask: (batch_size * num_headers, num_tokens)
        :param num_headers: [batch_size]
        :return:
        """
        # Shape: (batch_size * num_headers, max_num_tokens, embedding_dim)
        # Shape: (batch_size * num_headers, 1, encoding_dim)
        _, (headers_hidden, _) = self._header_encoder(headers_embedding, header_mask)  # type: _, (Tensor, _)
        unpacked_headers_hidden = headers_hidden.squeeze().split_with_sizes(num_headers)
        return rnn.pad_sequence(unpacked_headers_hidden, batch_first=True)

    def header_attention(
        self, question_encoding: Tensor, question_mask: Tensor, headers_hidden: Tensor, headers_mask: Tensor
    ):
        """
        :param question_encoding: (b, max_q_len, hidden)
        :param question_mask: (b, max_q_len)
        :param headers_hidden: (b, max_h_num, hidden)
        :param headers_mask: (b, max_h_num)
        :return: (batch_size, num_header, hidden_dim)
            conditioned question hidden. P(q|h)
        """
        attention_weight = self._header_attention(question_encoding, headers_hidden, question_mask, headers_mask)
        return attention_weight @ question_encoding


class SelectPredictor(SketchModule):
    """ lstm """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim, use_header_attention=True)
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim * 2, 1)

    def forward(self, question_embedding: Tensor, packed_headers_embeddings: Tensor, num_headers: List[int]):
        """
        :param question_embedding: (batch_size, sequence_length, embedding_dim)
        :param packed_headers_embeddings: (batch_size * num_headers, header_tokens, embedding_dim)
        :param num_headers: [batch_size]
        :return: (batch_size, num_columns)
            probability of each column to appear in SELECT clause
        """
        question_mask = nn_utils.compute_mask(question_embedding)
        # Shape: (batch_size, sequence_length, hidden_dim), (batch_size, 1, hidden_dim)
        question_hidden, question_encoding = self.encode_question(question_embedding, question_mask)
        # Shape: (batch_size, max_num_headers, hidden_dim)
        headers_hidden = self.encode_and_unpack_headers(
            packed_headers_embeddings, nn_utils.compute_mask(packed_headers_embeddings), num_headers
        )
        headers_mask = nn_utils.compute_mask(headers_hidden)
        conditioned_question_hidden = self.header_attention(
            question_encoding, question_mask, headers_hidden, headers_mask
        )
        x = torch.cat(
            [self.question_projector(conditioned_question_hidden), self.headers_projector(headers_hidden)], dim=-1
        )
        logits = self.output_projector(x.tanh())
        return logits.squeeze().masked_fill(headers_mask != 1, -float("inf"))


class AggregatorPredictor(SketchModule):
    """ aggregator predictor """

    def __init__(self, input_dim: int, hidden_dim: int, num_agg_op: int):
        super().__init__(input_dim, hidden_dim, use_header_attention=True)
        self.output_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim), nn.Tanh(), nn.Linear(hidden_dim, num_agg_op)
        )

    def forward(self, question_embedding: Tensor, packed_headers_embeddings: Tensor, num_headers: List[int]):
        """
        :param question_embedding: (batch_size, sequence_length, embedding_dim)
        :param packed_headers_embeddings: (batch_size * num_headers, header_tokens, embedding_dim)
        :param num_headers: [batch_size]
        :return: (batch_size, num_column, num_aggregator)
            probability of aggregator for each columns
        """
        question_mask = nn_utils.compute_mask(question_embedding)
        # Shape: (batch_size, sequence_length, hidden_dim), (batch_size, 1, hidden_dim)
        question_hidden, question_encoding = self.encode_question(question_embedding, question_mask)
        # Shape: (batch_size, max_num_headers, hidden_dim)
        headers_hidden = self.encode_and_unpack_headers(
            packed_headers_embeddings, nn_utils.compute_mask(packed_headers_embeddings), num_headers
        )
        headers_mask = nn_utils.compute_mask(headers_hidden)
        conditioned_question_hidden = self.header_attention(
            question_encoding, question_mask, headers_hidden, headers_mask
        )
        logits = self.output_projector(conditioned_question_hidden)
        return logits.masked_fill(headers_mask.unsqueeze(-1) != 1, -float("inf"))


class WhereNumPredictor(SketchModule):
    """ where num predictor """

    def __init__(self, input_dim: int, hidden_dim: int, num_conds: int):
        super().__init__(input_dim, hidden_dim, use_header_attention=False)
        self.output_projector = nn.Linear(hidden_dim, num_conds)

    def forward(self, question_embedding: Tensor):
        """
        :param question_embedding: (batch_size, sequence_length, embedding_dim)
        :return: (b, num
        """
        # Shape: (batch_size, sequence_length, hidden_dim), (batch_size, 1, hidden_dim)
        question_hidden, question_encoding = self.encode_question(
            question_embedding, nn_utils.compute_mask(question_embedding)
        )
        x = self.output_projector(question_hidden)
        return x.squeeze()


class WhereColumnPredictor(SketchModule):
    """ predict where column """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__(input_dim, hidden_dim, use_header_attention=True)
        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim * 2, 1)

    def forward(self, question_embedding: Tensor, headers_embeddings: Tensor, num_headers: List[int]):
        r"""
        :param num_headers:
        :param headers_embeddings: (batch_size, num_headers, num_header_tokens, embedding_dim)
        :param question_embedding: (batch_size, sequence_length, embedding_dim)
        :return: (batch_size, num_headers)
        """
        question_mask = nn_utils.compute_mask(question_embedding)
        # Shape: (batch_size, sequence_length, hidden_dim), (batch_size, 1, hidden_dim)
        question_hidden, question_encoding = self.encode_question(question_embedding, question_mask)
        # Shape: (batch_size, max_num_headers, hidden_dim)
        headers_hidden = self.encode_and_unpack_headers(
            headers_embeddings, nn_utils.compute_mask(headers_embeddings), num_headers
        )
        headers_mask = nn_utils.compute_mask(headers_hidden)
        conditioned_question_hidden = self.header_attention(
            question_encoding, question_mask, headers_hidden, headers_mask
        )
        x = torch.cat(
            [self.headers_projector(headers_hidden), self.question_projector(conditioned_question_hidden)], dim=-1
        )
        logits = self.output_projector(x.tanh())
        return logits.squeeze().masked_fill(headers_mask != 1, -float("inf"))


class WhereOperatorPredictor(SketchModule):
    """ operator predictor """

    def __init__(self, input_dim: int, hidden_dim: int, num_op: int):
        super().__init__(input_dim, hidden_dim, use_header_attention=True)
        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim * 2, num_op)

    def forward(self, question_embedding: Tensor, headers_embeddings: Tensor, num_headers: List[int]):
        """ forward """
        question_mask = nn_utils.compute_mask(question_embedding)
        # Shape: (batch_size, sequence_length, hidden_dim), (batch_size, 1, hidden_dim)
        question_hidden, question_encoding = self.encode_question(question_embedding, question_mask)
        # Shape: (batch_size, max_num_headers, hidden_dim)
        headers_hidden = self.encode_and_unpack_headers(
            headers_embeddings, nn_utils.compute_mask(headers_embeddings), num_headers
        )
        headers_mask = nn_utils.compute_mask(headers_hidden)
        conditioned_question_hidden = self.header_attention(
            question_encoding, question_mask, headers_hidden, headers_mask
        )
        x = torch.cat(
            [self.headers_projector(headers_hidden), self.question_projector(conditioned_question_hidden)], dim=-1
        )
        logits = self.output_projector(x.tanh())
        # Shape: (batch_size, max_num_headers, num_op)
        return logits.masked_fill(headers_mask.unsqueeze(-1) != 1, -float("inf"))


class WhereValuePredictor(SketchModule):
    """ value """

    def __init__(self, input_dim, hidden_dim, use_header_attention: bool):
        super().__init__(input_dim, hidden_dim, use_header_attention=use_header_attention)
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim * 2, 2)

    def forward(self, question_embedding: Tensor, headers_embeddings: Tensor, num_headers: List[int]):
        """
        :param num_headers:
        :param question_embedding: (batch_size, sequence_length, embedding_dim)
        :param headers_embeddings: (batch_size, num_headers, embed_dim)
        :return (batch_size, num_headers, sequence_length), (batch_size, num_headers, sequence_length)
        """
        question_mask = nn_utils.compute_mask(question_embedding)
        # Shape: (batch_size, sequence_length, hidden_dim), (batch_size, 1, hidden_dim)
        question_hidden, question_encoding = self.encode_question(
            question_embedding, question_mask
        )  # type: Tensor, Tensor

        # Shape: (batch_size, max_num_headers, hidden_dim)
        headers_hidden = self.encode_and_unpack_headers(
            headers_embeddings, nn_utils.compute_mask(headers_embeddings), num_headers
        )

        headers_mask = nn_utils.compute_mask(headers_hidden)
        conditioned_question_hidden = self.header_attention(
            question_encoding, question_mask, headers_hidden, nn_utils.compute_mask(headers_hidden)
        )  # type: Tensor

        # Shape: (batch_size, num_headers, sequence_length, hidden_dim)
        expanded_question_hiddens = conditioned_question_hidden.unsqueeze(2).expand(
            -1, -1, question_encoding.size(1), -1
        )
        expanded_question_encoding = question_encoding.unsqueeze(1).expand(-1, headers_hidden.size(1), -1, -1)
        x = torch.cat(
            [self.question_projector(expanded_question_hiddens), self.headers_projector(expanded_question_encoding)],
            dim=-1,
        )
        mask = headers_mask.unsqueeze(2) & question_mask.unsqueeze(1)
        pos_logits: Tensor = self.output_projector(x.tanh())
        return (pos_logits.masked_fill(mask.unsqueeze(-1) != 1, -float("inf"))).unbind(-1)
