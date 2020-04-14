""" query predictors """
import abc
from typing import List

import torch
from torch import Tensor, nn
from torch.nn.utils import rnn

from sketched_nl2sql.modules import nn_utils
from sketched_nl2sql.modules.encoder import LSTMEncoder


class SketchModule(nn.Module, metaclass=abc.ABCMeta):
    """ basic sketch module """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
        use_column_attention: bool,
    ):
        super().__init__()
        self._header_encoder = LSTMEncoder(
            input_dim, hidden_dim, bidirectional, num_layers, dropout
        )
        self._question_encoder = LSTMEncoder(
            input_dim, hidden_dim, bidirectional, num_layers, dropout
        )
        self._header_hidden_attention = nn.Linear(hidden_dim, 1)
        self.use_column_attention = use_column_attention
        if self.use_column_attention:
            self._question_hidden_attention = nn.Linear(hidden_dim, hidden_dim)
        else:
            self._question_hidden_attention = nn.Linear(hidden_dim, 1)

    def encode_question(
        self, question_embedding: Tensor, question_mask: Tensor
    ):
        """ Purely encoding question
        :param question_embedding:
        :param question_mask:
        :return:
        """
        if self.use_column_attention:
            raise ValueError("")
        question_encoding, _ = self._question_encoder(
            question_embedding, question_mask
        )
        # Shape: (bs, max_sequence_length)
        hidden_weights = self._question_hidden_attention(
            question_encoding
        ).squeeze()
        hidden_weights = hidden_weights.masked_fill(
            question_mask == 0, -float("inf")
        )
        hidden_weights = hidden_weights.softmax(dim=-1).unsqueeze(dim=1)
        question_hidden = hidden_weights @ question_encoding
        return question_hidden

    def encode_and_expand_question(
        self,
        question_embedding: Tensor,
        question_mask: Tensor,
        header_encoding: Tensor,
        header_mask: Tensor,
    ):
        """
        :param question_embedding: (bs, max_sequence_length, embed_dim)
        :param question_mask: (bs, max_sequence_length)
        :param header_encoding: (bs, max_num_column, hidden_dim)
        :param header_mask: (bs, max_num_column)
        :return:
        """
        question_encoding, _ = self._question_encoder(
            question_embedding, question_mask
        )
        if self.use_column_attention:
            # Shape: (bs, max_num_column, max_sequence_length)
            hidden_weights = header_encoding @ self._question_hidden_attention(
                question_encoding
            ).transpose(1, 2)
            hidden_weights = hidden_weights.masked_fill(
                (header_mask == 0).unsqueeze(dim=-1), -float("inf")
            )
            hidden_weights = hidden_weights.masked_fill(
                (question_mask == 0).unsqueeze(dim=1), -float("inf")
            )
            hidden_weights = hidden_weights.softmax(dim=-1)
            hidden_weights = hidden_weights.masked_fill(
                torch.isnan(hidden_weights), 0
            )
        else:
            # Shape: (bs, max_sequence_length)
            hidden_weights = self._question_hidden_attention(
                question_encoding
            ).squeeze()
            hidden_weights = hidden_weights.masked_fill(
                question_mask == 0, -float("inf")
            )
            hidden_weights = hidden_weights.softmax(dim=-1).unsqueeze(dim=1)
            num_columns = header_encoding.size(1)
            hidden_weights = hidden_weights.expand(-1, num_columns, -1)

        question_hidden = hidden_weights @ question_encoding
        return question_hidden

    def encode_and_unpack_columns(
        self,
        column_embedding: Tensor,
        column_mask: Tensor,
        num_headers: List[int],
    ):
        """
        :param column_embedding: (bs * num_headers, num_tokens, embed_dim)
        :param column_mask: (bs * num_headers, num_tokens)
        :param num_headers: [bs]
        :return: (bs, max_num_columns, hidden_dim)
        """
        # Shape: (bs * num_headers, max_num_tokens, hidden_dim)
        column_encoding, _ = self._header_encoder(
            column_embedding, column_mask
        )  # type: Tensor, _
        # Shape: (bs * num_headers, max_num_tokens)
        hidden_weights = self._header_hidden_attention(
            column_encoding
        ).squeeze(
            dim=-1
        )  # in case of column are all one token
        hidden_weights = hidden_weights.masked_fill(
            column_mask == 0, -float("inf")
        )
        hidden_weights = hidden_weights.softmax(dim=-1)
        column_hidden = hidden_weights.unsqueeze(dim=1) @ column_encoding
        unpacked_headers_hidden = column_hidden.squeeze().split_with_sizes(
            num_headers
        )
        return rnn.pad_sequence(unpacked_headers_hidden, batch_first=True)


class SelectPredictor(SketchModule):
    """ lstm """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
        use_column_attention: bool,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            bidirectional,
            num_layers,
            dropout,
            use_column_attention,
        )
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim * 2, 1)

    # pylint:disable=arguments-differ
    def forward(
        self,
        question_embedding: Tensor,
        packed_column_embedding: Tensor,
        num_headers: List[int],
    ):
        """
        :param question_embedding: (bs, sequence_length, embedding_dim)
        :param packed_column_embedding:
            (bs * num_headers, header_tokens, embedding_dim)
        :param num_headers: [bs]
        :return: (bs, num_columns)
            probability of each column to appear in SELECT clause
        """
        # Shape: (bs, max_num_columns, hidden_dim)
        header_encoding = self.encode_and_unpack_columns(
            packed_column_embedding,
            nn_utils.compute_mask(packed_column_embedding),
            num_headers,
        )
        header_mask = nn_utils.compute_mask(header_encoding)

        question_mask = nn_utils.compute_mask(question_embedding)
        # Shape: (bs, num_column, hidden_dim)
        question_hidden = self.encode_and_expand_question(
            question_embedding, question_mask, header_encoding, header_mask
        )

        x = torch.cat(
            [
                self.question_projector(question_hidden),
                self.headers_projector(header_encoding),
            ],
            dim=-1,
        )
        # Shape: (bs, num_column)
        logits = self.output_projector(x.tanh()).squeeze()
        return logits.masked_fill(header_mask == 0, -float("inf"))


class AggregatorPredictor(SketchModule):
    """ aggregator predictor """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_agg_op: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
        use_column_attention: bool,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            bidirectional,
            num_layers,
            dropout,
            use_column_attention,
        )
        self.output_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_agg_op),
        )

    # pylint:disable=arguments-differ
    def forward(
        self,
        question_embedding: Tensor,
        packed_headers_embeddings: Tensor,
        num_headers: List[int],
    ):
        """
        :param question_embedding: (bs, sequence_length, embedding_dim)
        :param packed_headers_embeddings:
            (bs * num_headers, header_tokens, embedding_dim)
        :param num_headers: [bs]
        :return: (bs, num_column, num_aggregator)
            probability of aggregator for each columns
        """
        # Shape: (bs, max_num_headers, hidden_dim)
        header_encoding = self.encode_and_unpack_columns(
            packed_headers_embeddings,
            nn_utils.compute_mask(packed_headers_embeddings),
            num_headers,
        )
        headers_mask = nn_utils.compute_mask(header_encoding)
        question_mask = nn_utils.compute_mask(question_embedding)
        # Shape: (bs, num_column, hidden_dim)
        question_hidden = self.encode_and_expand_question(
            question_embedding, question_mask, header_encoding, headers_mask
        )
        logits = self.output_projector(question_hidden)
        return logits.masked_fill(
            headers_mask.unsqueeze(dim=-1) == 0, -float("inf")
        )


class WhereNumPredictor(SketchModule):
    """ where num predictor """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_conds: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            bidirectional,
            num_layers,
            dropout,
            use_column_attention=False,
        )
        self.output_projector = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_conds),
        )

    # pylint:disable=arguments-differ
    def forward(self, question_embedding: Tensor):
        """
        :param question_embedding: (bs, sequence_length, embedding_dim)
        :return: (b, num
        """
        # Shape: (bs, 1, hidden_dim)
        question_hidden = self.encode_question(
            question_embedding, nn_utils.compute_mask(question_embedding)
        )
        # Shape: (bs, num)
        logits = self.output_projector(question_hidden).squeeze()
        return logits


class WhereColumnPredictor(SketchModule):
    """ predict where column """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
        use_column_attention: bool,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            bidirectional,
            num_layers,
            dropout,
            use_column_attention,
        )
        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim * 2, 1)

    # pylint:disable=arguments-differ
    def forward(
        self,
        question_embedding: Tensor,
        headers_embeddings: Tensor,
        num_headers: List[int],
    ):
        r"""
        :param num_headers:
        :param headers_embeddings:
            (bs, num_headers, num_header_tokens, embedding_dim)
        :param question_embedding: (bs, sequence_length, embedding_dim)
        :return: (bs, num_headers)
        """
        # Shape: (bs, max_num_columns, hidden_dim)
        header_encoding = self.encode_and_unpack_columns(
            headers_embeddings,
            nn_utils.compute_mask(headers_embeddings),
            num_headers,
        )
        header_mask = nn_utils.compute_mask(header_encoding)

        # Shape: (bs, num_columns, hidden_dim)
        question_hidden = self.encode_and_expand_question(
            question_embedding,
            nn_utils.compute_mask(question_embedding),
            header_encoding,
            header_mask,
        )
        x = torch.cat(
            [
                self.headers_projector(header_encoding),
                self.question_projector(question_hidden),
            ],
            dim=-1,
        )
        # Shape: (bs, num_columns)
        logits = self.output_projector(x.tanh()).squeeze()
        return logits.masked_fill(header_mask == 0, -float("inf"))


class WhereOperatorPredictor(SketchModule):
    """ operator predictor """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_op: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
        use_column_attention: bool,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            bidirectional,
            num_layers,
            dropout,
            use_column_attention,
        )
        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim * 2, num_op)

    # pylint:disable=arguments-differ
    def forward(
        self,
        question_embedding: Tensor,
        headers_embeddings: Tensor,
        num_headers: List[int],
    ):
        """ forward """
        # Shape: (bs, max_num_headers, hidden_dim)
        header_encoding = self.encode_and_unpack_columns(
            headers_embeddings,
            nn_utils.compute_mask(headers_embeddings),
            num_headers,
        )
        headers_mask = nn_utils.compute_mask(header_encoding)

        question_mask = nn_utils.compute_mask(question_embedding)
        # Shape: (bs, max_num_col, hidden_dim)
        question_hidden = self.encode_and_expand_question(
            question_embedding, question_mask, header_encoding, headers_mask
        )

        x = torch.cat(
            [
                self.headers_projector(header_encoding),
                self.question_projector(question_hidden),
            ],
            dim=-1,
        )
        logits = self.output_projector(x.tanh())
        # Shape: (bs, max_num_headers, num_op)
        return logits.masked_fill(
            headers_mask.unsqueeze(-1) == 0, -float("inf")
        )


class WhereValuePredictor(SketchModule):
    """ value """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
        use_column_attention: bool,
    ):
        super().__init__(
            input_dim,
            hidden_dim,
            bidirectional,
            num_layers,
            dropout,
            use_column_attention,
        )
        # self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        # self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim, 2)

    # pylint:disable=arguments-differ
    def forward(
        self,
        question_embedding: Tensor,
        headers_embeddings: Tensor,
        num_headers: List[int],
        where_num_logits: Tensor,
        where_col_logits: Tensor,
    ):
        """
        :param question_embedding: (bs, sequence_length, embedding_dim)
        :param headers_embeddings: (bs, num_headers, embed_dim)
        :param num_headers:
        :param where_num_logits: (bs, num)
        :param where_col_logits: (bs, num_cols)
        :return (bs, num_headers, sequence_length),
            (bs, num_headers, sequence_length)
        """
        del where_num_logits, where_col_logits
        # Shape: (bs, max_num_cols, hidden_dim)
        header_encoding = self.encode_and_unpack_columns(
            headers_embeddings,
            nn_utils.compute_mask(headers_embeddings),
            num_headers,
        )
        headers_mask = nn_utils.compute_mask(header_encoding)
        question_mask = nn_utils.compute_mask(question_embedding)
        question_encoding, _ = self._question_encoder(
            question_embedding, question_mask
        )
        # Shape: (bs, num_cols, hidden_dim)
        # question_hidden = self.encode_and_expand_question(
        #     question_embedding, question_mask, header_encoding, headers_mask
        # )

        # Shape: (bs, num_headers, sequence_length, hidden_dim)
        x = torch.einsum("bch,bqh->bcqh", header_encoding, question_encoding)
        mask = headers_mask.unsqueeze(2) & question_mask.unsqueeze(1)
        pos_logits: Tensor = self.output_projector(x)
        return (
            pos_logits.masked_fill(mask.unsqueeze(-1) != 1, -float("inf"))
        ).unbind(-1)
