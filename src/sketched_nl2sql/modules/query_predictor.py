""" query predictors """
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn.utils import rnn

from sketched_nl2sql.modules import nn_utils

from torchnlp.modules.rnn import BetterLSTM


class SelectPredictor(nn.Module):
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
        super().__init__()
        self.q_lstm = BetterLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.c_lstm = BetterLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.c_attn_proj = nn.Linear(hidden_dim, 1)

        self.q_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.c_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 1),
        )

    # pylint:disable=arguments-differ
    def forward(
        self,
        question_embedding: Tensor,
        packed_column_embedding: Tensor,
        num_columns: List[int],
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
        column_hidden = encode_and_unpack_columnn(
            self.c_lstm,
            self.c_attn_proj,
            packed_column_embedding,
            nn_utils.compute_mask(packed_column_embedding),
            num_columns,
        )
        question_mask = nn_utils.compute_mask(question_embedding)
        # Shape: (bs, num_column, hidden_dim)
        question_encoding, _ = self.q_lstm(question_embedding, question_mask)
        column_mask = nn_utils.compute_mask(column_hidden)

        attention_weight = nn_utils.mm_attention(
            column_hidden, question_encoding, column_mask, question_mask
        )
        question_hidden = attention_weight @ question_encoding

        x = torch.cat(
            [
                self.q_out_proj(question_hidden),
                self.c_out_proj(column_hidden),
            ],
            dim=-1,
        )
        # Shape: (bs, num_column)
        logits = self.out(x).squeeze()
        return logits.masked_fill(~column_mask.bool(), -float("inf"))


class AggregatorPredictor(nn.Module):
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
        super().__init__()
        self.q_lstm = BetterLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.c_lstm = BetterLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.col_hid_attn_proj = nn.Linear(hidden_dim, 1)

        self.out = nn.Sequential(
            nn.Linear(hidden_dim, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_agg_op),
        )

    # pylint:disable=arguments-differ
    def forward(
        self,
        question_embedding: Tensor,
        packed_headers_embeddings: Tensor,
        num_columns: List[int],
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
        column_hidden = encode_and_unpack_columnn(
            self.c_lstm,
            self.col_hid_attn_proj,
            packed_headers_embeddings,
            nn_utils.compute_mask(packed_headers_embeddings),
            num_columns,
        )
        column_mask = nn_utils.compute_mask(column_hidden)

        question_mask = nn_utils.compute_mask(question_embedding)
        question_encoding, _ = self.q_lstm(question_embedding, question_mask)
        attention_weight = nn_utils.mm_attention(
            column_hidden, question_encoding, column_mask, question_mask
        )
        question_hidden = attention_weight @ question_encoding
        # (bs, num_column, num_aggregator)
        logits = self.out(question_hidden)
        return logits.masked_fill(
            ~column_mask.unsqueeze(dim=-1).bool(), -float("inf")
        )


class WhereNumPredictor(nn.Module):
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
        super().__init__()
        self.q_lstm = BetterLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.q_hid_attn_proj = nn.Linear(hidden_dim, 1)
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
        question_mask = nn_utils.compute_mask(question_embedding)

        question_encoding, _ = self.q_lstm(question_embedding, question_mask)
        attention_weight = nn_utils.make_weight(
            self.q_hid_attn_proj(question_encoding).squeeze(dim=-1),
            question_mask,
        )
        # Shape: (BS, 1, hidden_dim)
        question_hidden = attention_weight.unsqueeze(dim=1) @ question_encoding
        # Shape: (bs, num)
        logits = self.output_projector(question_hidden).squeeze(dim=1)
        return logits  # no pad


class WhereColumnPredictor(nn.Module):
    """ predict where column """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_conds: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
        use_column_attention: bool,
    ):
        super().__init__()
        self.q_lstm = BetterLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.c_lstm = BetterLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.c_hid_attn_proj = nn.Linear(hidden_dim, 1)
        self.headers_projector = nn.Linear(hidden_dim, hidden_dim)
        self.question_projector = nn.Linear(hidden_dim, hidden_dim)
        self.output_projector = nn.Linear(hidden_dim * 2, 1)

    def forward(
        self,
        question_embedding: Tensor,
        packed_column_embedding: Tensor,
        num_columns: List[int],
        gold_num: Tensor,
    ):
        """
        :param num_headers:
        :param headers_embeddings:
            (bs, num_headers, num_header_tokens, embedding_dim)
        :param question_embedding: (bs, sequence_length, embedding_dim)
        :return: (bs, num_headers)
        """
        # Shape: (bs, max_num_columns, hidden_dim)
        column_hidden = encode_and_unpack_columnn(
            self.c_lstm,
            self.c_hid_attn_proj,
            packed_column_embedding,
            nn_utils.compute_mask(packed_column_embedding),
            num_columns,
        )
        column_mask = nn_utils.compute_mask(column_hidden)
        question_mask = nn_utils.compute_mask(question_embedding)
        # Shape: (bs, num_columns, hidden_dim)
        question_encoding, _ = self.q_lstm(question_embedding, question_mask)
        attention_weight = nn_utils.mm_attention(
            column_hidden, question_encoding, column_mask, question_mask
        )
        question_hidden = attention_weight @ question_encoding

        x = torch.cat(
            [
                self.headers_projector(column_hidden),
                self.question_projector(question_hidden),
            ],
            dim=-1,
        )
        # Shape: (bs, num_columns)
        logits = self.output_projector(x.tanh()).squeeze()
        return logits.masked_fill(~column_mask.bool(), -float("inf"))


class WhereOperatorPredictor(nn.Module):
    """ operator predictor """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_op: int,
        num_conds: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
        use_column_attention: bool,
    ):
        super().__init__()
        self.num_conds = num_conds
        self.q_lstm = BetterLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.c_lstm = BetterLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            dropout=dropout,
            bidirectional=bidirectional,
        )
        self.c_attn_proj = nn.Linear(hidden_dim, 1)
        self.ob_col_attn_proj = nn.Linear(hidden_dim, hidden_dim)

        self.c_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.q_out_proj = nn.Linear(hidden_dim, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 2, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, num_op),
        )

    # pylint:disable=arguments-differ
    def forward(
        self,
        question_embedding: Tensor,
        packed_column_embedding: Tensor,
        num_columns: List[int],
        gold_num: Tensor,
        gold_cols: List[Tensor],
    ):
        """
        :param question_embedding: (B, seq_len, emb_dim)
        :param headers_embeddings: (B, num_headers, emb_dim)
        :param num_headers
        :return: (B, num_conds, num_op)
        """
        question_mask = nn_utils.compute_mask(question_embedding)
        question_encoding, _ = self.q_lstm(question_embedding, question_mask)

        packed_col_mask = nn_utils.compute_mask(packed_column_embedding)
        column_hidden = encode_and_unpack_columnn(
            self.c_lstm,
            self.c_attn_proj,
            packed_column_embedding,
            packed_col_mask,
            num_columns,
        )

        # Shape: (BS, num_conds, hidden)
        observed_column_hidden, observed_mask = get_observed_column_hidden(
            column_hidden, gold_cols, self.num_conds
        )
        attention = nn_utils.mm_attention(
            self.ob_col_attn_proj(observed_column_hidden),
            question_encoding,
            observed_mask,
            question_mask,
        )
        # Shape: (BS, num_conds, hidden)
        question_hidden = attention @ question_encoding

        x = torch.cat(
            [
                self.c_out_proj(observed_column_hidden),
                self.q_out_proj(question_hidden),
            ],
            dim=-1,
        )
        # Shape: (B, num_conds, num_op)
        logits = self.out(x)
        return logits.masked_fill(
            ~observed_mask.unsqueeze(dim=-1).bool(), -float("inf")
        )


class WhereValuePredictor(nn.Module):
    """ value """

    # pylint: disable=too-many-arguments
    def __init__(
        self,
        input_dim: int,
        hidden_dim: int,
        num_conds: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float,
        use_column_attention: bool,
    ):
        super().__init__()
        self.num_conds = num_conds

        self.q_lstm = BetterLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.c_lstm = BetterLSTM(
            input_dim,
            hidden_dim,
            num_layers,
            batch_first=True,
            bidirectional=bidirectional,
            dropout=dropout,
        )
        self.col_hid_attn_proj = nn.Linear(hidden_dim, 1)
        self.ob_col_attn_proj = nn.Linear(hidden_dim, hidden_dim)
        self.ob_col_proj = nn.Linear(hidden_dim, hidden_dim)
        self.q_proj = nn.Linear(hidden_dim, hidden_dim)
        # self.op_out_proj = nn.Linear(num_ops, hidden_dim)
        self.out = nn.Sequential(
            nn.Linear(hidden_dim * 3, hidden_dim),
            nn.Tanh(),
            nn.Linear(hidden_dim, 2),
        )

    def forward(
        self,
        question_embedding: Tensor,
        packed_column_embedding: Tensor,
        num_columns: List[int],
        gold_num: Tensor,
        gold_cols: List[Tensor],
        gold_ops: List[Tensor],
    ):
        """
        :param question_embedding: (bs, sequence_length, embedding_dim)
        :param headers_embeddings: (bs, num_headers, embed_dim)
        :param num_headers:
        :param where_num_logits: (bs, num_conds)
        :param where_col_logits: (bs, num_cols)
        :return (bs, num_conds, sequence_length),
            (bs, num_conds, sequence_length)
        """

        question_mask = nn_utils.compute_mask(question_embedding)
        question_encoding, _ = self.q_lstm(question_embedding, question_mask)

        # Shape: (bs, max_num_cols, hidden_dim)
        column_hidden = encode_and_unpack_columnn(
            self.c_lstm,
            self.col_hid_attn_proj,
            packed_column_embedding,
            nn_utils.compute_mask(packed_column_embedding),
            num_columns,
        )

        # Shape: (BS, num_conds, hidden)
        observed_column_hidden, observed_mask = get_observed_column_hidden(
            column_hidden, gold_cols, self.num_conds
        )
        # Shape: (BS, num_conds, seq_len)
        attention = nn_utils.mm_attention(
            self.ob_col_attn_proj(observed_column_hidden),
            question_encoding,
            observed_mask,
            question_mask,
        )
        # Shape: (BS, num_conds, hidden)
        question_hidden = attention @ question_encoding

        # observed_op = get_observed_op(where_op_logits, where_col_logits)

        vec = torch.cat(
            [
                self.ob_col_proj(observed_column_hidden),
                self.q_proj(question_hidden),
                # self.op_out_proj(op_hidden),
            ],
            dim=-1,
        )
        expanded_vec = vec.unsqueeze(dim=2).expand(
            -1, -1, question_encoding.size(1), -1
        )
        expanded_question_encoding = question_encoding.unsqueeze(1).expand(
            -1, vec.size(1), -1, -1
        )
        # Shape: (B, num_cond, seq_len, h+2h)
        x = torch.cat([expanded_question_encoding, expanded_vec], dim=-1)
        # Shape: (B, num_cond, seq_len, 2)
        pos_logits = self.out(x)
        mask = torch.einsum("bc,bq->bcq", observed_mask, question_mask)
        return (
            pos_logits.masked_fill(
                ~mask.unsqueeze(dim=-1).bool(), -float("inf")
            )
        ).unbind(-1)


@torch.no_grad()
def get_observed_column_hidden(
    header_encoding: Tensor, gold_cols: List[Tensor], num_conds: int
):
    """
    :param header_encoding: (BS, num_column, hidden_dim)
    :param where_num_logits: [bs, num_conds]
    :return:
    """

    bs, _, hidden_dim = header_encoding.size()

    observed = header_encoding.new_zeros(bs, num_conds, hidden_dim)
    mask = header_encoding.new_zeros(bs, num_conds)
    for b, cols in enumerate(gold_cols):
        observed[b, : len(cols)] = header_encoding[b, cols]
        mask[b, : len(cols)] = 1
    return observed, mask


# def get_observed_op(where_op_logits: Tensor, where_num_logits: Tensor):
#     """
#     :param where_op_logits: (BS, num_cond, num_op)
#     :param where_col_logits: (BS, max_cols)
#     :return (BS, num_cond, num_op)
#     """
#     num_cols = where_num_logits.argmax(dim=-1).tolist()
#     observed = where_op_logits


def rnn_forward(rnn: nn.RNNBase, embedding: Tensor, mask: Tensor):
    lengths = (mask != 0).sum(dim=1)
    packed_sequence = rnn.pack_padded_sequence(
        embedding, lengths, batch_first=True, enforce_sorted=False
    )
    packed_encoder_output, last_state = rnn(
        packed_sequence
    )  # type: rnn.PackedSequence, Tuple[torch.Tensor, torch.Tensor]

    padded_encoder_output, lengths = rnn.pad_packed_sequence(
        packed_encoder_output, batch_first=True
    )
    return padded_encoder_output, last_state


def encode_and_unpack_columnn(
    c_rnn: BetterLSTM,
    c_proj: nn.Linear,
    embedding: Tensor,
    mask: Tensor,
    num_columns: List[int],
):
    """
    :param embedding (BS, num_token, embed_dim)
    """
    encoding, _ = c_rnn(embedding, mask)
    weight = nn_utils.make_weight(c_proj(encoding).squeeze(dim=-1), mask)
    hidden = (weight.unsqueeze(dim=1) @ encoding).squeeze(dim=1)
    return rnn.pad_sequence(
        hidden.split_with_sizes(num_columns), batch_first=True, padding_value=0
    )
