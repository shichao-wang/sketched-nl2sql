import logging

import torch
from torch import nn, Tensor

from modules import nn_utils

logger = logging.getLogger(__name__)


class ColumnAttention(nn.Module):
    """ column start_attention """

    def __init__(self, hidden_dim: int):
        super().__init__()
        self.w = nn.Linear(hidden_dim, hidden_dim)

    def forward(
        self,
        question_encoding: Tensor,
        headers_hidden: Tensor,
        question_mask: Tensor = None,
        headers_mask: Tensor = None,
    ) -> Tensor:
        """ compute start_attention mask and apply to question_hidden
        :param question_encoding: (batch_size, sequence_length, hidden_dim)
        :param headers_hidden: (batch_size, header_num, hidden_dim)
        :param question_mask: (batch_size, sequence_length)
        :param headers_mask: (batch_size, header_num)
        :return (batch_size, header_num, sequence_length)
        """
        question_mask = nn_utils.compute_mask(question_encoding) if question_mask is None else question_mask
        headers_mask = nn_utils.compute_mask(headers_hidden) if headers_mask is None else headers_mask

        # Shape: (batch_size, header_num, sequence_length)
        attention_weight = headers_hidden @ self.w(question_encoding).transpose(1, 2)
        # add penalty on padding
        mask = headers_mask.unsqueeze(2) @ question_mask.unsqueeze(1)
        attention_weight = attention_weight.masked_fill(mask.bool(), -float("inf"))
        attention_weight = attention_weight.softmax(dim=-1)
        # defending replace nan to 0, those place with nan should not be used
        attention_weight = attention_weight.masked_fill(torch.isnan(attention_weight), 0)
        return attention_weight


class LuongAttention(nn.Module):
    """ only implement global dot start_attention """

    def __init__(self, hidden_dim: int):
        super().__init__()

    def forward(self, current_hidden: Tensor, source_states: Tensor, source_mask: Tensor = None) -> Tensor:
        """
        :param current_hidden: (batch_size, 1, hidden_dim)
        :param source_states: (batch_size, source_length, hidden_dim)
        :param source_mask: (batch_size, source_length)
        :return: (batch_size, 1, source_length)
        """

        source_mask = source_mask or nn_utils.compute_mask(source_states)
        # Shape: (batch_size, 1, source_length) dot
        align_score = current_hidden @ source_states.transpose(1, 2)
        align_score[source_mask == 0] = -float("inf")  # penalty on pad token
        attention_weight = align_score.softmax(dim=-1)
        return attention_weight
        # Shape: (batch_size, 1, hidden_dim)
        # context = attention_weight.transpose(1, 2) @ source_states
        # x = self.start_output_projector(torch.cat([context, current_hidden], dim=-1))
        # return x
