from typing import Tuple

import torch
from torch import nn
from torch.nn.utils import rnn


class LSTM(nn.Module):
    """ wrapper class for lstm"""

    def __init__(
        self,
        input_dim: int,
        output_dim: int,
        bidirectional: bool,
        num_layers: int,
        dropout: float = 0.0,
        batch_first=True,
    ):
        super().__init__()
        if bidirectional:
            assert output_dim & 1 == 0  # odd
            output_dim //= 2

        self.lstm = nn.LSTM(
            input_dim,
            output_dim,
            num_layers,
            batch_first=batch_first,
            dropout=dropout,
            bidirectional=bidirectional,
        )

    def forward(
        self, embedding: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param embedding: (batch_size, sequence_length, embedding_dim)
        :param mask: (batch_size, sequence_length)
        :return: (batch_size, sequence_length, hidden_dim * num_layers), 
            (batch_size, hidden_dim * num_layers)
        """
        lengths = (mask != 0).sum(dim=1)
        packed_sequence = rnn.pack_padded_sequence(
            embedding, lengths, batch_first=True, enforce_sorted=False
        )
        packed_encoder_output, last_state = self.lstm(
            packed_sequence
        )  # type: rnn.PackedSequence, Tuple[torch.Tensor, torch.Tensor]
        # num_directions = 2 if self.lstm.bidirectional else 1

        padded_encoder_output, lengths = rnn.pad_packed_sequence(
            packed_encoder_output, batch_first=True
        )
        return padded_encoder_output, last_state
