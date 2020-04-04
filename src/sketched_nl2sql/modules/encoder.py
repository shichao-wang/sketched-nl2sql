from typing import Tuple

import torch
from torch import nn, Tensor
from torch.nn.utils import rnn

from sketched_nl2sql.modules import nn_utils


class LSTMEncoder(nn.Module):
    """ wrapper class for lstm"""

    def __init__(self, input_dim: int, output_dim: int, num_layers: int = 1, batch_first=True):
        super().__init__()
        self.lstm = nn.LSTM(input_dim, output_dim, num_layers, batch_first=batch_first)

    def forward(
        self, embedding: torch.Tensor, mask: torch.Tensor
    ) -> Tuple[torch.Tensor, Tuple[torch.Tensor, torch.Tensor]]:
        """
        :param embedding: (batch_size, sequence_length, embedding_dim)
        :param mask: (batch_size, sequence_length)
        :return: (batch_size, sequence_length, hidden_dim * num_layers), (batch_size, hidden_dim * num_layers)
        """
        lengths = (mask != 0).sum(dim=1)
        packed_sequence = rnn.pack_padded_sequence(embedding, lengths, batch_first=True, enforce_sorted=False)
        packed_encoder_output, (last_hidden_state, last_cell_state) = self.lstm(
            packed_sequence
        )  # type: rnn.PackedSequence, (torch.Tensor, torch.Tensor)
        padded_encoder_output, lengths = rnn.pad_packed_sequence(packed_encoder_output, batch_first=True)
        # CHECK(Shichao Wang): Someone said if rnn is bidirectional reversed hidden in unpacked tensors is wrong
        return padded_encoder_output, (last_hidden_state.transpose(0, 1), last_cell_state.transpose(0, 1))


class HeadersEncoder(nn.Module):
    """ headers encoder """

    def __init__(self, input_dim: int, hidden_dim: int):
        super().__init__()
        self.inner_encoder = LSTMEncoder(input_dim, hidden_dim)

    def forward(self, headers_embeddings: Tuple[Tensor, ...]):
        """
        :param headers_embeddings: [batch_size, (num_headers, max_header_tokens, embedding_dim)]
        :return:
        """
        headers_num_list = [emb.size(0) for emb in headers_embeddings]
        # Shape: (batch_size * num_headers, max_num_tokens, embedding_dim)
        concatenated_embedding = torch.cat(headers_embeddings)
        mask = nn_utils.compute_mask(concatenated_embedding)
        # Shape: (batch_size * num_headers, 1, encoding_dim)
        _, (concatenated_encoding, _) = self.inner_encoder.forward(concatenated_embedding, mask)
        unpadded_headers_encodings = concatenated_encoding.squeeze().split_with_sizes(headers_num_list)
        # Shape: (batch_size, max_num_headers, hidden_dim)
        headers_encoding = rnn.pad_sequence(unpadded_headers_encodings, batch_first=True)  # pad on header number
        return headers_encoding
