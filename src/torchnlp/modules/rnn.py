from torch.nn.utils import rnn
from torch import Tensor, nn
from typing import Tuple


class BetterRNNMixin(nn.LSTM):
    def forward(
        self,
        embedding: Tensor,
        mask: Tensor,
        state: Tuple[Tensor, Tensor] = None,
    ):
        lengths = (mask != 0).sum(dim=1)
        packed_sequence = rnn.pack_padded_sequence(
            embedding, lengths, batch_first=True, enforce_sorted=False
        )
        packed_encoder_output, last_state = super().forward(
            packed_sequence, state
        )  # type: rnn.PackedSequence, Tuple[Tensor, Tensor]

        padded_encoder_output, lengths = rnn.pad_packed_sequence(
            packed_encoder_output, batch_first=True
        )
        return padded_encoder_output, last_state


class BetterLSTM(BetterRNNMixin, nn.LSTM):
    def __init__(
        self,
        input_size,
        hidden_size,
        num_layers=1,
        bias=True,
        batch_first=False,
        dropout=0.0,
        bidirectional=False,
    ):
        super().__init__(
            input_size,
            hidden_size // 2 if bidirectional else hidden_size,
            num_layers,
            bias,
            batch_first,
            dropout,
            bidirectional,
        )

    def forward(self, embedding, mask, state=None):
        return super().forward(embedding, mask, state=state)
