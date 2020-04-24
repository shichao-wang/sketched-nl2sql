import torch
from torch.nn.utils import rnn

from sketched_nl2sql.modules.query_predictor import BetterLSTM


def test_better_rnn():
    print("yes")
    lstm = BetterLSTM(100, 200)
    inputs = torch.empty(8, 20, 100)
    outputs = lstm(inputs, torch.empty(8, 20))


def test_scatter():
    import random

    num_class = 10
    batch_size = 4
    target = rnn.pad_sequence([
        torch.as_tensor(
            random.choices(list(range(num_class)), k=random.randint(0, 3))
        )
        for b in range(batch_size)
    ], batch_first=True)
    logits = torch.zeros((batch_size, num_class), dtype=torch.long)
    logits.scatter_(1, target, 1)
    pass
