from typing import List, Tuple

import torch
from torch import nn, Tensor
from torch.nn import functional as f


class QueryLoss(nn.Module):
    """ query loss """

    def __init__(self):
        super().__init__()

    def forward(
        self,
        logits: Tuple[Tensor, ...],
        target: Tuple[Tensor, Tensor, Tensor, List[Tensor], List[Tensor], List[Tensor], List[Tensor]],
    ):
        """
        :param logits: tuple
        :param target: tuple
        :return:
        """
        (
            agg_logits,
            sel_logits,
            where_num_logits,
            where_col_logits,
            where_op_logits,
            where_start_logits,  # (batch_size, num_headers, question_length)
            where_end_logits,
        ) = logits
        (
            agg_target,
            sel_target,
            where_num_target,
            where_col_target,
            where_op_target,  # (batch_size, num_column, num_operator)
            where_start_target,
            where_end_target,
        ) = target

        batch_size = agg_target.size(0)
        loss = torch.tensor(0, dtype=torch.float32)
        # select
        loss = loss + f.cross_entropy(sel_logits, sel_target)
        selected_col = sel_logits.argmax(1)
        # agg
        loss = loss + f.cross_entropy(agg_logits[torch.arange(batch_size), selected_col], agg_target)

        # where num
        loss = loss + f.cross_entropy(where_num_logits, where_num_target)
        # where column
        for b in range(batch_size):
            for masked_col_logits, col_target in zip(where_col_logits, where_col_target):  # type: Tensor, Tensor
                if col_target.size(0) == 0:
                    continue
                col_logits = masked_col_logits.masked_select(masked_col_logits != 0)
                one_hot_col_target = torch.zeros_like(col_logits).scatter_(0, col_target, 1)
                pos_weight = torch.empty_like(col_logits).fill_(3)
                loss = loss + f.binary_cross_entropy_with_logits(col_logits, one_hot_col_target, pos_weight=pos_weight)

        # where op
        op_logits = torch.cat([logits[index] for logits, index in zip(where_op_logits, where_col_target)])
        op_target = torch.cat(where_op_target)
        loss = loss + f.cross_entropy(op_logits, op_target)

        # where start value
        start_logits = torch.cat([logits[index] for logits, index in zip(where_start_logits, where_col_target)])
        start_target = torch.cat(where_start_target)
        loss = loss + f.cross_entropy(start_logits, start_target)
        end_logits = torch.cat([logits[index] for logits, index in zip(where_end_logits, where_col_target)])
        end_target = torch.cat(where_end_target)
        loss = loss + f.cross_entropy(end_logits, end_target)
        return loss