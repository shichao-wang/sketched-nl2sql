"""define loss"""
from typing import List, Tuple

import torch
from torch import Tensor, nn
from torch.nn import functional as f


class QueryLoss(nn.Module):
    """ query loss """

    # noinspection PyAugmentAssignment
    # pylint: disable=arguments-differ
    def forward(
        self,
        logits: Tuple[Tensor, ...],
        target: Tuple[
            Tensor,
            Tensor,
            Tensor,
            List[Tensor],
            List[Tensor],
            List[Tensor],
            List[Tensor],
        ],
    ):
        """
        :param logits: tuple
        :param target: tuple
        :return:
        """
        (
            sel_logits,
            agg_logits,
            where_num_logits,
            where_col_logits,
            where_op_logits,
            where_start_logits,  # (batch_size, num_headers, question_length)
            where_end_logits,
        ) = logits
        (
            sel_target,
            agg_target,
            where_num_target,
            where_col_target,
            where_op_target,  # (batch_size, num_column, num_operator)
            where_start_target,
            where_end_target,
        ) = target

        selected_col = sel_logits.argmax(1).tolist()
        # select
        loss = f.cross_entropy(sel_logits, sel_target)
        # agg
        loss = loss + sel_agg_loss(agg_logits, agg_target, selected_col)
        # where num
        loss = loss + f.cross_entropy(where_num_logits, where_num_target)
        # where column
        loss = loss + where_col_loss(where_col_logits, where_col_target)
        # for b in range(batch_size):
        #     for masked_col_logits, col_target in zip(
        #         where_col_logits, where_col_target
        #     ):  # type: Tensor, Tensor
        #         if col_target.size(0) == 0:
        #             continue
        #         col_logits = masked_col_logits.masked_select(
        #             masked_col_logits != -float("inf")
        #         )
        #         one_hot_col_target = torch.zeros_like(col_logits).scatter_(
        #             0, col_target, 1
        #         )
        #         pos_weight = torch.empty_like(col_logits).fill_(3)
        #         loss = loss + f.binary_cross_entropy_with_logits(
        #             col_logits, one_hot_col_target, pos_weight=pos_weight
        #         )

        # where op
        op_logits = torch.cat(
            [
                logits[:num]
                for logits, num in zip(where_op_logits, where_num_target)
            ]
        )
        loss = loss + f.cross_entropy(op_logits, torch.cat(where_op_target))

        # where start value
        start_logits = torch.cat(
            [
                logits[:num]
                for logits, num in zip(where_start_logits, where_num_target)
            ]
        )
        start_target = torch.cat(where_start_target)
        loss = loss + f.cross_entropy(start_logits, start_target)

        end_logits = torch.cat(
            [
                logits[:num]
                for logits, num in zip(where_end_logits, where_num_target)
            ]
        )
        end_target = torch.cat(where_end_target)
        loss = loss + f.cross_entropy(end_logits, end_target)
        return loss


def sel_agg_loss(
    agg_logits: Tensor, agg_target: Tensor, selected_col: List[int]
):
    batch_size = len(selected_col)
    return f.cross_entropy(
        agg_logits[torch.arange(batch_size), selected_col], agg_target
    )


def where_col_loss(where_col_logits: Tensor, where_col_target: List[int]):

    loss = where_col_logits.new_tensor(0.0)
    for masked_logits, col_target in zip(where_col_logits, where_col_target):
        if col_target.size(0) == 0:
            continue
        col_logits = masked_logits.masked_select(
            masked_logits != -float("inf")
        )
        one_hot_target = torch.zeros_like(col_logits).scatter_(
            0, col_target, 1
        )
        pos_weight = torch.empty_like(col_logits).fill_(3)

        loss = loss + f.binary_cross_entropy_with_logits(
            col_logits, one_hot_target, pos_weight=pos_weight
        )

    return loss


def num_based_cond_loss(logits: Tensor, target: List[int]):
    """
    :param logits: (B, num_cond, logits)
    :param target: [B, (target,)]

    """
