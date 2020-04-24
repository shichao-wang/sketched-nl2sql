"""
nn utilities package
"""
from torch import Tensor
import torch


def compute_mask(tensor: Tensor, dim=-1):
    """ compute mask """
    return (tensor.sum(dim=dim) != 0).bool()


def mm_attention(m_1: Tensor, m_2: Tensor, mask_1: Tensor, mask_2: Tensor):
    """ compute attention between matrixes
    :param m_1: (B, a, h)
    :param m_2: (B, b, h)
    :param mask_1: (B, a)
    :param mask_2: (B, b)
    :return: (B, a. b)
    """
    # Shape: (B, a, b)
    attention = m_1 @ m_2.transpose(1, 2)
    attention = attention.masked_fill(
        ~mask_1.unsqueeze(dim=-1).bool(), -float("inf")
    )
    attention = attention.masked_fill(
        ~mask_2.unsqueeze(dim=1).bool(), -float("inf")
    )
    weight = attention.softmax(dim=-1)
    weight = weight.masked_fill(torch.isnan(weight), 0)
    return weight


def make_weight(v: Tensor, mask: Tensor):
    """ compute attention between matrixes
    :param v: (B, a)
    :param mask: (B, a)
    :return: (B, a)
    """
    attention = v
    attention = attention.masked_fill(~mask.bool(), -float("inf"))
    weight = attention.softmax(dim=-1)
    return weight
