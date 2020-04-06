"""
nn utilities package
"""
from torch import Tensor


def compute_mask(tensor: Tensor, dim=-1):
    """ compute mask """
    return (tensor.sum(dim=dim) != 0).bool()
