import torch
import torch.nn as nn
import warnings
from torch.autograd import Function
from typing import *
import tools_xue_cuda

class FurthestPointSampling(Function):
    @staticmethod
    def forward(ctx, xyz, npoint):
        # type: (Any, torch.Tensor, int) -> torch.Tensor
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = tools_xue_cuda.furthest_point_sampling(xyz, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()

class FurthestPointSamplingWithMask(Function):
    @staticmethod
    def forward(ctx, xyz, mask, npoint):
        r"""
        Uses iterative furthest point sampling to select a set of npoint features that have the largest
        minimum distance

        Parameters
        ----------
        xyz : torch.Tensor
            (B, N, 3) tensor where N > npoint
        npoint : int32
            number of features in the sampled set

        Returns
        -------
        torch.Tensor
            (B, npoint) tensor containing the set
        """
        out = tools_xue_cuda.furthest_point_sampling_with_mask(xyz, mask, npoint)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()

class GenerateGaussianHeatmap(Function):
    @staticmethod
    def forward(ctx, heatmap_size, points, sigma):
        out = tools_xue_cuda.generate_gaussian_heatmap(heatmap_size, points, sigma)

        ctx.mark_non_differentiable(out)

        return out

    @staticmethod
    def backward(ctx, grad_out):
        return ()



furthest_point_sample = FurthestPointSampling.apply
furthest_point_sample_with_mask = FurthestPointSamplingWithMask.apply
generate_gaussian_heatmap = GenerateGaussianHeatmap.apply
