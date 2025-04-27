# MIT License
# Copyright (c) 2017 Christian Cosgrove
# Downloaded from https://github.com/christiancosgrove/pytorch-spectral-normalization-gan

import torch
import torch.nn.functional as F
from torch import nn
from torch.nn import Parameter


def l2normalize(v, eps=1e-12):
    """🧮 L2-normalize a vector with small epsilon for numerical stability"""
    return v / (v.norm() + eps)


class SpectralNorm(nn.Module):
    """⚖️ Spectral Normalization wrapper for stable GAN training

    Parameters:
        module (nn.Module): the layer to wrap (e.g. Conv2d, Linear)
        name (str): parameter name to normalize (default: 'weight')
        power_iterations (int): number of power iterations to approximate the spectral norm
    """

    def __init__(self, module, name="weight", power_iterations=1):
        super(SpectralNorm, self).__init__()
        self.module = module
        self.name = name
        self.power_iterations = power_iterations
        if not self._made_params():
            self._make_params()

    def _update_u_v(self):
        """🔁 Update the approximation of singular vectors u and v via power iteration"""
        u = getattr(self.module, self.name + "_u")
        v = getattr(self.module, self.name + "_v")
        w = getattr(self.module, self.name + "_bar")

        height = w.data.shape[0]
        for _ in range(self.power_iterations):
            v.data = l2normalize(torch.mv(torch.t(w.view(height, -1).data), u.data))
            u.data = l2normalize(torch.mv(w.view(height, -1).data, v.data))

        # compute sigma (spectral norm estimate)
        sigma = u.dot(w.view(height, -1).mv(v))
        setattr(self.module, self.name, w / sigma.expand_as(w))

    def _made_params(self):
        """✅ Check if spectral norm params are already registered"""
        try:
            getattr(self.module, self.name + "_u")
            getattr(self.module, self.name + "_v")
            getattr(self.module, self.name + "_bar")
            return True
        except AttributeError:
            return False

    def _make_params(self):
        """🧠 Register u, v, and w_bar as module parameters"""
        w = getattr(self.module, self.name)

        height = w.data.shape[0]
        width = w.view(height, -1).data.shape[1]

        u = Parameter(w.data.new(height).normal_(0, 1), requires_grad=False)
        v = Parameter(w.data.new(width).normal_(0, 1), requires_grad=False)
        u.data = l2normalize(u.data)
        v.data = l2normalize(v.data)
        w_bar = Parameter(w.data)

        del self.module._parameters[self.name]

        self.module.register_parameter(self.name + "_u", u)
        self.module.register_parameter(self.name + "_v", v)
        self.module.register_parameter(self.name + "_bar", w_bar)

    def forward(self, *args):
        """🚀 Forward pass with updated spectral normalization"""
        self._update_u_v()
        return self.module(*args)
