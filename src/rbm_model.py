import torch
import torch.nn as nn


class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden, device='cpu'):
        super(RBM, self).__init__()
        self.device = device
        self.W = torch.randn(num_hidden, num_visible).to(device)
        self.h_bias = torch.zeros(num_hidden).to(device)
        self.v_bias = torch.zeros(num_visible).to(device)

        self.to(device)

    def forward(self, v):
        """Sample hidden units given visible units."""
        v = v.float().to(self.device)

        h_linear = torch.matmul(v, self.W.t()) + self.h_bias
        h_prob = torch.sigmoid(h_linear)
        h_sample = torch.bernoulli(h_prob)

        return h_prob, h_sample

    def backward(self, h):
        """Sample visible units given hidden units."""

        h = h.float().to(self.device)

        v_linear = torch.matmul(h, self.W) + self.v_bias
        v_prob = torch.sigmoid(v_linear)
        v_sample = torch.bernoulli(v_prob)

        return v_prob, v_sample
