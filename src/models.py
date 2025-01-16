import torch
import torch.nn as nn


class RBM(nn.Module):
    def __init__(self, num_visible, num_hidden, device='cpu'):
        super(RBM, self).__init__()
        self.device = device
        self.W = torch.randn(num_hidden, num_visible).to(device)
        self.h_bias = torch.zeros(num_hidden).to(device)
        self.v_bias = torch.zeros(num_visible).to(device)
        self.num_hidden = num_hidden    

        self.to(device)

    @staticmethod
    def load_from_files(config_path, device='cpu'):
        config = torch.load(config_path, map_location=device)
        rbm = RBM(config['num_visible'], config['num_hidden'], device)
        rbm.W = config['rbm_w'].to(device)
        rbm.v_bias = config['rbm_v'].to(device)
        rbm.h_bias = config['rbm_h'].to(device)

        return rbm

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

class DBN(nn.Module):
    def __init__(self, rbm1: RBM, num_hidden2, device='cpu'):
        super(DBN, self).__init__()
        self.device = device
        self.rbm1 = rbm1
        self.rbm2 = RBM(rbm1.num_hidden, num_hidden2, device)

        self.to(device)

    def forward(self, v):
        h_prob1, h_sample1 = self.rbm1.forward(v)
        h_prob2, h_sample2 = self.rbm2.forward(h_sample1)

        return h_prob1, h_sample1, h_prob2, h_sample2