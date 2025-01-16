import torch
import torch.nn as nn
from src.utils import quantize

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

    def forward(self, v, quantize_flag=True):
        """Sample hidden units given visible units."""
        v0 = v.clone().detach()
        if quantize_flag:
            v0 = quantize(v0)

        h_linear = torch.matmul(v0, self.W.t()) + self.h_bias
        h_prob = torch.sigmoid(h_linear)
        h_sample = (h_prob >= 0.5).float()

        return h_prob, h_sample

    def backward(self, h, quantize_flag=True):
        """Sample visible units given hidden units."""

        h0 = h.clone().detach()

        v_linear = torch.matmul(h0, self.W) + self.v_bias
        v_prob = torch.sigmoid(v_linear)
        if quantize_flag:
            v_sample = quantize(v_prob)
        else:
            v_sample = (v_prob >= 0.5).float()

        return v_prob, v_sample
    
    def reconstruct(self, v, k=10):
        v0 = v.clone().detach()
        h_prob_pos, h_sample_pos = self.forward(v0)
        h_sample_current = h_sample_pos
        for _ in range(k):
            v_prob_neg, v_sample_neg = self.backward(h_sample_current)
            h_prob_neg, h_sample_current = self.forward(v_prob_neg)
        
        return v_prob_neg, v_sample_neg

class DBN(nn.Module):
    def __init__(self, rbm1: RBM, num_hidden2, device='cpu'):
        super(DBN, self).__init__()
        self.device = device
        self.rbm1 = rbm1
        self.rbm2 = RBM(rbm1.num_hidden, num_hidden2, device)

        self.to(device)

    def forward(self, v):
        v0 = v.clone().detach()
        h_prob1, h_sample1 = self.rbm1.forward(v0)
        h_prob2, h_sample2 = self.rbm2.forward(h_sample1, quantize_flag=False)

        return h_prob1, h_sample1, h_prob2, h_sample2
    
    def reconstruct(self, v, k=10):
        v0 = v.clone().detach()
        h_prob1_up, h_sample1_up, h_prob2_up, h_sample2_up = self.forward(v0)
        h_sample2_down = h_sample2_up.clone()
        for _ in range(k):
            h_prob1_down, h_sample1_down = self.rbm2.backward(h_sample2_down, quantize_flag=False)
            h_prob2_down, h_sample2_down = self.rbm2.forward(h_sample1_down, quantize_flag=False)
        
        h_prob1_down = torch.sigmoid(torch.matmul(h_sample2_down, self.rbm2.W) + self.rbm1.h_bias)
        h_sample1_down = (h_sample1_down>=0.5).float()  

        v_prob_down, v_sample_down = self.rbm1.backward(h_sample1_down)

        return v_prob_down, v_sample_down
