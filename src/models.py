import torch
import torch.nn as nn
from src.utils import quantize
from tqdm import tqdm
import torch.nn.functional as F
import torch.utils.data
from src.plot_utils import plot_training_loss

class GaussianBernoulliRBM(nn.Module):
    def __init__(self, num_visible, num_hidden, sigma=0.1, device='cpu'):
        super(GaussianBernoulliRBM, self).__init__()
        self.device = device
        self.W = torch.randn(num_hidden, num_visible).to(device) * 0.1
        self.h_bias = torch.zeros(num_hidden).to(device)
        self.v_bias = torch.zeros(num_visible).to(device)
        self.num_hidden = num_hidden    
        self.sigma = sigma  

        self.to(device)

    @staticmethod
    def load_from_files(config_path, device='cpu'):
        config = torch.load(config_path)
        rbm = GaussianBernoulliRBM(config['num_visible'], config['num_hidden'], device=device)
        rbm.W = config['rbm_w'].to(device)
        rbm.v_bias = config['rbm_v'].to(device)
        rbm.h_bias = config['rbm_h'].to(device)
        rbm.sigma = config['sigma']

        return rbm.to(device)

    def forward(self, v):
        """Sample hidden units given visible units."""
        v = v.to(self.device)
        h_linear = torch.matmul(v, self.W.t()) + self.h_bias
        h_prob = torch.sigmoid(h_linear)
        h_sample = torch.bernoulli(h_prob)

        return h_prob, h_sample

    def backward(self, h):
        """Sample visible units given hidden units."""
        h = h.to(self.device)
        mean_v = torch.matmul(h, self.W) * self.sigma + self.v_bias
        v_sample = mean_v + self.sigma * torch.randn_like(mean_v)
        return v_sample
    
    def reconstruct(self, v, k=1):
        h_prob_pos, h_sample_pos = self.forward(v)
        h_sample_current = h_sample_pos
        for _ in range(k):
            v_sample_neg = self.backward(h_sample_current)
            h_prob_neg, h_sample_current = self.forward(v_sample_neg)
        
        return v_sample_neg
    
    def train_rbm(self, data, batch_size=16, epochs=70, learning_rate=0.002, k=1, device='cpu'):
        print("\nTraining Gaussian-Bernoulli RBM")
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        losses = []
        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

            for batch in progress_bar:
                batch = batch.to(device)

                h_prob_pos, h_sample_pos = self.forward(batch)
                h_sample_current = h_sample_pos
                for _ in range(k):
                    v_sample_neg = self.backward(h_sample_current)
                    h_prob_neg, h_sample_current = self.forward(v_sample_neg)

                v_prob_neg_q = quantize(v_sample_neg)
                loss = F.mse_loss(batch, v_prob_neg_q)
                epoch_loss += loss.item()

                positive_grad = torch.matmul(h_prob_pos.t(), batch/self.sigma)
                negative_grad = torch.matmul(h_prob_neg.t(), v_sample_neg/self.sigma)

                dw = (positive_grad - negative_grad) / batch.size(0)
                self.W = self.W + learning_rate * dw
                self.v_bias = self.v_bias + learning_rate * torch.mean(batch - v_sample_neg, dim=0)
                self.h_bias = self.h_bias + learning_rate * torch.mean(h_sample_pos - h_prob_neg, dim=0)

                progress_bar.set_postfix({'Loss': loss.item()})

            epoch_loss /= len(data_loader)
            losses.append(epoch_loss)

            print(f"Reconstruction Loss: {epoch_loss:.4f}")

        return losses
    

class BinaryRBM(nn.Module):
    def __init__(self, num_visible, num_hidden, device='cpu'):
        super(BinaryRBM, self).__init__()
        self.device = device
        self.W = torch.randn(num_hidden, num_visible).to(device) * 0.1
        self.h_bias = torch.zeros(num_hidden).to(device)
        self.v_bias = torch.zeros(num_visible).to(device)

        self.to(device)

    def forward(self, v):
        """Sample hidden units given visible units."""
        v = v.to(self.device)
        h_linear = torch.matmul(v, self.W.t()) + self.h_bias
        h_prob = torch.sigmoid(h_linear)
        h_sample = torch.bernoulli(h_prob)
        return h_prob, h_sample

    def backward(self, h):
        """Sample visible units given hidden units."""
        h = h.to(self.device)
        v_linear = torch.matmul(h, self.W) + self.v_bias
        v_prob = torch.sigmoid(v_linear)
        v_sample = torch.bernoulli(v_prob)
        return v_prob, v_sample

    def reconstruct(self, v, k=1):
        """Reconstruct visible units by performing Gibbs sampling."""
        h_prob, h_sample = self.forward(v)
        for _ in range(k):
            v_prob, v_sample = self.backward(h_sample)
            h_prob, h_sample = self.forward(v_sample)
        return v_prob, v_sample
    
    def train_rbm(self, data, epochs=20, learning_rate=0.01, k=1, batch_size=16, device='cpu'):
        losses = []
        data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)
        for epoch in range(epochs):
            epoch_loss = 0.0
            progress_bar = tqdm(data_loader, desc=f"Epoch {epoch + 1}/{epochs}", leave=True)

            for batch in progress_bar:
                batch = batch.to(device)

                h_prob_pos, h_sample_pos = self.forward(batch)
                h_sample_current = h_sample_pos
                for _ in range(k):
                    v_prob_neg, v_sample_neg = self.backward(h_sample_current)
                    h_prob_neg, h_sample_current = self.forward(v_sample_neg)

                loss = F.mse_loss(batch, v_prob_neg)
                epoch_loss += loss.item()

                positive_grad = torch.matmul(h_sample_pos.t(), batch)
                negative_grad = torch.matmul(h_sample_current.t(), v_sample_neg)

                dw = (positive_grad - negative_grad) / batch.size(0)
                self.W = self.W + learning_rate * dw
                self.v_bias = self.v_bias + learning_rate * torch.mean(batch - v_sample_neg, dim=0)
                self.h_bias = self.h_bias + learning_rate * torch.mean(h_sample_pos - h_sample_current, dim=0)

                progress_bar.set_postfix({'Loss': loss.item()})

            epoch_loss /= len(data_loader)
            losses.append(epoch_loss)

            print(f"Reconstruction Loss: {epoch_loss:.4f}")

        return losses



class DBN(nn.Module):
    def __init__(self, rbm_layers, sigma=0.1, device='cpu', rbm_path=None):
        """
        Deep Belief Network with Gaussian-Bernoulli first layer and Binary-Binary subsequent layers,
        with shared biases between layers.
        
        Args:
            rbm_layers (list of int): Number of units in each RBM layer, including visible and hidden layers.
            sigma (float): Standard deviation for Gaussian visible units in the first RBM.
            device (str): Device to run the DBN on ('cpu' or 'cuda').
        """
        super(DBN, self).__init__()
        self.rbm_layers = rbm_layers
        self.device = device
        self.to(device)
        if rbm_path is not None:
            self.first_rbm = GaussianBernoulliRBM.load_from_files(rbm_path, device=device)
        else: self.first_rbm = GaussianBernoulliRBM(rbm_layers[0], rbm_layers[1], sigma=sigma, device=device)
        self.binary_rbms = nn.ModuleList([
            BinaryRBM(rbm_layers[i], rbm_layers[i + 1], device=device)
            for i in range(1, len(rbm_layers) - 1)
        ])
        
        # Link biases
        self.link_biases()

        self.first_trained = rbm_path is not None

        self.to(device)

    def link_biases(self):
        """ Link biases between layers so that v_bias of layer n is the same as h_bias of layer n-1."""
        if len(self.binary_rbms) > 0:
            self.binary_rbms[0].v_bias = self.first_rbm.h_bias

            for i in range(1, len(self.binary_rbms)):
                self.binary_rbms[i].v_bias = self.binary_rbms[i - 1].h_bias

    def forward(self, v):
        """
        Perform a forward pass through the stacked RBMs.
        """
        v = v.to(self.device)
        _, h_sample = self.first_rbm.forward(v)

        for rbm in self.binary_rbms:
            h_prob, h_sample = rbm.forward(v)

        return h_prob, h_sample

    def pretrain(self, data, epochs=[60,60], learning_rate=[0.01, 0.002], k=1, batch_size=16, device='cpu', plot=True):
        """
        Train the DBN layer by layer.
        """
        print("\nTraining Deep Belief Network")
        if not self.first_trained:
            print("Training Gaussian-Bernoulli RBM")
            self.first_rbm.train_rbm(data, batch_size=16, epochs=epochs[0], learning_rate=learning_rate[0], k=k, device=device)
        else:
            print("\nFirst RBM already trained")
        
        rbm_input = self.first_rbm.forward(data)[1]

        for layer_idx, rbm in enumerate(self.binary_rbms):
            print(f"\nTraining {layer_idx + 1}. Binary-Binary RBM layer")
            losses = rbm.train_rbm(rbm_input, epochs=epochs[1], learning_rate=learning_rate[1], k=k, batch_size=batch_size, device=device)
            if layer_idx == 0:
                self.first_rbm.h_bias = rbm.v_bias
            else: 
                self.binary_rbms[layer_idx - 1].h_bias = rbm.v_bias
            if plot:
                plot_training_loss(losses, title=f"RBM {layer_idx + 1} Training Reconstruction Loss", xlabel="Epoch", ylabel="Reconstruction Loss")
            rbm_input = rbm.forward(rbm_input)[1]

    def reconstruct(self, v, k=1):
        """
        Reconstruct the input by passing it through the DBN and backpropagating.
        """
        v = v.to(self.device)
        h = self.first_rbm.forward(v)[1]
        for rbm in self.binary_rbms[:-1]:
            h = rbm.forward(h)[1]

        last_rbm = self.binary_rbms[-1]
        h = last_rbm.forward(h)[1]
        for _ in range(k):
            _, v_sample = last_rbm.backward(h)
            _, h = last_rbm.forward(v_sample) 

        h = v_sample
        for rbm in reversed(self.binary_rbms[:-1]):
            h = rbm.backward(h)[1]

        h = self.first_rbm.backward(h)  

        return h
    
    @staticmethod
    def save_dbn(dbn, file_path):
        """
        Save a Deep Belief Network (DBN) to a file.
        
        Args:
            dbn: The DBN object to save.
            file_path: Path to the file where the DBN will be saved.
        """
        dbn_data = {
            'num_layers': len(dbn.rbm_layers) - 1,  # Number of RBMs
            'architecture': dbn.rbm_layers,  # List of layer sizes
        }
        dbn_data['rbm1'] = {
            'W': dbn.first_rbm.W,
            'v_bias': dbn.first_rbm.v_bias,
            'h_bias': dbn.first_rbm.h_bias,
            'sigma': dbn.first_rbm.sigma,
        }
        for i, rbm in enumerate(dbn.binary_rbms, start=2):
            dbn_data[f'rbm{i}'] = {
                'W': rbm.W,
                'v_bias': rbm.v_bias,
                'h_bias': rbm.h_bias,
            }
        with open(file_path, 'wb') as f:
            torch.save(dbn_data, f)