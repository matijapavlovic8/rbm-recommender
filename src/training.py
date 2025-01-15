import torch
import torch.nn.functional as F
import torch.nn as nn
import torch.utils.data
from tqdm import tqdm


def train_rbm(rbm, data, epochs=20, learning_rate=0.001, k=10, batch_size=32):
    """
    Training loop with mini-batching, TQDM progress tracking

    Args:
        rbm: RBM model
        data: Training tensor
        epochs: Number of training epochs
        learning_rate: Learning rate
        k: Contrastive Divergence steps
        batch_size: Mini-batch size
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    losses = []
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)

        for batch in progress_bar:
            batch = batch.to(device)

            h_prob_pos, h_sample_pos = rbm.forward(batch)

            h_sample_current = h_sample_pos
            for _ in range(k):
                v_prob_neg, v_sample_neg = rbm.backward(h_sample_current)
                h_prob_neg, h_sample_current = rbm.forward(v_sample_neg)

            loss = F.mse_loss(batch, v_sample_neg)
            epoch_loss += loss.item()

            positive_grad = torch.matmul(h_sample_pos.t(), batch)
            negative_grad = torch.matmul(h_sample_current.t(), v_sample_neg)

            dw = (positive_grad - negative_grad) / batch.size(0)
            rbm.W = rbm.W + learning_rate * dw
            rbm.v_bias = rbm.v_bias + learning_rate * (torch.mean(batch - v_sample_neg, dim=0))
            rbm.h_bias = rbm.h_bias + learning_rate * (torch.mean(h_sample_pos - h_sample_current, dim=0))

            progress_bar.set_postfix({'Loss': loss.item()})

        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)

        print(f"Reconstruction Loss: {epoch_loss:.4f}")

    return losses



def train_dbn(dbn, data, epochs=20, learning_rate=0.001, k=10, batch_size=32):
    """
    Training loop with mini-batching, TQDM progress tracking

    Args:
        dbn: DBN model
        data: Training tensor
        epochs: Number of training epochs
        learning_rate: Learning rate
        k: Contrastive Divergence steps
        batch_size: Mini-batch size
        This training process doesn't update rbm1 parameters
    """
    device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    data = data.to(device)

    losses = []
    data_loader = torch.utils.data.DataLoader(data, batch_size=batch_size, shuffle=True)

    for epoch in range(epochs):
        epoch_loss = 0.0
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}", leave=True)

        for batch in progress_bar:
            batch = batch.to(device)

            h_prob1_up, h_sample1_up, h_prob2_up, h_sample2_up = dbn.forward(batch)
            h_sample2_down = h_sample2_up.clone()
            for _ in range(k):
                h_prob1_down, h_sample1_down = dbn.rbm2.backward(h_sample2_down)
                h_prob2_down, h_sample2_down = dbn.rbm2.forward(h_sample1_down)
            
            _, v_sample_down = dbn.rbm1.backward(h_sample1_down)

            loss = F.mse_loss(v_sample_down, batch)
            epoch_loss += loss.item()

            positive_grad = torch.matmul(h_sample2_up.t(), h_sample1_up)
            negative_grad = torch.matmul(h_sample2_down.t(), h_sample1_down)

            dw = (positive_grad - negative_grad) / batch.size(0)
            dbn.rbm2.W = dbn.rbm2.W + learning_rate * dw
            dbn.rbm2.v_bias = dbn.rbm2.v_bias + learning_rate * (torch.mean(h_sample1_up - h_sample1_down, dim=0))
            dbn.rbm1.h_bias = dbn.rbm2.v_bias.clone()
            dbn.rbm2.h_bias = dbn.rbm2.h_bias + learning_rate * (torch.mean(h_sample2_up - h_sample2_down, dim=0))

            progress_bar.set_postfix({'Loss': loss.item()})

        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)

        print(f"Reconstruction Loss: {epoch_loss:.4f}")

    return losses