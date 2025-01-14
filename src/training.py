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
        progress_bar = tqdm(data_loader, desc=f"Epoch {epoch}/{epochs}", leave=False)

        for batch in progress_bar:
            batch = batch.to(device)

            h_prob_pos, h_sample_pos = rbm.forward(batch)

            h_sample_current = h_sample_pos
            for _ in range(k):
                v_prob_neg, v_sample_neg = rbm.backward(h_sample_current)
                h_prob_neg, h_sample_current = rbm.forward(v_sample_neg)

            positive_grad = torch.matmul(h_sample_pos.t(), batch)
            negative_grad = torch.matmul(h_sample_current.t(), v_sample_neg)

            dw = (positive_grad - negative_grad) / batch.size(0)
            rbm.W = rbm.W + learning_rate * dw
            rbm.v_bias = rbm.v_bias + learning_rate * (torch.mean(batch - v_sample_neg, dim=0))
            rbm.h_bias = rbm.h_bias + learning_rate * (torch.mean(h_sample_pos - h_sample_current, dim=0))

            loss = F.mse_loss(batch, v_sample_neg)
            epoch_loss += loss.item()

            progress_bar.set_postfix({'Loss': loss.item()})

        epoch_loss /= len(data_loader)
        losses.append(epoch_loss)

        print(f"Reconstruction Loss: {epoch_loss:.4f}")

    return losses

