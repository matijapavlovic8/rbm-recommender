import torch
import torch.nn.functional as F

def evaluate_rbm(rbm, test_data):
    """
    Evaluate the RBM on the test set by calculating the reconstruction loss.

    Args:
        rbm: Trained RBM instance.
        test_data: Test set data as a PyTorch tensor.

    Returns:
        test_loss: Average reconstruction loss on the test set.
    """
    test_loss = 0.0
    with torch.no_grad():
        for user_vector in test_data:
            user_vector = user_vector.float()
            v0 = user_vector.clone()
            h_prob, h_sample = rbm.forward(v0)
            vk, _ = rbm.backward(h_sample)
            loss = F.mse_loss(v0, vk)
            test_loss += loss.item()

    test_loss /= len(test_data)
    return test_loss