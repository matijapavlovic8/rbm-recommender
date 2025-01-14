import torch
import torch.nn.functional as F


def evaluate_rbm(rbm, test_data, device):
    """
    Evaluate the RBM on the test set by calculating the reconstruction loss.

    Args:
        rbm: Trained RBM instance.
        test_data: Test set data as a PyTorch tensor.

    Returns:
        test_loss: Average reconstruction loss on the test set.
    """
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for user_vector in test_data:
            user_vector = user_vector.float().to(device)
            v0 = user_vector.clone().to(device)
            h_prob, h_sample = rbm.forward(v0)
            _, vk = rbm.backward(h_sample)
            equal_elements = (v0 == vk)
            num_equal = equal_elements.sum().item()
            correct += num_equal
            total += vk.shape[0]
            loss = F.mse_loss(v0, vk)
            test_loss += loss.item()

    test_loss /= len(test_data)
    acc = correct/total
    return test_loss, acc