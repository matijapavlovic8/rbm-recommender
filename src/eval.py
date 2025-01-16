import torch
import torch.nn.functional as F
from src.utils import quantize
from src.models import RBM, DBN


def evaluate_rbm(rbm: RBM, test_data, device, k=10):
    """
    Evaluate the RBM on the test set by calculating the
    reconstruction loss and accuracy.

    Args:
        rbm: Trained RBM instance.
        test_data: Test set data as a PyTorch tensor.

    Returns:
        test_loss: Average reconstruction loss on the test set.
        acc: average accuracy of reconstruction on the test set.
    """
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for user_vector in test_data:
            user_vector = user_vector.float().to(device)
            v0 = user_vector.clone().detach().to(device)
            
            v_prob_neg, v_sample_neg = rbm.reconstruct(v0, k=k)
            
            vk = quantize(v_prob_neg)

            equal_elements = (v0 == vk)
            num_equal = equal_elements.sum().item()
            correct += num_equal
            total += vk.shape[0]
            loss = F.mse_loss(v0, vk)
            test_loss += loss.item()

    test_loss /= len(test_data)
    acc = correct/total
    return test_loss, acc



def evaluate_dbn(dbn: DBN, test_data, device, k=10):
    """
    Evaluate the DBN on the test set by calculating the
    reconstruction loss and accuracy.

    Args:
        dbn: Trained DBN instance.
        test_data: Test set data as a PyTorch tensor.

    Returns:
        test_loss: Average reconstruction loss on the test set.
        acc: average accuracy of reconstruction on the test set.
    """
    test_loss = 0.0
    correct = 0
    total = 0
    with torch.no_grad():
        for user_vector in test_data:
            user_vector = user_vector.float().to(device)
            v0 = user_vector.clone().detach().to(device)
            
            v_prob_down, v_sample_down = dbn.reconstruct(v0, k=k)

            vk = quantize(v_prob_down)

            equal_elements = (v0 == vk)
            num_equal = equal_elements.sum().item()
            correct += num_equal
            total += vk.shape[0]
            loss = F.mse_loss(v0, vk)
            test_loss += loss.item()

    test_loss /= len(test_data)
    acc = correct/total
    return test_loss, acc