import torch
import torch.nn.functional as F
from src.utils import quantize


def evaluate_rbm(rbm, test_data, device, k=10):
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
            v0 = user_vector.clone().to(device)
            
            h_prob_pos, h_sample_pos = rbm.forward(v0)
            h_sample_current = h_sample_pos
            for _ in range(k):
                v_prob_neg, v_sample_neg = rbm.backward(h_sample_current)
                h_prob_neg, h_sample_current = rbm.forward(v_sample_neg)
            
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



def evaluate_dbn(dbn, test_data, device, k=10):
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
            v0 = user_vector.clone().to(device)
            h_prob1_up, h_sample1_up, h_prob2_up, h_sample2_up = dbn.forward(v0)
            h_sample2_down = h_sample2_up.clone()
            for _ in range(k):
                h_prob1_down, h_sample1_down = dbn.rbm2.backward(h_sample2_down)
                h_prob2_down, h_sample2_down = dbn.rbm2.forward(h_sample1_down)
            
            v_prob_down, v_sample_down = dbn.rbm1.backward(h_sample1_down)

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