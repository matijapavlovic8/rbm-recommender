from src.eval import evaluate_rbm, evaluate_dbn
from src.data_preprocessing import load_data, preprocess_data
from src.rbm_model import RBM, DBN
from src.training import train_rbm, train_dbn
from src.plot_utils import plot_training_loss
from sklearn.model_selection import train_test_split
import torch


def main():
    file_path = "../data/ml-100k/u.data"
    data = load_data(file_path)
    interaction_matrix = preprocess_data(data)


    interaction_tensor = torch.tensor(interaction_matrix.values, dtype=torch.float32)

    train_data, test_data = train_test_split(interaction_tensor.numpy(), test_size=0.2, random_state=42)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_visible = interaction_tensor.shape[1]
    num_hidden = 400

    # RBM
    rbm = RBM(num_visible=num_visible, num_hidden=num_hidden, device=device)

    losses = train_rbm(rbm, train_data, epochs=40, learning_rate=0.01, batch_size=32)

    with open('../models/rbm.th', 'wb') as f:
        torch.save({
            'rbm_w': rbm.W,
            'rbm_v': rbm.v_bias,
            'rbm_h': rbm.h_bias,
            'num_hidden': num_hidden,
            'num_visible': num_visible
        }, f)

    plot_training_loss(losses, title="RBM Training Reconstruction Loss", xlabel="Epoch", ylabel="Reconstruction Loss")

    loss, acc = evaluate_rbm(rbm, test_data, device)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc*100:.2f}%")

    
    # DBN
    rbm_for_dbn = RBM.load_from_files("../models/rbm.th", device=device)
    
    num_hidden2 = 200
    dbn = DBN(rbm_for_dbn, num_hidden2, device=device)
    loss, acc = evaluate_dbn(dbn, test_data, device)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc*100:.2f}%")

    losses = train_dbn(dbn, train_data, epochs=20, learning_rate=0.01, batch_size=32)
    
    with open('../models/dbn.th', 'wb') as f:
        torch.save({
            'rbm1_w': dbn.rbm1.W,
            'rbm1_v': dbn.rbm1.v_bias,
            'rbm1_h': dbn.rbm1.h_bias,
            'rbm2_w': dbn.rbm2.W,
            'rbm2_v': dbn.rbm2.v_bias,
            'rbm2_h': dbn.rbm2.h_bias,
            'num_hidden1': dbn.rbm1.num_hidden,
            'num_hidden2': dbn.rbm2.num_hidden,
        }, f)

    plot_training_loss(losses, title="DBN Training Reconstruction Loss", xlabel="Epoch", ylabel="Reconstruction Loss")

    loss, acc = evaluate_dbn(dbn, test_data, device)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc*100:.2f}%")



if __name__ == "__main__":
    main()