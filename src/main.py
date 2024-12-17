from src.eval import evaluate_rbm
from src.data_preprocessing import load_data, preprocess_data
from src.rbm_model import RBM
from src.training import train_rbm
from src.plot_utils import plot_training_loss
from sklearn.model_selection import train_test_split
import torch


def main():
    file_path = "../data/ml-100k/u.data"
    data = load_data(file_path)
    interaction_matrix = preprocess_data(data)


    interaction_tensor = torch.tensor(interaction_matrix.values, dtype=torch.float32)
    interaction_tensor = (interaction_tensor - interaction_tensor.mean()) / interaction_tensor.std()

    train_data, test_data = train_test_split(interaction_tensor.numpy(), test_size=0.2, random_state=42)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_visible = interaction_tensor.shape[1]
    num_hidden = 100
    rbm = RBM(num_visible=num_visible, num_hidden=num_hidden, device=device)

    losses = train_rbm(rbm, train_data, epochs=20, learning_rate=0.01, k=10, batch_size=32)

    plot_training_loss(losses, title="RBM Training Reconstruction Loss", xlabel="Epoch", ylabel="Reconstruction Loss")

    print(evaluate_rbm(rbm, test_data))


if __name__ == "__main__":
    main()