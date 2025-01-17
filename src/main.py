from src.eval import evaluate_rbm, evaluate_dbn
from src.data_preprocessing import load_data, preprocess_data, load_movies
from src.models import GaussianBernoulliRBM, DBN
from src.plot_utils import plot_training_loss
from sklearn.model_selection import train_test_split
import torch
from src.utils import test_recommendation_ability, movie_from_tensor, recommend, set_global_seed
import random

set_global_seed(42)

def main():
    file_path = "../data/ml-100k/u.data"
    data = load_data(file_path)
    interaction_matrix = preprocess_data(data)

    interaction_tensor = torch.tensor(interaction_matrix.values, dtype=torch.float32)

    train_data, test_data = train_test_split(interaction_tensor.numpy(), test_size=0.3, random_state=42)

    train_data = torch.tensor(train_data, dtype=torch.float32)
    test_data = torch.tensor(test_data, dtype=torch.float32)

    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    num_visible = interaction_tensor.shape[1]
    num_hidden = 300

    # RBM
    rbm = GaussianBernoulliRBM(num_visible=num_visible, num_hidden=num_hidden, device=device, sigma=0.1)

    losses = rbm.train_rbm(train_data, batch_size=16, epochs=70, learning_rate=0.002, device=device)

    with open('../models/rbm.th', 'wb') as f:
        torch.save({
            'rbm_w': rbm.W,
            'rbm_v': rbm.v_bias,
            'rbm_h': rbm.h_bias,
           'num_hidden': num_hidden,
            'num_visible': num_visible,
            'sigma': rbm.sigma
        }, f)

    plot_training_loss(losses, title="RBM Training Reconstruction Loss", xlabel="Epoch", ylabel="Reconstruction Loss")

    loss, acc = evaluate_rbm(rbm, test_data, device)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc*100:.2f}%")


    # DBN 
    num_hidden2 = 200
    rbm_layers = [num_visible, num_hidden, num_hidden2]
    dbn = DBN(rbm_layers, rbm_path='../models/rbm.th', device=device)

    dbn.pretrain(train_data, epochs=[60,60], learning_rate=[0.03, 0.003], batch_size=16, device=device)
    
    DBN.save_dbn(dbn, '../models/dbn.th')

    loss, acc = evaluate_dbn(dbn, test_data, device)
    print(f"Validation loss: {loss:.4f}")
    print(f"Validation accuracy: {acc*100:.2f}%")


    hide_fraction = 0.3  # Hide 30% of rated movies
    accuracy_rbm, accuracy_dbn = test_recommendation_ability(rbm, dbn, test_data, device, hide_fraction, k=10)
    print(f"RBM Recommendation accuracy: {accuracy_rbm*100:.2f}%")
    print(f"DBN Recommendation accuracy: {accuracy_dbn*100:.2f}%")

    # movies = load_movies("data\\ml-100k\\u.item")
    # random_user = random.randint(1, len(test_data))
    # rates = test_data[random_user]
    # movie_from_tensor(rates, movies)

    # _,h = rbm_for_dbn.forward(rates)
    # probs, v = rbm_for_dbn.backward(h)

    # watched = torch.zeros_like(rates)
    # watched[rates != 0] = 1
    # to_recommend = recommend(watched, probs, 5)
    # movie_from_tensor(to_recommend.cpu(), movies)

    # h_prob1_up, h_sample1_up, h_prob2_up, h_sample2_up = dbn.forward(watched)
    # h_sample2_down = h_sample2_up.clone()
    # for _ in range(1):
    #     h_prob1_down, h_sample1_down = dbn.rbm2.backward(h_sample2_down)
    #     h_prob2_down, h_sample2_down = dbn.rbm2.forward(h_sample1_down)

    # probs, v = dbn.rbm1.backward(h_sample1_down)
    # to_recommend = recommend(watched, probs, 5)
    # movie_from_tensor(to_recommend.cpu(), movies)



if __name__ == "__main__":
    main()