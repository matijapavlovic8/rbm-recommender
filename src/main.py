from src.eval import evaluate_rbm, evaluate_dbn
from src.data_preprocessing import load_data, preprocess_data, load_movies
from src.models import RBM, DBN
from src.training import train_rbm, train_dbn
from src.plot_utils import plot_training_loss
from sklearn.model_selection import train_test_split
import torch
from src.utils import test_recommendation_ability, movie_from_tensor, recommend
import random



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
    num_hidden = 500

    # RBM
    rbm = RBM(num_visible=num_visible, num_hidden=num_hidden, device=device)

    losses = train_rbm(rbm, train_data, epochs=50, learning_rate=0.01, batch_size=16)

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

    hide_fraction = 0.2  # Hide 20% of rated movies
    accuracy = test_recommendation_ability(rbm, test_data, device, hide_fraction=hide_fraction, k=10)
    print(f"Recommendation accuracy: {accuracy*100:.2f}%")

    
    # DBN
    rbm_for_dbn = RBM.load_from_files("../models/rbm.th", device=device)
    
    num_hidden2 = 200
    dbn = DBN(rbm_for_dbn, num_hidden2, device=device)

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


    hide_fraction = 0.2  # Hide 20% of rated movies
    accuracy = test_recommendation_ability(dbn, test_data, device, hide_fraction=hide_fraction, k=10)
    print(f"Recommendation accuracy: {accuracy*100:.2f}%")

    movies = load_movies("data\\ml-100k\\u.item")
    random_user = random.randint(1, len(test_data))
    rates = test_data[random_user]
    movie_from_tensor(rates, movies)

    _,h = rbm_for_dbn.forward(rates)
    probs, v = rbm_for_dbn.backward(h)

    watched = torch.zeros_like(rates)
    watched[rates != 0] = 1
    to_recommend = recommend(watched, probs, 5)
    movie_from_tensor(to_recommend.cpu(), movies)

    h_prob1_up, h_sample1_up, h_prob2_up, h_sample2_up = dbn.forward(watched)
    h_sample2_down = h_sample2_up.clone()
    for _ in range(1):
        h_prob1_down, h_sample1_down = dbn.rbm2.backward(h_sample2_down)
        h_prob2_down, h_sample2_down = dbn.rbm2.forward(h_sample1_down)

    probs, v = dbn.rbm1.backward(h_sample1_down)
    to_recommend = recommend(watched, probs, 5)
    movie_from_tensor(to_recommend.cpu(), movies)



if __name__ == "__main__":
    main()