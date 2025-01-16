import numpy as np
import torch
from sklearn.model_selection import train_test_split
from src.data_preprocessing import load_data, preprocess_data
from src.rbm_model import DBN, RBM


# Evaluating the recommendations for true and false positives
def evaluate_recommendations(model, user_idx, original_data, masked_data, top_k=10):
    # Ensure the input data has the correct shape before passing it to the model
    masked_data = masked_data.to(torch.float32)  # Ensure the correct type (if not already)

    with torch.no_grad():
        predictions = model(masked_data)

    # Get the liked and disliked movies for the user
    original_liked = set(np.where(original_data[user_idx, 1].numpy() == 1)[0])  # Liked movies (second bit)
    original_disliked = set(np.where(original_data[user_idx, 1].numpy() == 0)[0])  # Disliked movies

    # Get recommended movies (top K from predictions)
    recommended_movies = torch.topk(predictions[user_idx], top_k).indices.cpu().numpy()
    recommended_set = set(map(int, recommended_movies.flatten()))

    # Count true positives: recommended and liked by user
    true_positives = original_liked & recommended_set

    # Count false positives: recommended but disliked by user
    false_positives = original_disliked & recommended_set

    return true_positives, false_positives


# Main function to execute the steps
def main():
    file_path = "../data/ml-100k/u.data"
    data = load_data(file_path)
    interaction_matrix = preprocess_data(data)
    interaction_tensor = torch.tensor(interaction_matrix.values, dtype=torch.float32)

    # Train-test split
    _, test_data = train_test_split(interaction_tensor.numpy(), test_size=0.2, random_state=42)
    test_data = torch.tensor(test_data, dtype=torch.float32)
    print(test_data.shape)
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

    # Load the pre-trained RBM model
    print(interaction_tensor.shape[1])
    rbm = RBM(num_visible=3364, num_hidden=400, device=device)
    rbm_checkpoint = torch.load('../models/rbm2.th', map_location=device, weights_only=False)
    rbm.W = rbm_checkpoint['rbm_w']
    rbm.v_bias = rbm_checkpoint['rbm_v']
    rbm.h_bias = rbm_checkpoint['rbm_h']
    print(f"Shape of rbm.W: {rbm_checkpoint['rbm_w'].shape}")  # Should be (400, 3364)
    print(f"Shape of rbm.v_bias: {rbm_checkpoint['rbm_v'].shape}")  # Should be (3364,)
    print(f"Shape of rbm.h_bias: {rbm_checkpoint['rbm_h'].shape}")  # Should be (400,)

    rbm.eval()

    # Load the pre-trained DBN model (stacked RBMs)
    num_hidden2 = 200
    dbn = DBN(rbm, num_hidden2, device=device)
    dbn_checkpoint = torch.load('../models/dbn.th', map_location=device, weights_only=False)

    dbn.rbm1.W = dbn_checkpoint['rbm1_w']
    dbn.rbm1.v_bias = dbn_checkpoint['rbm1_v']
    dbn.rbm1.h_bias = dbn_checkpoint['rbm1_h']

    dbn.rbm2.W = dbn_checkpoint['rbm2_w']
    dbn.rbm2.v_bias = dbn_checkpoint['rbm2_v']
    dbn.rbm2.h_bias = dbn_checkpoint['rbm2_h']

    dbn.eval()

    # Select a random user for testing
    np.random.seed(42)
    user_idx = np.random.choice(len(test_data))

    # Mask half of their rated movies
    masked_data = test_data.clone()

    # Get rated movies (entire row for the selected user)
    rated_movies = np.where(test_data[user_idx, 0::2].numpy() == 1)[0]  # Get rated movies (first bit)

    num_to_mask = len(rated_movies) // 2  # Mask half of the rated movies
    masked_movies = np.random.choice(rated_movies, num_to_mask, replace=False)

    # Masking rated and liked bits (first and second columns)
    for movie_idx in masked_movies:
        # Mask the rated and liked bit (corresponding bits for the movie)
        masked_data[user_idx, movie_idx * 2:movie_idx * 2 + 2] = 0

    # Ensure no unintended dimension changes after masking
    print(masked_data.shape)

    # Evaluate the model (RBM first)
    print("Evaluating RBM:")
    rbm_true_positives, rbm_false_positives = evaluate_recommendations(rbm, user_idx, test_data, masked_data, top_k=10)
    print(f"RBM True Positives: {rbm_true_positives}")
    print(f"RBM False Positives: {rbm_false_positives}")

    # Evaluate the model (DBN second)
    print("\nEvaluating DBN:")
    dbn_true_positives, dbn_false_positives = evaluate_recommendations(dbn, user_idx, test_data, masked_data, top_k=10)
    print(f"DBN True Positives: {dbn_true_positives}")
    print(f"DBN False Positives: {dbn_false_positives}")


if __name__ == "__main__":
    main()
