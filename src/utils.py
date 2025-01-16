import torch

def recommend(watched, probs, n):
    """
    Function that gets an binary input tensor, where 1 represents the
    index of a rated movie, and 0 otherwise. 
    Returns a tensor size len_movies, where it recomends n movies
    with indices 1 on recommended movies, and 0 elsewhere.

    Arguments:
        - watched: binary input tensor of watched movies
        - probs: probability output of backward pass of RBM,
        containing model probabilities for every movie.
        - n: number of recommendations
    """
    probs = quantize(probs.clone())
    probs[watched == 1] = 0

    # Filter probabilities >= 0.6
    valid_probs_mask = probs >= 0.6
    filtered_probs = probs * valid_probs_mask

    # Get the non-zero indices and their corresponding probabilities
    non_zero_indices = torch.nonzero(filtered_probs, as_tuple=True)[0]
    non_zero_probs = filtered_probs[non_zero_indices]

    top_n = min(n, len(non_zero_probs))
    _, top_indices = torch.topk(non_zero_probs, top_n)

    indices = non_zero_indices[top_indices]

    result_tensor = torch.zeros_like(probs)
    result_tensor[indices] = 1
    return result_tensor


def movie_from_tensor(tensor, movies):
    """
    Function that prints out movie genre and name 
    for every movie inside binary tensor.
    Movies whose indexes contain 1 in tensor will be printed out.

    Arguments:
        - tensor: binary tensor with movies
        - movies: dataframe containing movie info
    """
    indices = torch.nonzero(tensor >= 0.6).squeeze()
    if len(indices) == 0:
        print("No recommendations.")
        return

    movie_ids = indices+1

    for movie_id in movie_ids:
        row = movies[movies['movie_id'] == movie_id.item()]  # Convert tensor to Python int
        if not row.empty:
            columns_with_ones = row.columns[row.iloc[0] == 1].tolist()

            print(f"'{row['movie_title'].values[0]}' has the following genres:")
            print(", ".join(columns_with_ones[1:]))  # Exclude 'movie_id' from genres
            print("-" * 50)



def test_recommendation_ability(rbm, dbn, data, device, hide_fraction=0.2, k=10):
    """
    Test the RBM's ability to recommend movies by hiding some ratings
    and checking if the model predicts them as high-probability recommendations.

    Args:
        rbm: Trained RBM instance.
        data: User-movie interaction tensor.
        device: Device to use (e.g., 'cuda' or 'cpu').
        hide_fraction: Fraction of rated movies to hide for each user.
        k: Number of Gibbs sampling steps for evaluation.
    Returns:
        accuracy: Proportion of correctly predicted ratings among top-n recommendations.
    """
    device = torch.device(device)
    data = data.to(device)
    correct_top_n_rbm = 0
    total_top_n_rbm = 0
    correct_top_n_dbn = 0
    total_top_n_dbn = 0

    with torch.no_grad():
        for user_vector in data:
            user_vector = user_vector.clone().float().to(device)

            rated_indices = torch.where(user_vector > 0)[0]

            if len(rated_indices) < 30:
                continue

            num_to_hide = max(1, int(len(rated_indices) * hide_fraction))
            hidden_indices = rated_indices[torch.randperm(len(rated_indices))[:num_to_hide]]

            test_vector = user_vector.clone()
            test_vector[hidden_indices] = 0  # Hide selected ratings

            filtered_watched = torch.zeros_like(user_vector)
            for i in range(len(filtered_watched)):
                if i in rated_indices and i not in hidden_indices:
                    filtered_watched[i] = 1

            top_n = num_to_hide

            v_prob_down, v_sample_down = rbm.reconstruct(test_vector, k=k)
            predictions = quantize(v_prob_down.clone())

            recommendations = recommend(filtered_watched, predictions, top_n)
            recomm_indices = torch.where(recommendations == 1)[0]

            for r in recomm_indices:
                if r in hidden_indices:
                    if user_vector[r] >= 0.6:
                        correct_top_n_rbm += 1
                    total_top_n_rbm += 1

                    
            v_prob_down, v_sample_down = dbn.reconstruct(test_vector, k=k)
            predictions = quantize(v_prob_down.clone())

            recommendations = recommend(filtered_watched, predictions, top_n)
            recomm_indices = torch.where(recommendations == 1)[0]

            for r in recomm_indices:
                if r in hidden_indices:
                    if user_vector[r] >= 0.6:
                        correct_top_n_dbn += 1
                    total_top_n_dbn += 1


    accuracy_rbm = correct_top_n_rbm / total_top_n_rbm if total_top_n_rbm > 0 else 0
    accuracy_dbn = correct_top_n_dbn / total_top_n_dbn if total_top_n_dbn > 0 else 0
    return accuracy_rbm, accuracy_dbn
           


def quantize(v_prob, valid_levels=[0, 0.2, 0.4, 0.6, 0.8, 1]):
    # Find the closest valid level for each value in v_prob
    quantized = torch.zeros_like(v_prob)
    for i, level in enumerate(valid_levels):
        if i == 0:
            quantized = torch.where(v_prob < (valid_levels[i + 1] + level) / 2, level, quantized)
        elif i == len(valid_levels) - 1:
            quantized = torch.where(v_prob >= (valid_levels[i - 1] + level) / 2, level, quantized)
        else:
            quantized = torch.where(
                (v_prob >= (valid_levels[i - 1] + level) / 2) & (v_prob < (valid_levels[i + 1] + level) / 2),
                level,
                quantized,
            )
    return quantized



def set_global_seed(seed):
    import random
    import numpy as np
    import torch

    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    if torch.cuda.is_available():
        torch.cuda.manual_seed(seed)
        torch.cuda.manual_seed_all(seed)
    # Enable deterministic behavior in PyTorch (if required)
    torch.backends.cudnn.deterministic = True
    torch.backends.cudnn.benchmark = False
