import torch
import pandas as pd


def recommend(watched, probs, n):
    """
    Function that gets an binary input tensor, where 1 represents the
    index of a liked/watched movie, and 0 otherwise. 
    Returns a tensor size len_movies, where it recomends n movies
    with indices 1 on recommended movies, and 0 elsewhere.

    Arguments:
        - watched: binary input tensor of watched movies
        - probs: probability output of backward pass of RBM,
        containing model probabilities for every movie.
        - n: number of recommendations
    """
    probs[watched == 1] = 0
    _, indices = torch.topk(probs, n)
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
    indices = torch.nonzero(tensor == 1).squeeze()
    adjusted_indices = indices + 1  

    movie_ids = movies['movie_id'].iloc[adjusted_indices - 1].values  

    for movie_id in movie_ids:
        row = movies[movies['movie_id'] == movie_id]
        columns_with_ones = row.columns[row.iloc[0] == 1].tolist()

        print(f"'{row['movie_title'].values[0]}' has the following genre:")
        print(columns_with_ones)
        print("-" * 50)