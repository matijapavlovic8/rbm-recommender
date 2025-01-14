import torch
import pandas as pd


def recommend(watched, probs, n):
    probs[watched == 1] = 0
    _, indices = torch.topk(probs, n)
    result_tensor = torch.zeros_like(probs)
    result_tensor[indices] = 1
    return result_tensor


def movie_from_tensor(tensor, movies):
    indices = torch.nonzero(tensor == 1).squeeze()
    adjusted_indices = indices + 1  

    movie_ids = movies['movie_id'].iloc[adjusted_indices - 1].values  

    for movie_id in movie_ids:
        row = movies[movies['movie_id'] == movie_id]
        columns_with_ones = row.columns[row.iloc[0] == 1].tolist()

        print(f"'{row['movie_title'].values[0]}' has the following genre:")
        print(columns_with_ones)
        print("-" * 50)