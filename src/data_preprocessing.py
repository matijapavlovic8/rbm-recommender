import pandas as pd

def load_data(file_path):
    """Load the MovieLens dataset."""
    column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
    data = pd.read_csv(file_path, sep='\t', names=column_names)
    return data


def preprocess_data(data):
    """Create user-movie interaction matrix and binarize ratings."""
    interaction_matrix = data.pivot(index='user_id', columns='movie_id', values='rating').fillna(0)
    binary_matrix = (interaction_matrix >= 3).astype(int)
    return binary_matrix


def load_movies(file_path):
    column_names = [ 'movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western' ] 
    movies = pd.read_csv( file_path, sep='|', header=None, names=column_names, encoding='ISO-8859-1' )
    movies = movies.drop(columns = ["release_date", "video_release_date", "IMDb_URL", "unknown"])
    return movies

