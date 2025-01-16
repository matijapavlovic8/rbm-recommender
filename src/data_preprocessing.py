import pandas as pd

def load_data(file_path):
    """Load the MovieLens dataset."""
    column_names = ['user_id', 'movie_id', 'rating', 'timestamp']
    data = pd.read_csv(file_path, sep='\t', names=column_names)
    return data


def preprocess_data(data):
    """Create user-movie interaction matrix with dual bit encoding."""
    interaction_matrix = data.pivot(index='user_id', columns='movie_id', values='rating')

    def encode_rating(rating):
        if pd.isna(rating):  # unrated (NaN)
            return [0, 0]
        elif rating < 3:  # disliked
            return [1, 0]
        else:  # liked
            return [1, 1]

    interaction_matrix = interaction_matrix.map(encode_rating)

    rated_column = interaction_matrix.map(lambda x: x[0])  # First bit: rated
    liked_column = interaction_matrix.map(lambda x: x[1])  # Second bit: liked

    interaction_matrix = pd.concat([rated_column, liked_column], axis=1, keys=['rated', 'liked'])

    return interaction_matrix


def load_movies(file_path):
    "Load movie details into dataframe."
    column_names = [ 'movie_id', 'movie_title', 'release_date', 'video_release_date', 'IMDb_URL', 'unknown', 'Action', 'Adventure', 'Animation', "Children's", 'Comedy', 'Crime', 'Documentary', 'Drama', 'Fantasy', 'Film-Noir', 'Horror', 'Musical', 'Mystery', 'Romance', 'Sci-Fi', 'Thriller', 'War', 'Western' ] 
    movies = pd.read_csv( file_path, sep='|', header=None, names=column_names, encoding='ISO-8859-1' )
    movies = movies.drop(columns = ["release_date", "video_release_date", "IMDb_URL", "unknown"])
    return movies