import pandas as pd
from sklearn.compose import ColumnTransformer
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.pipeline import make_pipeline
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
df = pd.read_csv('archive/subset.csv')
print('::read csv')

# Select relevant features
features = ['danceability', 'energy', 'key', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Normalize the selected features
scaler = MinMaxScaler()
# df_preprocessed = pd.DataFrame(scaler.fit_transform(df[features]), columns=features)

# TF-IDF vectorizer for text features (artists, genre)
tfidf_vectorizer = TfidfVectorizer()

# Combine numeric and text features using ColumnTransformer
preprocessor = ColumnTransformer(
    transformers=[
        ('num', scaler, features),  # Numeric features
        ('artist', tfidf_vectorizer, 'artists'),  # Artists feature
        ('genre', tfidf_vectorizer, 'track_genre')  # Genre feature
    ])

# Preprocess the entire dataset
df_preprocessed = preprocessor.fit_transform(df)

# print(df, df_preprocessed)
# Compute similarity with combined features
similarity_matrix = cosine_similarity(df_preprocessed)

# Compute cosine similarity between songs
# similarity_matrix = cosine_similarity(df_scaled)

# Function to recommend similar songs


def recommend_songs(song_index, n_recommendations=5):
    # Get similarity scores for the song at the given index
    similarity_scores = list(enumerate(similarity_matrix[song_index]))

    # Sort songs by similarity score
    similarity_scores = sorted(
        similarity_scores, key=lambda x: x[1], reverse=True)

    # Get top n recommendations (excluding the song itself)
    recommendations = similarity_scores[1:n_recommendations+1]

    # Return song indices and similarity scores
    return [(df.iloc[i]['track_name'], df.iloc[i]['artists'], df.iloc[i]['track_id'], score) for i, score in recommendations]


# Example: Recommend songs similar to the song at index 0
print(recommend_songs(850))
