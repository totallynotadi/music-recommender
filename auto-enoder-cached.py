import numpy as np
import pandas as pd
from keras.layers import Dense, Input
from keras.models import Model, load_model
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('archive/dataset.csv')

# Select relevant features
features = ['danceability', 'energy', 'acousticness',
            'instrumentalness', 'valence', 'tempo']

# Normalize the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

encoder_model = load_model('encoder_model.h5')

# Get latent representations of songs
latent_representations = encoder_model.predict(df_scaled)


def recommend_autoencoder(song_index, n_recommendations=5):
    song_rep = latent_representations[song_index].reshape(1, -1)
    similarities = np.dot(latent_representations, song_rep.T).flatten()
    similar_songs_indices = similarities.argsort()[::-1][1:n_recommendations+1]
    return df.iloc[similar_songs_indices][['track_name', 'artists', 'track_id']]


# Example usage
song_index = 10  # Index of the song in the dataset
recommendations = recommend_autoencoder(song_index)
print(recommendations)
