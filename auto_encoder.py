import numpy as np
import pandas as pd
from keras.layers import Dense, Input
from keras.models import Model
from sklearn.preprocessing import MinMaxScaler

# Load the dataset
df = pd.read_csv('archive/dataset.csv')

# Select relevant features
features = ['danceability', 'energy', 'acousticness',
            'instrumentalness', 'valence', 'tempo']

# Normalize the data
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

# Define the size of the latent space
latent_dim = 10

# Build the autoencoder model
input_layer = Input(shape=(len(features),))
encoder = Dense(32, activation="relu")(input_layer)
encoder = Dense(16, activation="relu")(encoder)
latent_layer = Dense(latent_dim, activation="relu")(encoder)

decoder = Dense(16, activation="relu")(latent_layer)
decoder = Dense(32, activation="relu")(decoder)
output_layer = Dense(len(features), activation="sigmoid")(decoder)

autoencoder = Model(input_layer, output_layer)
autoencoder.compile(optimizer='adam', loss='mean_squared_error')

# Train the autoencoder
autoencoder.fit(df_scaled, df_scaled, epochs=50, batch_size=32)

# Extract the encoder model
encoder_model = Model(input_layer, latent_layer)

# Get latent representations of songs
latent_representations = encoder_model.predict(df_scaled)

# Function to recommend songs based on latent representations


def recommend_autoencoder(song_index, n_recommendations=5):
    song_rep = latent_representations[song_index].reshape(1, -1)
    similarities = np.dot(latent_representations, song_rep.T).flatten()
    similar_songs_indices = similarities.argsort()[::-1][1:n_recommendations+1]
    return df.iloc[similar_songs_indices][['track_name', 'artists']]


# Example usage
song_index = 10  # Index of the song in the dataset
recommendations = recommend_autoencoder(song_index)
autoencoder.save('autoencoder_model.h5')
encoder_model.save('encoder_model.h5')
print(recommendations)
