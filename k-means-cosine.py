import pandas as pd
from sklearn.cluster import KMeans
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
df = pd.read_csv('archive/dataset.csv')

# Select relevant features
features = ['danceability', 'energy', 'key', 'loudness', 'speechiness',
            'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo']

# Normalize the features
scaler = MinMaxScaler()
df_scaled = scaler.fit_transform(df[features])

# Perform K-means clustering
# Adjust the number of clusters (K)
kmeans = KMeans(n_clusters=10, random_state=42)
df['cluster'] = kmeans.fit_predict(df_scaled)

# Function to recommend songs from the same cluster


# Function to recommend songs using both K-means and cosine similarity
def recommend_hybrid(track_name, n_recommendations=5):
    # Get the cluster of the input song
    song_cluster = df[df['track_name'] == track_name]['cluster'].values[0]

    # Get songs from the same cluster
    cluster_songs = df[df['cluster'] == song_cluster]

    # Extract features of the cluster songs
    cluster_features = df_scaled[cluster_songs.index]

    # Get the input song's index and features
    song_index = df[df['track_name'] == track_name].index[0]
    song_features = df_scaled[song_index].reshape(1, -1)

    # Compute cosine similarity within the cluster
    similarity_scores = cosine_similarity(song_features, cluster_features)[0]

    # Get top N most similar songs from the same cluster
    similar_songs_indices = similarity_scores.argsort()[
        ::-1][1:n_recommendations + 1]

    # Return recommended songs
    return cluster_songs.iloc[similar_songs_indices][['track_name', 'artists']]


# Example usage
recommendations = recommend_hybrid("I'm an Albatraoz", n_recommendations=5)
print(recommendations)
