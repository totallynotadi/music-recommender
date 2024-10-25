import pandas as pd
from sklearn.cluster import KMeans
from sklearn.preprocessing import MinMaxScaler

# Load your dataset
df = pd.read_csv('archive/dataset.csv')
print('::read csv')

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
``

def recommend_from_cluster(track_name, n_recommendations=5):
    # Get the cluster of the input song
    song_cluster = df[df['track_name'] == track_name]['cluster'].values[0]

    # Get songs from the same cluster
    cluster_songs = df[df['cluster'] == song_cluster]

    # Recommend top N songs from the same cluster (excluding the song itself)
    recommendations = cluster_songs[cluster_songs['track_name'] != track_name].sample(
        n=n_recommendations)

    return recommendations[['track_name', 'artists']]


# Example usage
recommendations = recommend_from_cluster(
    'Gimme! Gimme! Gimme!', n_recommendations=5)
print(recommendations)
