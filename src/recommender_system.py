import pandas as pd
import re
from sklearn.neighbors import NearestNeighbors
from lyrics_embeddings import generate_embeddings
from utils import normalize

nb_files = 9

df = pd.read_csv("../data/spotify_songs_cleaned.csv")

lyrics_df = pd.DataFrame()

for i in range(0, nb_files):
    lyrics_df = lyrics_df.append(pd.read_csv(f"../data/song_lyrics/song_lyrics_{i}.csv"))

lyrics_df.set_index('Unnamed: 0', inplace=True)

# Add lyrics column to main dataframe
df = pd.concat([df, lyrics_df['lyrics']], axis=1, join="inner")

df = df[:1000]

# Replace NaN (instrumental songs) by empty string
df.loc[df['lyrics'].isna(), 'lyrics'] = ''

lyrics = df['lyrics'].tolist()
song_names = df['track_name'].tolist()

lyrics_embeddings = generate_embeddings(lyrics, song_names)

embeddings_size = len(lyrics_embeddings[0])

# Add embeddings to main dataframe
df.reset_index(inplace=True)
df = pd.concat([df, pd.DataFrame(data=lyrics_embeddings)], axis=1)

# Normalize the embeddings (values between 0 and 1)
for i in range(embeddings_size):
    df[i] = normalize(df[i], min_is_zero=False)

# Extract all the relevant features for the kNN algorithm
features = df[['track_popularity', 'track_album_release_date', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', *range(0, embeddings_size)]]

# Initialize and fit the kNN model
knn = NearestNeighbors(n_neighbors=2, radius=0.4)
knn.fit(features)

# User enters a song (case insensitive) and song index is retrieved in the dataframe
song_name = input("Enter the name of a song:")
song_index = df[df['track_name'].str.lower() == song_name.lower()].index

# Find 10 nearest neighbours for the specified song
neighbours = knn.kneighbors(features.iloc[song_index], 11, return_distance=True)
print(df.iloc[neighbours[1][0]][['track_name', 'track_artist']].assign(distance=neighbours[0][0]))