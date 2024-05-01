import os
import pandas as pd
import re
from sklearn.neighbors import NearestNeighbors
from lyrics_embeddings import generate_embeddings
from utils import normalize

def init_recommender_system():
    df = pd.read_csv("../data/spotify_songs_cleaned.csv")
    
    nb_lyrics_files = 9
    lyrics_df = pd.DataFrame()

    for i in range(0, nb_lyrics_files):
        lyrics_df = pd.concat([lyrics_df, pd.read_csv(f"../data/song_lyrics/song_lyrics_{i}.csv")])

    lyrics_df.set_index('Unnamed: 0', inplace=True)

    # Add lyrics column to main dataframe
    df = pd.concat([df, lyrics_df['lyrics']], axis=1, join="inner")

    # Replace NaN (instrumental songs) by empty string
    df.loc[df['lyrics'].isna(), 'lyrics'] = ''

    lyrics = df['lyrics'].tolist()
    song_names = df['track_name'].tolist()

    embeddings_path = '../data/lyrics_embeddings.csv'
    
    if not os.path.isfile(embeddings_path):
        embeddings = generate_embeddings(lyrics, song_names)
        embeddings = pd.DataFrame(data=embeddings)
        embeddings.to_csv(embeddings_path, index=False)

    lyrics_embeddings = pd.read_csv(embeddings_path)
    embeddings_size = lyrics_embeddings.shape[1]

    # Add embeddings to main dataframe
    df.reset_index(inplace=True)
    df = pd.concat([df, lyrics_embeddings], axis=1)

    # Normalize the embeddings (values between 0 and 1)
    for i in range(embeddings_size):
        df[str(i)] = normalize(df[str(i)], min_is_zero=False)

    return df

def make_recommendation(df, song_name, artist_name):
    # Extract all the relevant features for the kNN algorithm
    features = df[['track_popularity', 'track_album_release_date', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms', *[str(x) for x in range(50)]]]
    
    # Initialize and fit the kNN model
    knn = NearestNeighbors(n_neighbors=2, radius=0.4)
    knn.fit(features)

    # User enters a song (case insensitive) and song index is retrieved in the dataframe
    #song_name = input("Enter the name of a song:")
    song_index = df[df['track_name'].str.lower() == song_name.lower()].index

    if song_index.empty:
        return None, None, None
    elif len(song_index) > 1 and artist_name != '':
        song_index_good_artist = df.iloc[song_index][df.iloc[song_index]['track_artist'].str.lower() == artist_name.lower()].index
        if not song_index_good_artist.empty:
            song_index = song_index_good_artist

    # Find 10 nearest neighbours for the specified song
    neighbours = knn.kneighbors(features.iloc[song_index], 16, return_distance=False)
    recommendation = df.iloc[neighbours[0]][['track_name', 'track_artist']][1:]

    song_name = df.iloc[neighbours[0]]['track_name'].tolist()[0]
    artist_name =  df.iloc[neighbours[0]]['track_artist'].tolist()[0]

    return recommendation.values.tolist(), song_name, artist_name