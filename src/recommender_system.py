import os
import pandas as pd
import re
from sklearn.neighbors import NearestNeighbors
from lyrics_embeddings import generate_embeddings
from utils import normalize

def init_recommender_system():
    music_df = pd.read_csv("../data/spotify_songs_cleaned.csv")
    
    nb_lyrics_files = 9
    lyrics_df = pd.DataFrame()

    for i in range(0, nb_lyrics_files):
        lyrics_df = pd.concat([lyrics_df, pd.read_csv(f"../data/song_lyrics/song_lyrics_{i}.csv")])

    lyrics_df.set_index('Unnamed: 0', inplace=True)
    lyrics_df.sort_index(inplace=True)

    # Only keep songs for which lyrics have been collected
    music_df = music_df.iloc[lyrics_df.index]

    # Replace NaN (instrumental songs) by empty string
    lyrics_df.loc[lyrics_df['lyrics'].isna(), 'lyrics'] = ''

    lyrics = lyrics_df['lyrics'].tolist()
    song_names = music_df['track_name'].tolist()

    embeddings_path = '../data/lyrics_embeddings.csv'
    
    if not os.path.isfile(embeddings_path):
        embeddings = generate_embeddings(lyrics, song_names)
        embeddings = pd.DataFrame(data=embeddings)
        embeddings.index = lyrics_df.index
        embeddings.to_csv(embeddings_path, index=True)

    lyrics_embeddings = pd.read_csv(embeddings_path)
    lyrics_embeddings.set_index('Unnamed: 0', inplace=True)
    embeddings_size = lyrics_embeddings.shape[1]

    # Normalize the embeddings (values between 0 and 1)
    for i in range(embeddings_size):
        lyrics_df[str(i)] = normalize(lyrics_embeddings[str(i)], min_is_zero=False) / 2

    return music_df, lyrics_df

def make_recommendation(music_df, lyrics_df, song_name, artist_name):
    # Extract all the relevant features for the kNN algorithm
    music_features = music_df[['track_popularity', 'track_album_release_date', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]
    lyrics_features = lyrics_df[[*[str(x) for x in range(50)]]]
    
    # Initialize and fit the kNN models
    music_knn = NearestNeighbors(n_neighbors=2, radius=0.4)
    music_knn.fit(music_features)
    
    lyrics_knn = NearestNeighbors(n_neighbors=2, radius=0.4)
    lyrics_knn.fit(lyrics_features)

    # User enters a song (case insensitive) and song index is retrieved in the dataframe
    song_index = music_df[music_df['track_name'].str.lower() == song_name.lower()].index

    if song_index.empty: # Unknown song
        return None, None, None
    elif len(song_index) > 1 and artist_name != '': # Multiple songs found -> Find the right one by artist name
        song_index_good_artist = music_df.iloc[song_index][music_df.iloc[song_index]['track_artist'].str.lower() == artist_name.lower()].index
        if not song_index_good_artist.empty:
            song_index = song_index_good_artist

    print(music_features.loc[song_index])
    nb_songs = len(music_df)
    
    # Find 10 nearest neighbours for the specified song
    music_distances, music_neighbours = music_knn.kneighbors(music_features.loc[song_index], nb_songs, return_distance=True)
    lyrics_distances, lyrics_neighbours = lyrics_knn.kneighbors(lyrics_features.loc[song_index], nb_songs, return_distance=True)

    music_recommendations = music_df.iloc[music_neighbours[0]][['track_name', 'track_artist']]
    music_recommendations['distance_music'] = normalize(music_distances[0], min_is_zero=True)
    
    lyrics_recommendations = lyrics_df.iloc[lyrics_neighbours[0]][['track_name', 'track_artist']]
    lyrics_recommendations['distance_lyrics'] = normalize(lyrics_distances[0], min_is_zero=True)

    recommendations = pd.concat([music_recommendations, lyrics_recommendations["distance_lyrics"]], axis=1)
    recommendations['distance_both'] = (recommendations['distance_music'] + recommendations['distance_lyrics']) / 2
    recommendations.sort_values(by='distance_both', inplace=True)
    
    song_name = music_df.iloc[music_neighbours[0]]['track_name'].tolist()[0]
    artist_name =  music_df.iloc[music_neighbours[0]]['track_artist'].tolist()[0]

    return recommendations, song_name, artist_name