import os
import pandas as pd
import re
from sklearn.neighbors import NearestNeighbors
from lyrics_embeddings import generate_embeddings
from utils import normalize

nb_lyrics_files = 9
spotify_data_path = "data/spotify_songs_cleaned.csv"
embeddings_path = 'data/lyrics_embeddings.csv'

def init_recommender_system():
    music_df = pd.read_csv(spotify_data_path)

    # Check if embeddings were already created
    if not os.path.isfile(embeddings_path):
        # Create a DataFrame containing all the lyrics retrieved from multiple files
        lyrics_df = pd.DataFrame()
        for i in range(0, nb_lyrics_files):
            lyrics_df = pd.concat([lyrics_df, pd.read_csv(f"data/song_lyrics/song_lyrics_{i}.csv")])

        lyrics_df.set_index('Unnamed: 0', inplace=True)
        lyrics_df.sort_index(inplace=True)

        # Replace NaN (instrumental songs) by empty string
        lyrics_df.loc[lyrics_df['lyrics'].isna(), 'lyrics'] = ''

        lyrics = lyrics_df['lyrics'].tolist()
        song_names = music_df['track_name'].tolist()

        # Generate and save the embeddings for all the lyrics
        embeddings = generate_embeddings(lyrics, song_names)
        embeddings = pd.DataFrame(data=embeddings)
        embeddings.index = lyrics_df.index
        embeddings.to_csv(embeddings_path, index=True)

    # Retrieve the saved embeddings
    lyrics_embeddings_df = pd.read_csv(embeddings_path)
    lyrics_embeddings_df.set_index('Unnamed: 0', inplace=True)
    embeddings_size = lyrics_embeddings_df.shape[1]

    # Normalize the embeddings (values between 0 and 1)
    for i in range(embeddings_size):
        lyrics_embeddings_df[str(i)] = normalize(lyrics_embeddings_df[str(i)], min_is_zero=False)

    # Only keep musical features of songs for which lyrics have been collected 
    music_df = music_df.iloc[lyrics_embeddings_df.index]
    
    return music_df, lyrics_embeddings_df

def make_recommendation(music_df, lyrics_df, song_name, artist_name):
    # Extract all the relevant features for the kNN algorithm
    music_features = music_df[['track_popularity', 'track_album_release_date', 'energy', 'key', 'loudness', 'speechiness', 'acousticness', 'instrumentalness', 'liveness', 'valence', 'tempo', 'duration_ms']]
    lyrics_features = lyrics_df[[*[str(x) for x in range(50)]]]
    
    # Initialize and fit the two kNN models
    music_knn = NearestNeighbors(n_neighbors=2, radius=0.4)
    music_knn.fit(music_features)
    
    lyrics_knn = NearestNeighbors(n_neighbors=2, radius=0.4)
    lyrics_knn.fit(lyrics_features)

    # User enters a song (case insensitive) and song index is retrieved in the dataframe
    song_index = music_df[music_df['track_name'].str.lower() == song_name.lower()].index

    if song_index.empty: # Unknown song
        return None, None, None
    elif len(song_index) > 1 and artist_name != '': # Multiple songs found -> Find the right one by artist name
        song_index_good_artist = music_df.loc[song_index][music_df.loc[song_index]['track_artist'].str.lower() == artist_name.lower()].index
        if not song_index_good_artist.empty:
            song_index = song_index_good_artist

    nb_songs = len(music_df)
    
    # Calculate distances between the specified song and all the other songs
    music_distances, music_neighbours = music_knn.kneighbors(music_features.loc[song_index], nb_songs, return_distance=True)
    lyrics_distances, lyrics_neighbours = lyrics_knn.kneighbors(lyrics_features.loc[song_index], nb_songs, return_distance=True)

    # Create a DataFrame for each type of distances and normalize them
    music_recommendations = music_df.iloc[music_neighbours[0]][['track_name', 'track_artist']]
    music_recommendations['distance_music'] = normalize(music_distances[0], min_is_zero=True)

    lyrics_recommendations = pd.DataFrame()
    lyrics_recommendations['distance_lyrics'] = normalize(lyrics_distances[0], min_is_zero=True)
    lyrics_recommendations.index = lyrics_df.iloc[lyrics_neighbours[0]].index

    recommendations = pd.concat([music_recommendations, lyrics_recommendations["distance_lyrics"]], axis=1)
    recommendations['distance_both'] = (recommendations['distance_music'] + recommendations['distance_lyrics']) / 2
    recommendations.sort_values(by='distance_both', inplace=True)

    # Retrieve the exact song and artist names from the dataset
    song_name = music_df.iloc[music_neighbours[0]]['track_name'].tolist()[0]
    artist_name =  music_df.iloc[music_neighbours[0]]['track_artist'].tolist()[0]

    # Convert all distances to floats with 4 decimal places (because values were too long to display)
    recommendations['distance_music'] = recommendations['distance_music'].apply(lambda x: float(f'{x:.4f}'))
    recommendations['distance_lyrics'] = recommendations['distance_lyrics'].apply(lambda x: float(f'{x:.4f}'))
    recommendations['distance_both'] = recommendations['distance_both'].apply(lambda x: float(f'{x:.4f}'))

    return recommendations, song_name, artist_name