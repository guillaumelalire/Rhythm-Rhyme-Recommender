import pandas as pd

# Load Spotify songs dataset
df = pd.read_csv("../data/spotify_songs.csv")

# Remove duplicate songs and reindex the dataframe
df.drop_duplicates(subset=['track_name', 'track_artist'], inplace=True)
df.reset_index(inplace=True, drop=True)

# Remove useless columns
df.drop(['track_id', 'track_album_id', 'playlist_name', 'track_album_name', 'playlist_id', 'playlist_genre', 'playlist_subgenre'], axis=1, inplace=True)

# Normalize the array by scaling values between 0 and 1
def normalize(arr, min_is_zero=True):
    if min_is_zero:
        return arr / max(arr)
    else:
        return (arr - min(arr)) / (max(arr) - min(arr))

df['duration_ms'] = normalize(df['duration_ms'])
df['tempo'] = normalize(df['tempo'])
df['track_popularity'] = normalize(df['track_popularity'])
df['key'] = normalize(df['key'])

df['loudness'] = normalize(df['loudness'], min_is_zero=False)

# Convert release date string to an int, then normalize the date (0 for oldest song, 1 for newest song)
df['track_album_release_date'] = pd.to_datetime(df['track_album_release_date']).astype(int)
df['track_album_release_date'] = normalize(df['track_album_release_date'], min_is_zero=False)

df.to_csv('../data/spotify_songs_cleaned.csv', index=False)  