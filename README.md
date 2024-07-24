# Rhythm & Rhyme Recommender

## Overview

This project is designed to offer personalized song recommendations based on a specific song using various musical features and lyrics.

By analyzing both the audio characteristics and the thematic elements of a track, the system helps discovering new music which closely aligns with the user's preferences.

https://github.com/user-attachments/assets/cf719158-8b45-4abe-b17f-c8f8bbd90533

## Running the Project

Follow these simple steps to get the project up and running on your local machine:
### 1. **Install the Requirements**

First, ensure you have all the necessary dependencies installed. You can do this by running the following command in the main folder:
```
pip install -r requirements.txt
```

### 2. **Start the Project**

Once the dependencies are installed, you can start the project by executing the main application file. Run the following command:
```
python3 src/app.py
```

### 3. **Access the Application**

After the project has started, open your web browser and navigate to the following URL to access the application: [http://127.0.0.1:5000](http://127.0.0.1:5000)

## Workflow

The project is based on training two K-nearest neighbours (KNN) models, each considering different features, to compare the distance between songs:
- The first model uses various features about songs provided by Spotify.
- The second model uses song lyrics converted into features through embeddings.

![song-recommender-workflow](https://github.com/user-attachments/assets/04bd1301-f2fd-4e04-bb6b-b141d14f2421)

### 1. Musical Features

- **Dataset** : The dataset used is the [30000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs) dataset, available on Kaggle, which includes information from Spotify on 30,000 songs.

- **Preprocessing**: 12 features representing the song characteristics that can be expressed in numerical format are selected and formatted appropriately: popularity, release date, energy, key, loudness, speechiness, acousticness, instrumentalness, liveness, valence, tempo, duration.

- **K-nearest neighbours**: Songs are compared based on their Spotify features to find the closest songs in terms of musical attributes. This generates a list of distances, indicating how similar each song is to a target song based on its musical features.

### 2. Lyrical Features

- **Scraping**: Lyrics are retrieved from Genius.

- **Preprocessing**: Lyrics are cleaned to ensure consistency and relevance: verse and chorus indicators are removed, instrumental songs are handled, etc.

- **Embeddings**: Lyrics are converted into numerical vectors (embeddings) using Doc2Vec, a technique that captures semantic meaning.

- **K-nearest neighbours**: Songs are compared based on their lyrical content embeddings to find the closest songs in terms of lyrical meaning. This generates a list of distances, indicating how similar each song is to a target song based on its lyrical content.

### 3. Recommendation

- **Normalization**: Distances are normalized to be between 0 and 1, giving equal weight to the two differently calculated distances.

- **Combining Distances**: The average of the two obtained distances is taken to get the final distance, indicating how similar the songs are to the target song considering both musical and lyrical features. Songs with lower combined average distance scores are more similar and therefore better recommendations.
