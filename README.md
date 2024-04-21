# Spotify Song Recommender

## Overview
This project aims to provide users song recommendations similar to a specific song they provide.

![SpotifySongRecommenderDemo](https://github.com/guillaumelalire/Spotify-Song-Recommender/assets/77934673/6afbf05e-5933-4af8-90e4-15151402ae04)

## Features
- **Web Scraping:** Lyrics were not included in the base dataset and are retrieved from [Genius](https://genius.com/).
- **Recommender System:** The system employs the k-Nearest Neighbors (kNN) algorithm to find songs with attributes closest to a given song.
- **Embeddings:** NLP techniques such as embeddings are employed to incorporate song lyrics into the recommender system.

## Dataset
The project utilizes the [30000 Spotify Songs](https://www.kaggle.com/datasets/joebeachcapital/30000-spotify-songs) dataset available on Kaggle. This dataset of 30,000 songs includes song information such as name, artist, and many musical features like popularity, danceability, energy, and acousticness.
