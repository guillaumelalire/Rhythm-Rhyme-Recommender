import aiohttp
import asyncio
import os
import pandas as pd
import re
from bs4 import BeautifulSoup
from dotenv import load_dotenv

# Load access token for Genius API from environment (.env file)
load_dotenv()
access_token = os.getenv("GENIUS_ACCESS_TOKEN")

# Load Spotify songs dataset
df = pd.read_csv("../data/spotify_songs_cleaned.csv")

batch_size = 3000

# Fetch search results from Genius API
async def fetch_genius_search_results(params):
    async with aiohttp.ClientSession() as session:
        url = 'https://api.genius.com/search'
        headers = {'Authorization': f'Bearer {access_token}'}
        async with session.get(url, params=params, headers=headers) as response:
            return await response.json()

# Standard scraping function
async def scrape(url):
    async with aiohttp.ClientSession() as session:
        async with session.get(url) as response:
            return await response.text()

# Scrape lyrics for specified song and add them to lyrics dataframe
async def scrape_lyrics(session, song, index, lyrics_df):
    song_name = song[0]
    artist_name = song[1]
    
    song_name = re.sub(r" (-|\(|\[).*", "", song_name) # Remove feat, with, remix
    
    params = {'q': f"{artist_name} {song_name}"} # Genius search query
    
    response = await fetch_genius_search_results(params)

    url = None

    # Iterate of search results
    for results in response['response']['hits']:
        # Lyrics URL is first result found with corresponding artist
        if artist_name.lower() in results['result']['artist_names'].lower():
            url = results['result']['url']
            break
        # Else URL is first result with corresponding song name
        if song_name.lower() in results['result']['full_title'].lower() and url == None:
            url = results['result']['url']

    if url != None: # Lyrics URL was found
        response = await scrape(url)
        html = BeautifulSoup(response, 'html.parser')
        lyrics_html = html.findAll("div", {"data-lyrics-container": True}) # Lyrics are separated in different HTML blocks

        lyrics = ""
        for block in lyrics_html:
           # Replace line breaks by new lines to avoid loosing line separators
           for line_break in block.findAll('br'): 
                line_break.replaceWith('\n')
           # Merge lyrics blocks
           lyrics += block.get_text() + '\n'

        lyrics_df.loc[index] = [song_name, artist_name, lyrics]
        
        return lyrics

    return None # Lyrics URL was not found

# Process one batch of songs and save their lyrics to a CSV
async def process_batch(df, start, end):
    async with aiohttp.ClientSession() as session:
        lyrics_df = pd.DataFrame(columns=['track_name', 'track_artist', 'lyrics'])
        
        tasks = [asyncio.create_task(scrape_lyrics(session, song, start + i, lyrics_df)) for i, song in enumerate(df.values[start:end])]
        responses = await asyncio.gather(*tasks)
        
        batch_id = int(start / batch_size) # Batch ID to identify the correct file for saving
        filepath = f'../data/song_lyrics/song_lyrics_{batch_id}.csv'
        
        lyrics_df.to_csv(filepath, index=True)

nb_songs = len(df)

for start in range(0, nb_songs, batch_size):
    loop = asyncio.get_event_loop()
    responses = loop.run_until_complete(process_batch(df, start, start + batch_size))
