import os
from flask import Flask, render_template, request, redirect, url_for, session
from recommender_system import init_recommender_system, make_recommendation

app = Flask(__name__)

@app.route('/', methods=['GET', 'POST'])
def index():
    if request.method == 'POST':
        session['song_name'] = request.form['song_name']
        session['artist_name'] = request.form['artist_name']
        return redirect(url_for('loading'))
    else:
        return render_template('index.html', not_found=False)

@app.route('/loading', methods=['GET', 'POST'])
def loading():    
    return render_template('loading.html')

@app.route('/results', methods=['GET', 'POST'])
def results():
    song_name = session['song_name']
    artist_name = session['artist_name']

    music_df, lyrics_df = init_recommender_system()
    recommendation, song_name, artist_name = make_recommendation(music_df, lyrics_df, song_name, artist_name)

    if recommendation is None: # Song not in the dataset
        return render_template('index.html', not_found=True)

    # Determine which features (musical, lyrics, or both) to use for calculating distances (default is both)
    distances = request.args.get('distances', 'both')

    # Keep only the relevant distances column and sort it, then get the top 15 results
    if distances == 'music':
        sorted_recommendation = (recommendation.drop(['distance_lyrics', 'distance_both'], axis=1)
                                              .sort_values(by='distance_music')
                                              .values
                                              .tolist()[1:15])
    elif distances == 'lyrics':
        sorted_recommendation = (recommendation.drop(['distance_music', 'distance_both'], axis=1)
                                              .sort_values(by='distance_lyrics')
                                              .values
                                              .tolist()[1:15])
    else:
        sorted_recommendation = (recommendation.drop(['distance_music', 'distance_lyrics'], axis=1)
                                              .sort_values(by='distance_both')
                                              .values
                                              .tolist()[1:15])
    
    return render_template('results.html', song_name=song_name, artist_name=artist_name, recommendation=sorted_recommendation, distances=distances)

if __name__ == '__main__':
    app.secret_key = ".."
    app.run(debug=True)