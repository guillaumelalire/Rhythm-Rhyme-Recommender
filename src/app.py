import os
from flask import Flask, render_template, request, redirect, url_for, session
from recommender_system import init_recommender_system, make_recommendation

app = Flask(__name__)

result_generated = False

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

    if recommendation is None:
        return render_template('index.html', not_found=True)

    distances = request.args.get('distances', 'both')

    if distances == 'music':
        distance_id = 2
        sorted_recommendation = recommendation.sort_values(by='distance_music').values.tolist()[1:15]
    elif distances == 'lyrics':
        distance_id = 3
        sorted_recommendation = recommendation.sort_values(by='distance_lyrics').values.tolist()[1:15]
    else:
        distance_id = 4
        sorted_recommendation = recommendation.values.tolist()[1:15]
    
    return render_template('results.html', song_name=song_name, artist_name=artist_name, recommendation=sorted_recommendation, distances=distances, distance_id=distance_id)

if __name__ == '__main__':
    app.secret_key = ".."
    app.run(debug=True)