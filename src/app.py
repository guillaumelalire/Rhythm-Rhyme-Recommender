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

    df = init_recommender_system()
    recommendation, song_name, artist_name = make_recommendation(df, song_name, artist_name)

    if not recommendation:
        return render_template('index.html', not_found=True)

    return render_template('results.html', song_name=song_name, artist_name=artist_name, recommendation=recommendation)

if __name__ == '__main__':
    app.secret_key = ".."
    app.run(debug=True)