import os
import sys
import signal
from flask import Flask, render_template, send_from_directory

app = Flask(__name__)

# Force clean exit on SIGINT (Ctrl+C) and SIGTERM to prevent Windows ghost processes
def clean_exit(signum, frame):
    sys.exit(0)

signal.signal(signal.SIGINT, clean_exit)
signal.signal(signal.SIGTERM, clean_exit)

@app.route('/')
def index():
    return render_template('index.html')

@app.route('/music')
def serve_music():
    # Serve the local mp3 file directly from the app folder
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(current_dir, 'Foto kita blur - Sal Priadi _ Lirik Lagu.mp3')

@app.route('/kicau')
def serve_kicau():
    # Serve the Kicau Mania mp3 file
    current_dir = os.path.dirname(os.path.abspath(__file__))
    return send_from_directory(current_dir, 'KICAU MANIA  NDARBOY GENK, BANDITOZ YAOW 86 (Lyrics).mp3')

if __name__ == '__main__':
    # use_reloader=False prevents Flask from spawning a background watcher process on Windows
    app.run(host='127.0.0.1', port=5000, debug=True, use_reloader=False)
