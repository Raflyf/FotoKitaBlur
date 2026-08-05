import os

from flask import Flask, jsonify, render_template, send_from_directory

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
MEDIA_DIR = os.path.join(BASE_DIR, "media", "music")

HOST = os.environ.get("FOTO_BLUR_HOST", "127.0.0.1")
PORT = int(os.environ.get("FOTO_BLUR_PORT", "5000"))
DEBUG = os.environ.get("FOTO_BLUR_DEBUG", "0") == "1"

# Filenames are resolved against MEDIA_DIR only; the routes below are fixed
# identifiers so arbitrary paths can never reach the server-side filesystem.
AUDIO_FILES = {
    "music": "foto-kita-blur.mp3",
    "kicau": "kicau-mania.mp3",
}

app = Flask(__name__)


@app.after_request
def set_security_headers(resp):
    resp.headers.setdefault("X-Content-Type-Options", "nosniff")
    resp.headers.setdefault("X-Frame-Options", "DENY")
    resp.headers.setdefault("Referrer-Policy", "no-referrer")
    # Strict-but-workable CSP: MediaPipe Tasks Web is a WASM binary and needs
    # 'wasm-unsafe-eval' to compile/instantiate; scripts/fonts come from
    # jsdelivr and Google Fonts; all other resources are same-origin.
    resp.headers.setdefault(
        "Content-Security-Policy",
        "default-src 'self'; "
        "script-src 'self' 'wasm-unsafe-eval' https://cdn.jsdelivr.net; "
        "style-src 'self' 'unsafe-inline' https://fonts.googleapis.com; "
        "font-src https://fonts.gstatic.com; "
        "img-src 'self' data:; "
        "media-src 'self'; "
        "connect-src 'self' https://cdn.jsdelivr.net; "
        "worker-src 'self' blob:",
    )
    return resp


@app.route("/")
def index():
    return render_template("index.html")


def _serve_audio(key):
    filename = AUDIO_FILES.get(key)
    if filename is None:
        return jsonify({"error": "Unknown audio"}), 404
    return send_from_directory(MEDIA_DIR, filename, conditional=True, max_age=86400)


@app.route("/music")
def serve_music():
    return _serve_audio("music")


@app.route("/kicau")
def serve_kicau():
    return _serve_audio("kicau")


@app.errorhandler(404)
def not_found(_err):
    return jsonify({"error": "Not found"}), 404


@app.errorhandler(500)
def server_error(err):
    app.logger.exception("Unhandled error: %s", err)
    return jsonify({"error": "Internal server error"}), 500


if __name__ == "__main__":
    app.run(host=HOST, port=PORT, debug=DEBUG, use_reloader=False)