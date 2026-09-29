from flask import Flask, render_template, request, send_file, jsonify
from pathlib import Path
import subprocess
import uuid
import threading
import time
import sys

app = Flask(__name__)

UPLOAD_DIR = Path("uploads")
OUTPUT_DIR = Path("outputs")

UPLOAD_DIR.mkdir(exist_ok=True)
OUTPUT_DIR.mkdir(exist_ok=True)

# ---------------------------
# GLOBAL PROGRESS STATUS
# ---------------------------

progress_status = {
    "percent": 0,
    "message": "Idle"
}

# ---------------------------
# HOME PAGE
# ---------------------------

@app.route("/")
def index():
    return render_template("index.html")

# ---------------------------
# PROGRESS API
# ---------------------------

@app.route("/progress")
def progress():
    return jsonify(progress_status)

# ---------------------------
# PROCESS VIDEO THREAD
# ---------------------------

def process_video(input_path, output_path):

    global progress_status

    try:

        progress_status.update({
            "percent": 20,
            "message": "Extracting audio..."
        })

        time.sleep(0.5)

        progress_status.update({
            "percent": 60,
            "message": "Reducing noise..."
        })

        time.sleep(1)

        # IMPORTANT: Use same Python environment
        subprocess.run(
            [
                sys.executable,
                "9_video_noise_suppression.py",
                str(input_path),
                str(output_path)
            ],
            check=True
        )

        progress_status.update({
            "percent": 100,
            "message": "Processing completed"
        })

    except Exception as e:

        progress_status.update({
            "percent": 0,
            "message": f"Error: {str(e)}"
        })


# ---------------------------
# UPLOAD ROUTE
# ---------------------------

@app.route("/upload", methods=["POST"])
def upload():

    global progress_status

    video = request.files.get("video")

    if not video:
        return jsonify({"error": "No file uploaded"}), 400

    uid = str(uuid.uuid4())

    input_path = UPLOAD_DIR / f"{uid}_{video.filename}"
    output_path = OUTPUT_DIR / f"enhanced_{uid}.mp4"

    video.save(input_path)

    progress_status = {
        "percent": 5,
        "message": "Upload complete"
    }

    thread = threading.Thread(
        target=process_video,
        args=(input_path, output_path)
    )

    thread.start()

    return jsonify({
        "status": "processing",
        "output": str(output_path)
    })


# ---------------------------
# DOWNLOAD OUTPUT VIDEO
# ---------------------------

@app.route("/download")
def download():

    files = sorted(
        OUTPUT_DIR.glob("*.mp4"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not files:
        return "No output available", 404

    return send_file(files[0], as_attachment=True)


# ---------------------------
# PREVIEW VIDEO
# ---------------------------

@app.route("/preview")
def preview():

    files = sorted(
        OUTPUT_DIR.glob("*.mp4"),
        key=lambda x: x.stat().st_mtime,
        reverse=True
    )

    if not files:
        return "No preview available", 404

    return send_file(files[0], mimetype="video/mp4")


# ---------------------------
# RUN SERVER
# ---------------------------

if __name__ == "__main__":
    app.run(debug=True)