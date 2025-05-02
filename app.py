# app.py

import os, uuid
from flask import Flask, render_template, request, send_from_directory
from processing import run_model_on_video

app = Flask(__name__)
UPLOAD = "uploads"
OUTPUT = "processed"
os.makedirs(UPLOAD, exist_ok=True)
os.makedirs(OUTPUT, exist_ok=True)

@app.route("/", methods=["GET","POST"])
def index():
    if request.method == "POST":
        model = request.form["model"]
        vid    = request.files["video"]
        name   = f"{uuid.uuid4()}.mp4"
        in_path = os.path.join(UPLOAD, name)
        out_path= os.path.join(OUTPUT, "out_"+name)
        vid.save(in_path)

        metrics = run_model_on_video(in_path, out_path, model)
        return render_template("result.html",
                               video_file=os.path.basename(out_path),
                               metrics=metrics)
    return render_template("index.html")

@app.route("/processed/<path:filename>")
def processed_video(filename):
    return send_from_directory(OUTPUT, filename, as_attachment=False)

if __name__ == "__main__":
    app.debug = True
    app.run(host="0.0.0.0", port=5000)
