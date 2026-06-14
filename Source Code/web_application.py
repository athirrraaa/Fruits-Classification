from flask import Flask
from flask import render_template
from flask import request

import os

from joblib import load

from feature_extraction import (
    extract_features
)

app = Flask(__name__)

UPLOAD_FOLDER = "uploads"

app.config[
    "UPLOAD_FOLDER"
] = UPLOAD_FOLDER

# ==========================================
# LOAD MODEL
# ==========================================

model = load(
    "../models/fruit_svm.joblib"
)

# ==========================================
# HOME PAGE
# ==========================================

@app.route("/")

def home():

    return render_template(
        "index.html"
    )

# ==========================================
# PREDICTION
# ==========================================

@app.route(
    "/predict",
    methods=["POST"]
)

def predict():

    if "image" not in request.files:

        return "No file uploaded"

    file = request.files["image"]

    if file.filename == "":

        return "No selected file"

    os.makedirs(
        app.config["UPLOAD_FOLDER"],
        exist_ok=True
    )  
    
    filepath = os.path.join(
        app.config[
            "UPLOAD_FOLDER"
        ],
        file.filename
    )

    file.save(filepath)

    features = extract_features(
        filepath
    )

    prediction = model.predict(
        [features]
    )

    return render_template(
        "index.html",
        prediction=prediction[0]
    )

# ==========================================
# MAIN
# ==========================================

if __name__ == "__main__":

    app.run(
        debug=True
    )