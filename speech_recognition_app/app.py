from flask import Flask, render_template, request
import os
import numpy as np
from tensorflow.keras.models import load_model
from sklearn.preprocessing import LabelEncoder
from utils import extract_features

app = Flask(__name__)
UPLOAD_FOLDER = "uploads"
os.makedirs(UPLOAD_FOLDER, exist_ok=True)

# Loading model and label encoder too
model = load_model("ser_model.h5")
le = LabelEncoder()
le.classes_ = np.load("classes.npy", allow_pickle=True)

@app.route("/", methods=["GET", "POST"])
def index():
    emotion = None
    confidence_scores = None  # for GET requests

    if request.method == "POST":
        file = request.files["audio"]
        if file and file.filename.endswith(".wav"):
            file_path = os.path.join(UPLOAD_FOLDER, file.filename)
            file.save(file_path)

            features = extract_features(file_path)
            features = np.expand_dims(features, axis=0)
            prediction = model.predict(features)[0]
            predicted_label = np.argmax(prediction)
            emotion = le.inverse_transform([predicted_label])[0]

            label_map = {
            "ps": "(PS) Pleasant Surprise",
            "happy": "Happy",
            "angry": "Angry",
            "fear" : "Fear",
            "neutral" : "Neutral",
            "disgust" : "Disgust",
            "sad" : "Sad"
            }

            confidence_scores = {
            label_map.get(le.classes_[i], le.classes_[i]): round(float(prediction[i]) * 100, 2)
            for i in range(len(le.classes_))
            }

    return render_template("index.html", emotion=emotion, scores=confidence_scores)



if __name__ == "__main__":
    app.run(debug=True)
