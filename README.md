# 🎤 Speech Emotion Recognition Web App

This is a deep learning–based web application that detects **emotions from human speech**. Upload a `.wav` audio file, and the model will predict the emotion being expressed — along with confidence scores for each class.

> Built using TensorFlow, Flask, and Librosa — deployed locally and ready for web.

---

## 🧠 Emotions Detected

- Happy 😊  
- Sad 😢  
- Angry 😠  
- Disgust 😖  
- Fear 😨  
- Surprise 😮 *(labeled as `ps` = Pleasant Surprise)*  
- Neutral 😐  

---

### 🚀 Features

✅ Upload `.wav` files  
✅ Displays top predicted emotion  
✅ Shows confidence scores for all emotions  
✅ Tooltip explanations (e.g. PS = Pleasant Surprise)  
✅ Clean, simple HTML frontend  
✅ Powered by MFCC audio features and a custom-trained deep learning model

---

## 🛠️ Tech Stack

| Layer         | Tools Used                        |
|---------------|------------------------------------|
| Frontend      | HTML, Jinja2 (Flask Templates)     |
| Backend       | Flask (Python)                     |
| Deep Learning | TensorFlow / Keras                 |
| Audio         | Librosa                            |
| Data          | [TESS Dataset (Kaggle)](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess) |

---

## 📁 Folder Structure

```

speech\_emotion\_app/
├── app.py                # Flask server
├── utils.py              # Audio feature extraction
├── ser\_model.h5          # Trained Keras model
├── classes.npy           # Encoded emotion labels
├── templates/
│   └── index.html        # Upload form UI
|  
├── uploads/              # Uploaded audio files
└── README.md

````

---

## 📦 How to Run Locally

> You’ll need Python 3.10+ and pip installed.

### 1. Clone the repo
```bash
git clone https://github.com/your-username/speech-emotion-recognition-app.git
cd speech_emotion_app
````

### 2. Create and activate a virtual environment

```bash
python -m venv tf-env
.\tf-env\Scripts\activate  # On Windows
# source tf-env/bin/activate  # On Mac/Linux
```

### 3. Install dependencies

```bash
pip install -r requirements.txt
```

> ⚠️ **Important:** If you see errors with NumPy, install version <2.0:

```bash
pip install "numpy<2.0"
```

### 4. Run the app

```bash
python app.py
```

Then open your browser at:
👉 `http://127.0.0.1:5000`

---

## 📊 Model Training (Optional)

The model was trained on the [TESS dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess), using MFCC features and a deep neural network built with Keras.

You can retrain your own model using the `extract_features()` method in `utils.py`.

---

## 📜 License

This project is open-source and available under the MIT License.

---

## ❤️ Acknowledgements

* [Librosa](https://librosa.org/) for powerful audio processing
* [Kaggle TESS Dataset](https://www.kaggle.com/datasets/ejlok1/toronto-emotional-speech-set-tess)
* YouTube + TensorFlow docs + StackOverflow for support during bugs :)

---

## 🙋‍♂️ Author

Made by Sohan Raut (SRx210)

> Building AI tools that understand how we sound — not just what we say.

---
