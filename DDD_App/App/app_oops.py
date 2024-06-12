from flask import Flask, render_template, request
from gtts import gTTS
from werkzeug.utils import secure_filename
from PIL import Image
import numpy as np
import pickle
import os
from skimage.transform import resize
from skimage.io import imread
import pygame
import time
from sklearn.decomposition import PCA
from sklearn.preprocessing import StandardScaler

app = Flask(__name__)

pygame.mixer.init()

# Constants
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}
UPLOAD_FOLDER = r"C:\Users\shail\OneDrive\Desktop\DDD_App\App\static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER

# Load pre-trained model
with open('model_data.pkl', 'rb') as f:
    model_data = pickle.load(f)

class AudioAlert:
    def __init__(self, text):
        self.text = text

    def trigger(self):
        tts = gTTS(text=self.text, lang='en', slow=False)
        tts.save("alert_msg.mp3")

class ImageProcessor:
    def __init__(self, scaler, pca):
        self.scaler = scaler
        self.pca = pca

    def preprocess_image(self, filepath):
        img = imread(filepath)
        img_resized = resize(img, (150, 150, 3))
        flattened_image = img_resized.flatten().reshape(1, -1)
        scaled_image = self.scaler.transform(flattened_image)
        pca_transformed_image = self.pca.transform(scaled_image)
        return pca_transformed_image

class PredictionModel:
    def __init__(self, model):
        self.model = model

    def predict(self, processed_image):
        return self.model.predict(processed_image)

class DistractionDetector:
    def __init__(self, model, scaler, pca):
        self.image_processor = ImageProcessor(scaler, pca)
        self.prediction_model = PredictionModel(model)
        self.distraction_categories = {
            "0": "safe driving",
            "1": "texting - right",
            "2": "talking on the phone - right",
            "3": "texting - left",
            "4": "talking on the phone - left",
            "5": "operating the radio",
            "6": "drinking",
            "7": "reaching behind",
            "8": "hair and makeup",
            "9": "talking to passenger"
        }
        self.alert_thresholds = {
            "5": 0.1,  # Operating the radio
            "9": 0.1,  # Talking to passenger
            "2": 0.2,  # Talking on the phone - right
            "4": 0.2,  # Talking on the phone - left
            "1": 0.3,  # Texting - right
            "3": 0.3,  # Texting - left
            "7": 0.3,  # Reaching behind
            "6": 0.3,  # Drinking
            "8": 0.3   # Hair and makeup
        }

    def detect_distraction(self, filepath):
        processed_image = self.image_processor.preprocess_image(filepath)
        prediction = self.prediction_model.predict(processed_image)
        prediction = str(int(prediction[0]))
        return prediction

    def get_distraction_category(self, prediction):
        return self.distraction_categories[prediction]

    def handle_alert(self, prediction):
        detected_category = self.distraction_categories[prediction]
        alert_message = f"Warning: Detected {detected_category}! Please focus on driving."
        audio_alert = AudioAlert(alert_message)
        audio_alert.trigger()

        start = time.perf_counter()

        if self.alert_thresholds[prediction] < 0.2:
            print("\nLow Priority Risk: ",detected_category)
            audio_files = ["alert_msg.mp3", "low_priority.mp3"]
        elif self.alert_thresholds[prediction] > 0.1 and self.alert_thresholds[prediction] < 0.3:
            print("\nMedium Priority Risk: ",detected_category)
            audio_files = ["alert_msg.mp3", "medium_priority.wav"]
        else:
            print("\nHigh Priority Risk: ",detected_category)
            audio_files = ["alert_msg.mp3", "high_priority.wav"]

        end = time.perf_counter() - start
        print(f'{end:.6f}s taken for the warning to be conveyed')

        for audio in audio_files:
            pygame.mixer.music.load(audio)
            pygame.mixer.music.play()
            while pygame.mixer.music.get_busy():
                time.sleep(1)

# Initialize distraction detector
distraction_detector = DistractionDetector(model_data['model'], model_data['scaler'], model_data['pca'])

def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

@app.route('/home')
def index():
    return render_template('index.html')

@app.route('/predict', methods=['POST'])
def predict():
    if 'file' not in request.files:
        return render_template('index.html', prediction="No image found.")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No selected image.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        prediction = distraction_detector.detect_distraction(filepath)

        if prediction != "0":
            distraction_detector.handle_alert(prediction)

        prediction_category = distraction_detector.get_distraction_category(prediction)
        return render_template('index.html', prediction=prediction_category, image_url=filename)
    else:
        return render_template('index.html', prediction="Invalid file format.")

if __name__ == '__main__':
    app.run(debug=True)
