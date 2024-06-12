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

"""
# Load your pre-trained machine learning model from pickle file
with open('ddd.pkl', 'rb') as f:
    model = pickle.load(f)
"""

# Load your pre-trained machine learning model from pickle file with PCA
with open('model_data.pkl', 'rb') as f:
    model = pickle.load(f)

# Define the allowed extensions for file uploads
ALLOWED_EXTENSIONS = {'png', 'jpg', 'jpeg'}

# Define the upload folder
UPLOAD_FOLDER = r"C:\Users\shail\OneDrive\Desktop\DDD_App\App\static"
app.config['UPLOAD_FOLDER'] = UPLOAD_FOLDER


def trigger_audio_alert(text):
    #text to speech
    tts =gTTS(text=text,lang='en',slow=False)
    tts.save("alert_msg.mp3")


def allowed_file(filename):
    return '.' in filename and filename.rsplit('.', 1)[1].lower() in ALLOWED_EXTENSIONS

@app.route('/')
def home():
    return render_template('home.html')

#home page where the image will be uploaded
@app.route('/home')
def index():
    return render_template('index.html')

#predicted distraction class and auditory alerts to the driver
@app.route('/predict', methods=['POST'])
def predict():

    start=0.0
    end=0.0

    distraction_categories = {
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

    alert_thresholds = {

    "5": 0.1,  # Operating the radio
    "9": 0.1,  # Talking to passenger

    "2": 0.2,  # Talking on the phone - right
    "4": 0.2,  # Talking on the phone - left
    "7": 0.2,  # Reaching behind

    "1": 0.3,  # Texting - right
    "3": 0.3,  # Texting - left
    "6": 0.3,  # Drinking
    "8": 0.3   # Hair and makeup
    }

    if 'file' not in request.files:
        return render_template('index.html', prediction="No image found.")

    file = request.files['file']

    if file.filename == '':
        return render_template('index.html', prediction="No selected image.")

    if file and allowed_file(file.filename):
        filename = secure_filename(file.filename)
        filepath = os.path.join(app.config['UPLOAD_FOLDER'], filename)
        file.save(filepath)

        svc_model=model['model']
        scaler = model['scaler']
        pca = model['pca']


        # Preprocess the image
        img = imread(filepath)
        img_resize=resize(img,(150,150,3))
        flattened_image = img_resize.flatten().reshape(1, -1)

        # 2. Standardize the image
        flattened_image_scaled = scaler.transform(flattened_image)

        # 3. Apply PCA transformation
        flattened_image_pca = pca.transform(flattened_image_scaled)
        
        # Make prediction
        Pred_StartTime=time.time()
        prediction = svc_model.predict(flattened_image_pca)
        Pred_EndTime=time.time()

        print("Time taken for prediction by ML Model: ",Pred_EndTime-Pred_StartTime)

        # Process prediction to get readable output
        # Example: convert prediction array to class label or probability
        prediction=int(prediction)
        prediction=str(prediction)

        #Prediction received time
        start = time.perf_counter()


        #Auditory Alerts
        if prediction != "0":

            detected_category=distraction_categories[prediction]
            alert_message = f"Warning: Detected {detected_category}! Please focus on driving."
            #alert_message = f" ध्यान दें! आपका ड्राइविंग व्यवहार {detected_category} खतरनाक है कृपया सड़क पर ध्यान केंद्रित करें।"
        
            trigger_audio_alert(alert_message)


            if alert_thresholds[prediction]<0.2:  
                audio_files=["alert_msg.mp3","low_priority.mp3"]
                
                #Warning message recieved and audio starts playing
                print("hello")
                end = time.perf_counter() - start
                print('{:.6f}s taken for the warning to be conveyed'.format(end))

                for audio in audio_files:
                    pygame.mixer.music.load(audio)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(1)
               

            elif alert_thresholds[prediction]>0.1 and alert_thresholds[prediction]<0.3:
                audio_files=["alert_msg.mp3","medium_priority.wav"]

                #Warning message recieved and audio starts playing
                print("hello")
                end = time.perf_counter() - start
                print('{:.6f}s taken for the warning to be conveyed'.format(end))

                for audio in audio_files:
                    pygame.mixer.music.load(audio)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(1)

            else:
                audio_files=["alert_msg.mp3","high_priority.wav"] 

                print("hey")
                #Warning message recieved and audio starts playing
                end = time.perf_counter() - start
                print('{:.6f}s taken for the warning to be conveyed'.format(end))

                for audio in audio_files:
                    pygame.mixer.music.load(audio)
                    pygame.mixer.music.play()
                    while pygame.mixer.music.get_busy():
                        time.sleep(1)
                
        #Predicted Distracted Driver category
        prediction=distraction_categories[prediction]
        
        return render_template('index.html', prediction=prediction, image_url=filename)
    else:
        return render_template('index.html', prediction="Invalid file format.")

if __name__ == '__main__':
    app.run(debug=True)
