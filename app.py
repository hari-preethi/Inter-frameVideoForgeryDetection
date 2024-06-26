from flask import Flask, render_template, request, jsonify, send_file, url_for

from werkzeug.utils import secure_filename
import os
import numpy as np
from tensorflow.keras.models import load_model # type: ignore
from preprocess import video_to_grayscale
from msdcnn import feature_fusion
from lstmFC import reshape_to_3d

app = Flask(__name__)
app.template_folder = 'templates'

# Load your trained LSTM model
lstm_model = load_model("C:/Users/Preethi/Documents/InterframeVideoForgeryDetection/models/vfd_model.keras")

# Define functions for preprocessing and model prediction
def preprocess_video(video_path):
    preprocessed_video = video_to_grayscale(video_path)
    msd = feature_fusion(preprocessed_video)
    msd_reshaped = reshape_to_3d(msd)
    msd_re = np.array([msd_reshaped])
    return msd_re

def predict_forgery(msd_Feature):
    prediction = lstm_model.predict(msd_Feature)
    return round((float((prediction[0][0]))),4) # Convert to float

@app.route('/video/<filename>')
def serve_video(filename):
    video_path = os.path.join('uploads', filename)
    return send_file(video_path, mimetype='video/mp4')


@app.route('/')
def index():
    return render_template('index.html')

@app.route('/upload', methods=['POST'])
def upload_file():
    if request.method == 'POST':
        f = request.files['video']
        if f:
            filename = secure_filename(f.filename)
            video_path = os.path.join('uploads', filename)
            f.save(video_path)
            # Preprocess the uploaded video
            msd_feature = preprocess_video(video_path)
            # Predict forgery
            prediction = predict_forgery(msd_feature)
           
            # Return the prediction result as JSON
            return jsonify({'prediction': prediction, 'video_path': url_for('serve_video', filename=filename)})
    return render_template('index.html', error="No file uploaded")


if __name__ == '__main__':
    app.run(port=5000, debug=True)
