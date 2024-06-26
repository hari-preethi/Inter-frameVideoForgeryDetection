import os
import numpy as np # type: ignore
import pandas as pd # type: ignore
from keras.utils import to_categorical # type: ignore
from keras.preprocessing.sequence import pad_sequences # type: ignore
from tensorflow.keras.models import load_model # type: ignore

from preprocess import video_to_grayscale, video_to_grayscale_extract_frames
from msdcnn import feature_fusion
from lstmFC import reshape_to_3d
from lstmFC import lstm

def predict_forgery(vidpath):
    vfd_model = load_model("C:/Users/Preethi/Documents/InterframeVideoForgeryDetection/models/vfd_model.keras")
    preprocessed_video = video_to_grayscale(vidpath)
    msd = feature_fusion(preprocessed_video)
    msd_reshaped = reshape_to_3d(msd)
    msd_re = np.array([msd_reshaped])
    prediction = vfd_model.predict(msd_re)
    return prediction[0]

if __name__ == '__main__':
    pred1 = predict_forgery("D:/Final_Project/example video/vid3.mp4")
    threshold = 0.5
    print("Prediction : ",pred1)
    if pred1 >= threshold:
        print("Video is forged")
    else:
        print("Video is original")
