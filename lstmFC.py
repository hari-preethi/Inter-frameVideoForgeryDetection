import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import LSTM , Attention, Dense, Input # type: ignore
from tensorflow.keras.models import Model # type: ignore
from msdcnn import feature_fusion
from preprocess import video_to_grayscale


class CustomAttentionCell(tf.keras.layers.Layer):
    def __init__(self, lstm_layer, step_length):
        super(CustomAttentionCell, self).__init__()
        self.lstm_layer = lstm_layer
        self.step_length = step_length
        self.attention_mechanism = tf.keras.layers.Attention(
            use_scale=False,
            dropout=0.2
        )

    def call(self, inputs):
        lstm_output = self.lstm_layer(inputs)
        attention_output = self.attention_mechanism([lstm_output, lstm_output])
        return attention_output

# Define your LSTM cell
#lstm_cell = tf.keras.layers.LSTMCell(144)

# Wrap the LSTM cell with the CustomAttentionCell
#custom_attention_cell = CustomAttentionCell(lstm_cell, 2)

# Now you can use this custom_attention_cell in your model



"""def lstm():
    input_layer = Input(shape=(None,96))
    lstm_layer = LSTM(144, dropout=0.2, recurrent_dropout=0.2)(input_layer)
    #attention_output = Attention(use_scale=False, dropout=0.2)([lstm_layer, lstm_layer])
    output_layer = Dense(1, activation='softmax')(lstm_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model"""


def lstm():
    input_layer = Input(shape=(None,96))
    lstm_layer = LSTM(144, dropout=0.2, recurrent_dropout=0.2)(input_layer)
    output_layer = Dense(1, activation='sigmoid')(lstm_layer)
    model = Model(inputs=input_layer, outputs=output_layer)
    return model

def reshape_to_3d(input_data):
    # Reshape array to 3D
    """reshaped_data = np.squeeze(input_data)  # Remove single-dimensional entries
    reshaped_data = np.reshape(reshaped_data, (reshaped_data.shape[0], -1, 1))
    return reshaped_data"""
    reshaped_tensor = tf.squeeze(input_data)
    # Reshape the tensor to 3D
    reshaped_tensor = tf.reshape(reshaped_tensor, (tf.shape(reshaped_tensor)[0], -1, 1))
    return reshaped_tensor

def predict_output(output):
  for i in range(0, len(output)):    
    avg_prob = np.mean(output[i][0])
    if avg_prob > 0.5:
        print("Forged -", i, avg_prob)
    else:
        print("Original -", i, avg_prob)


if __name__=='__main__':

    input_video_path = 'D:/Final_Project/Dataset/VideoForgeryDataset/Training/Forged/Forgery_insertion/vid131.avi'
    preprocessed_video = video_to_grayscale(input_video_path)
    msd = feature_fusion(preprocessed_video)
    print(msd.shape)
    msd_reshaped = reshape_to_3d(msd)
    print(msd_reshaped.shape)
    lstm_model = lstm()
    output = lstm_model.predict(msd_reshaped)
    print(output,"\n",output.shape)
