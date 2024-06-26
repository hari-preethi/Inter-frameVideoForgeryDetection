import numpy as np
import cv2
import tensorflow as tf
from tensorflow.keras import Sequential # type: ignore
from tensorflow.keras.layers import Conv2D, MaxPooling2D, AveragePooling2D, BatchNormalization # type: ignore

from difference_msdcnn import forward_difference, post_difference, display_forward_difference,display_post_difference
from preprocess import video_to_grayscale

def create_cnn_model(input_shape):

    cnn = Sequential()

    cnn.add(tf.keras.Input(shape=input_shape))
    cnn.add(Conv2D(filters=2, kernel_size=(5, 5), strides=(1, 1), padding='valid', activation='elu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    
    cnn.add(Conv2D(filters=4, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu'))
    cnn.add(MaxPooling2D(pool_size=(2, 2)))
    
    cnn.add(Conv2D(filters=8, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu'))
    cnn.add(AveragePooling2D(pool_size=(2, 2)))
    
    cnn.add(Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu'))
    cnn.add(AveragePooling2D(pool_size=(2, 2)))
    
    cnn.add(Conv2D(filters=24, kernel_size=(1, 1), strides=(1, 1), padding='valid', activation='elu'))
    cnn.add(AveragePooling2D(pool_size=(2, 2)))
    
    cnn.add(Conv2D(filters=48, kernel_size=(3, 3), strides=(1, 1), padding='valid', activation='elu'))
    cnn.add(BatchNormalization())
    cnn.compile(loss="mse", optimizer = "adam")
    return cnn
    #cnn.summary()


input_shape=(128,128,1)
#CNN model for FDCNN
fdcnn = create_cnn_model(input_shape)
#CNN model for PDCNN
pdcnn = create_cnn_model(input_shape)

#Extracts pixel differene features
def fdfeature_extraction(preprocessed_video,fdcnn):
    forward_diff = forward_difference(preprocessed_video)
    display_forward_difference(preprocessed_video)
    fdfeature = []
    for frame in forward_diff:
        frame = frame.reshape((128, 128, 1))  
        fdf = fdcnn.predict(frame[np.newaxis, ...])
        fdfeature.append(fdf)
    return fdfeature

#Extracts deep differene features
def pdfeature_extraction(preprocessed_video,pdcnn):
    pdfeature = []
    for frame in preprocessed_video:
        frame = frame.reshape((128, 128, 1))  
        pdf = pdcnn.predict(frame[np.newaxis, ...])
        pdfeature.append(pdf)
    post_diff = post_difference(pdfeature)
    return post_diff

#Concatenates pixel difference features and deep difference features
def feature_fusion(preprocessed_video):
    pixel_difference_feature = fdfeature_extraction(preprocessed_video,fdcnn)
    deep_difference_feature = pdfeature_extraction(preprocessed_video,pdcnn)
    msd_feature = []
    for pdf, ddf in zip(pixel_difference_feature, deep_difference_feature):
        concatenated_feature = tf.concat([pdf, ddf], axis=-1)
        msd_feature.append(concatenated_feature)
    return tf.stack(msd_feature, axis=0)

if __name__=='__main__':
    input_video_path = 'D:/Final_Project/Dataset/ForgeryDataset/Deletion/Training/Original/original_train (20).avi'
    preprocessed_video = video_to_grayscale(input_video_path)
    msd = feature_fusion(preprocessed_video)
    print("Multi-Scale Difference Feature : ",msd)
    print("Shape of Multi-Scale Difference Feature : ",msd.shape)