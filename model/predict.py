import numpy as np
from tensorflow.keras.models import load_model
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import pandas as pd
from sklearn.preprocessing import LabelEncoder
from model.zenkluklase import classes
from paths_to_csv_sql.sql_load import fetch_data_from_db
from PIL import Image
import os


def load_trained_model(model_path='C:\\Users\\zenklai\\db\\my_model.keras'):
    return load_model(model_path)

def load_and_preprocess_image(image_path, target_size=(32, 32)):
    image = load_img(image_path, color_mode='grayscale', target_size=target_size)
    image = img_to_array(image)
    image = image / 255.0
    image = np.expand_dims(image, axis=0)   #nes darem batchsize, nauje dimension dadedame
    return image

def decode_label(encoded_label, label_encoder):
    return label_encoder.inverse_transform([encoded_label])[0]

def predict_image_class(model, image_path, label_encoder, target_size=(64, 64)):
    preprocessed_image = load_and_preprocess_image(image_path, target_size)
    predictions = model.predict(preprocessed_image)
    predicted_label = np.argmax(predictions, axis=1)[0]   
    return decode_label(predicted_label, label_encoder)

def start_prediction(image_path, model_path='C:\\Users\\zenklai\\db\\my_model.keras', data_csv="C:\\Users\\zenklai\\db\\train.db"):
    if not os.path.isfile(image_path):
        print("File not found. Please try again.")
        return
    
    table_name = "train"
    data_df = fetch_data_from_db(data_csv, table_name)    
    model = load_trained_model(model_path)

    label_encoder = LabelEncoder()
    label_encoder.fit(data_df['ClassId'])
    
    predicted_label = predict_image_class(model, image_path, label_encoder)
    
    print(f'Predicted Class Index: {predicted_label}')
    print(f'Predicted Class Description: {classes.get(predicted_label, "Unknown class")}')

    image = Image.open(image_path)
    image.show()