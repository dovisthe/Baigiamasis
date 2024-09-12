import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
import os
from sklearn.preprocessing import LabelEncoder
from paths_to_csv_sql.sql_load import fetch_data_from_db
from tensorflow.keras.models import load_model
import pandas as pd


def load_and_encode_data(data_df):
    label_encoder = LabelEncoder()
    data_df['ClassId'] = label_encoder.fit_transform(data_df['ClassId'])
    return data_df

def load_and_preprocess_image(image_path):
    image = load_img(image_path, color_mode='grayscale', target_size=(32, 32))
    image = img_to_array(image)
    image = image / 255.0
    return image

def load_data(dataset_dir, data_df):
    image_paths = [os.path.join(dataset_dir, path) for path in data_df['Path']] # a nuotrauka yra tokioj direktorijoje
    labels = data_df['ClassId'].values

    images = []
    for path in image_paths:
        image = load_and_preprocess_image(path)
        images.append(image)
    images = np.array(images)
    
    labels = np.array(labels)
    return images, labels

def prepare_datasets(X, y,batch_size=32):
    X_val,y_val = X, y
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    return val_dataset

def prd(model, val_dataset):
    predictions = model.predict(val_dataset)
    predicted_label = np.argmax(predictions, axis=1)
    return predicted_label

def load_trained_model(model_path='C:\\Users\\zenklai\\db\\my_model.keras'):
    return load_model(model_path)

def decode_label(encoded_label, label_encoder):
    return label_encoder.inverse_transform([encoded_label])[0]

def start_training_all(model_path='C:\\Users\\zenklai\\db\\my_model.keras', output_csv='C:\\Users\\zenklai\\db\\predictions.csv'):
    dataset_dir = "C:\\Users\\zenklai\\data\\Final_test\\Images"
    
    db_name = "C:\\Users\\zenklai\\db\\test.db"
    table_name = "test"

    data_df = fetch_data_from_db(db_name, table_name)
    data_df = load_and_encode_data(data_df)
    
    data_df['File_Name'] = data_df['Path'].apply(lambda x: os.path.basename(x))
    
    X, y = load_data(dataset_dir, data_df)
    
    model = load_trained_model(model_path)
    
    val_dataset = prepare_datasets(X, y)
    
    predictions = prd(model, val_dataset)
    
    results_df = pd.DataFrame({
        'File_Name': data_df['File_Name'],
        'Actual_ClassId': data_df['ClassId'],
        'Predicted': predictions
    })
    
    results_df.to_csv(output_csv, index=False)
    
    print(results_df)