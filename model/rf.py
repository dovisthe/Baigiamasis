import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestClassifier
from sklearn.model_selection import train_test_split
from sklearn.metrics import accuracy_score, classification_report
import os
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from paths_to_csv_sql.sql_load import fetch_data_from_db
import joblib
from model.zenkluklase import classes
from PIL import Image
from skimage.feature import hog
from skimage import exposure

def load_and_preprocess_image(image_path):
    image = load_img(image_path, color_mode='grayscale', target_size=(64, 64))
    image = img_to_array(image)
    image = image.squeeze()  # nuima dimensija su 1 skaiciumi
    hog_features, hog_image = hog(image, pixels_per_cell=(8, 8),
                                  cells_per_block=(2, 2), visualize=True,
                                  block_norm='L2-Hys')
    return hog_features

def load_and_preprocess_images(image_paths):
    images = [load_and_preprocess_image(path) for path in image_paths]
    return np.array(images)

def load_image_paths(dataset_dir, data_df):
    image_paths = [os.path.join(dataset_dir, path) for path in data_df['Path']]
    labels = data_df['ClassId'].values
    return image_paths, labels

def train_random_forest(X, y):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=0.2, random_state=42)
    
    rf_model = RandomForestClassifier(n_estimators=100, random_state=42, n_jobs=-1)
    rf_model.fit(X_train, y_train)
    
    y_pred = rf_model.predict(X_val)
    accuracy = accuracy_score(y_val, y_pred)
    print(f"Validation Accuracy: {accuracy}")
    print(classification_report(y_val, y_pred))
    
    return rf_model

def save_model(model, filename='C:\\Users\\zenklai\\db\\random_forest_model.pkl'):
    joblib.dump(model, filename)

def start_training():
    dataset_dir = "C:\\Users\\zenklai\\data\\Final_Training\\Images"
    db_name = "C:\\Users\\zenklai\\db\\train.db"
    table_name = "train"
    
    data_df = fetch_data_from_db(db_name, table_name)
    image_paths, labels = load_image_paths(dataset_dir, data_df)
    features = load_and_preprocess_images(image_paths)
    
    model = train_random_forest(features, labels)
    save_model(model)

def predict_image(image_path, model_path='C:\\Users\\zenklai\\db\\random_forest_model.pkl'):
    
    if not os.path.isfile(image_path):
            print("File not found. Please try again.")
            return
    
    
    model = joblib.load(model_path)
    
    image = load_and_preprocess_image(image_path).reshape(1, -1)  # vienas sample, -1 auto apskaiciuoja dydi(number of features)
    
    prediction = model.predict(image)
    predicted_label = prediction[0]

    print(f'Predicted Class Description: {classes.get(predicted_label, "Unknown class")}')
    
    image_to_show = Image.open(image_path)
    image_to_show.show()
    
    return prediction[0]
