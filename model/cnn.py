import pandas as pd
import numpy as np
import tensorflow as tf
from tensorflow.keras.preprocessing.image import load_img, img_to_array
from tensorflow.keras import layers, models
import os
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
from sklearn.model_selection import train_test_split
from tensorflow.keras.optimizers import Adam
import matplotlib.pyplot as plt
from paths_to_csv_sql.sql_load import fetch_data_from_db


def load_and_encode_data(data_df):
    label_encoder = LabelEncoder()
    data_df['ClassId'] = label_encoder.fit_transform(data_df['ClassId'])
    return data_df

def load_and_preprocess_image(image_path):
    image = load_img(image_path, color_mode='grayscale', target_size=(64, 64))
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

def prepare_datasets(X, y, test_size=0.20, batch_size=32):
    X_train, X_val, y_train, y_val = train_test_split(X, y, test_size=test_size, random_state=42)

    train_dataset = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(buffer_size=1000).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)
    val_dataset = tf.data.Dataset.from_tensor_slices((X_val, y_val)).batch(batch_size).prefetch(buffer_size=tf.data.AUTOTUNE)

    return train_dataset, val_dataset

def build_model(input_shape, num_classes, learning_rate=0.0005):
    model = models.Sequential([
        layers.Conv2D(128, (3, 3), activation="relu", input_shape=input_shape),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.Conv2D(128, (3, 3), activation="relu"),
        layers.BatchNormalization(),
        layers.MaxPooling2D((2, 2)),
        layers.GlobalAveragePooling2D(),
        layers.Dense(64, activation="relu"),
        layers.Dropout(0.25),
        layers.Dense(num_classes, activation="softmax")
    ])

    optimizer = Adam(learning_rate=learning_rate)
    model.compile(optimizer=optimizer, loss='sparse_categorical_crossentropy', metrics=['accuracy'])

    return model

def train_model(model, train_dataset, val_dataset, epochs=15):
    early_stopping = EarlyStopping(monitor='val_loss', patience=10, restore_best_weights=True)
    reduce_lr = ReduceLROnPlateau(monitor='val_loss', factor=0.2, patience=1, min_lr=0.000001)

    data_augmentation = tf.keras.Sequential([
    # layers.RandomFlip("horizontal"),
    # layers.RandomRotation(0.2),
    # layers.RandomZoom(0.1),
    # layers.RandomBrightness(0.1),
    layers.RandomContrast(0.1)
    ])

    history = model.fit(
        train_dataset.map(lambda x, y: (data_augmentation(x, training=True), y)),
        epochs=epochs,
        validation_data=val_dataset,
        callbacks=[early_stopping, reduce_lr]
    )

    return history

def evaluate_model(model, val_dataset):
    val_loss, val_accuracy = model.evaluate(val_dataset)
    print(f'Validation Loss: {val_loss}')
    print(f'Validation Accuracy: {val_accuracy}')
    return val_loss, val_accuracy

def save_model(model, filename='C:\\Users\\zenklai\\db\\my_model.keras'):
    model.save(filename)

def plot_history(history):
    plt.figure(figsize=(12, 8))

    plt.subplot(2, 1, 1)
    plt.plot(history.history['loss'], label='Training Loss')
    plt.plot(history.history['val_loss'], label='Validation Loss')
    plt.title('Model Loss')
    plt.ylabel('Loss')
    plt.xlabel('Epoch')
    plt.legend()

    plt.subplot(2, 1, 2)
    plt.plot(history.history['accuracy'], label='Training Accuracy')
    plt.plot(history.history['val_accuracy'], label='Validation Accuracy')
    plt.title('Model Accuracy')
    plt.ylabel('Accuracy')
    plt.xlabel('Epoch')
    plt.legend()

    plt.tight_layout()
    plt.show()

def start_training1():
    dataset_dir = "C:\\Users\\zenklai\\data\\Final_Training\\Images"
    
    db_name = "C:\\Users\\zenklai\\db\\train.db"
    table_name = "train"

    data_df = fetch_data_from_db(db_name, table_name)
    data_df = load_and_encode_data(data_df)
    
    X, y = load_data(dataset_dir, data_df)
    
    train_dataset, val_dataset = prepare_datasets(X, y)
    
    num_classes = len(np.unique(y))
    model = build_model(input_shape=(64, 64, 1), num_classes=num_classes)
    
    history = train_model(model, train_dataset, val_dataset)
    
    evaluate_model(model, val_dataset)
    
    
    save_model(model)
    
    plot_history(history)