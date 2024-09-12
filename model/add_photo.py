import os
import sqlite3

def add_new_photo_to_db(image_path, class_id, db_path='C:\\Users\\zenklai\\db\\train.db'):
    conn = sqlite3.connect(db_path)
    cursor = conn.cursor()
    
    cursor.execute('''
        INSERT INTO train (Path, ClassId)
        VALUES (?, ?)
    ''', (image_path, class_id))
    
    conn.commit()
    conn.close()

def handle_new_photo():
    image_path = input(r"Enter the path to new image file: ")
    
    if not os.path.isfile(image_path):
        print("File not found. Try again.")
        return
    
    dataset_dir = "C:\\Users\\zenklai\\data\\Final_Training\\Images"
    new_image_name = os.path.basename(image_path)   #grazina foto name tik
    new_image_path = os.path.join(dataset_dir, new_image_name)   #sukuria nauja path nuotraukai
    
    try:
        os.rename(image_path, new_image_path)  #pakeicia foto direktorija
    except Exception as e:
        print(f"Error moving file: {e}")
        return
    
    from model.zenkluklase import classes
    print(classes)
    
    class_id = input("Enter the class ID for this image: ")
    
    if not class_id.isdigit():
        print("Invalid class ID. Please enter a number.")
        return
    
    class_id = int(class_id)
    
    if class_id > 43:
        print("No such class exists.")
        return
    
    add_new_photo_to_db(new_image_path, class_id)
    
    print("New image added and database updated successfully.")