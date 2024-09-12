import os
import pandas as pd

input_csv = 'C:\\Users\\zenklai\\GT-final_test.csv'
output_csv = 'C:\\Users\\zenklai\\test.csv'
images_folder = 'C:\\Users\\zenklai\\data\\Final_Test\\Images'

df = pd.read_csv(input_csv, delimiter=';')

df['Path'] = df['Filename'].apply(lambda x: os.path.join(images_folder, x))
df_cnn = df[['Path', 'ClassId']]

df_cnn.to_csv(output_csv, index=False)

root_folder = 'C:\\Users\\zenklai\\data\\Final_Training\\Images'
output_csv = 'C:\\Users\\zenklai\\train.csv'

allowed_extensions = {'.jpg', '.jpeg', '.png', '.ppm'}

data = []

for class_folder in os.listdir(root_folder):
    class_folder_path = os.path.join(root_folder, class_folder)
    
    if os.path.isdir(class_folder_path):
        for image_filename in os.listdir(class_folder_path):
            image_path = os.path.join(class_folder_path, image_filename)
            
            _, ext = os.path.splitext(image_filename) # _ reiskia root nereikalinas, tik pabaiga pvz .jpg
            if ext.lower() in allowed_extensions:
                data.append({'Path': image_path, 'ClassId': class_folder})

df_cnn = pd.DataFrame(data)
df_cnn['ClassId'] = df_cnn['ClassId'].astype(int)
df_cnn.to_csv(output_csv, index=False)
print(f"CSV file saved to {output_csv} with {len(df_cnn)} entries.")