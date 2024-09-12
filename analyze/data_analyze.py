import sqlite3
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report
import numpy as np

data_df = pd.read_csv('C:\\Users\\zenklai\\db\\predictions.csv')

y_true = data_df['Actual_ClassId'].values
y_pred = data_df['Predicted'].values

class_names = [
    'Speed limit (20km/h)', 'Speed limit (30km/h)', 'Speed limit (50km/h)', 'Speed limit (60km/h)',
    'Speed limit (70km/h)', 'Speed limit (80km/h)', 'End of speed limit (80km/h)', 'Speed limit (100km/h)',
    'Speed limit (120km/h)', 'No passing', 'No passing veh over 3.5 tons', 'Right-of-way at intersection',
    'Priority road', 'Yield', 'Stop', 'No vehicles', 'Veh > 3.5 tons prohibited', 'No entry', 'General caution',
    'Dangerous curve left', 'Dangerous curve right', 'Double curve', 'Bumpy road', 'Slippery road',
    'Road narrows on the right', 'Road work', 'Traffic signals', 'Pedestrians', 'Children crossing',
    'Bicycles crossing', 'Beware of ice/snow', 'Wild animals crossing', 'End speed + passing limits',
    'Turn right ahead', 'Turn left ahead', 'Ahead only', 'Go straight or right', 'Go straight or left',
    'Keep right', 'Keep left', 'Roundabout mandatory', 'End of no passing', 'End no passing veh > 3.5 tons'
]

def fetch_data_from_db(db_path, table_name):
    conn = sqlite3.connect(db_path)
    query = f"SELECT * FROM {table_name}"
    df = pd.read_sql_query(query, conn)
    conn.close()
    return df

def plot_class_distribution(df):
    sns.countplot(x='ClassId', data=df)
    plt.title('Class Distribution in Dataset')
    plt.xlabel('Class')
    plt.ylabel('Count')
    plt.show()

def plot_it(df):
    sns.histplot(df['ClassId'])
    plt.title('Difrent graph')
    plt.xlabel('Feature Value')
    plt.ylabel('Count')
    plt.show()

def plot_evaluation_graphs(y_true, y_pred, class_names):
    print("Classification Report:")
    print(classification_report(y_true, y_pred, target_names=class_names))

def compare_prediction():
    true_counts = pd.Series(y_true).value_counts().sort_index()
    pred_counts = pd.Series(y_pred).value_counts().sort_index()

    counts_df = pd.DataFrame({
    'Actual': true_counts,
    'Predicted': pred_counts
    }).fillna(0).astype(int)

    counts_df.plot(kind='bar', figsize=(10, 6), width=0.8)
    plt.title('Class Distribution: Actual vs Predicted')
    plt.xlabel('Class ID')
    plt.ylabel('Count')
    plt.xticks(rotation=0)
    plt.legend(loc='upper right')
    plt.show()
    
    
def analyze_data_menu():
    print("\nWhat graph would you like to see?")
    print("1. Number of each class in the dataset")
    print("2. Class distribution")
    print("3. Show prediction results")
    print("4. Actual vs. Predicted count")

    choice = input("\nPlease enter your choice: ")
    db_name = "C:\\Users\\zenklai\\db\\train.db"
    table_name = "train"

    df = fetch_data_from_db(db_name, table_name)   

    if choice == '1':
        plot_class_distribution(df)
    elif choice == '2':
        plot_it(df)
    elif choice == '3':
        plot_evaluation_graphs(y_true, y_pred, class_names)
    elif choice == '4':
        compare_prediction()
    else:
        print("Invalid choice. Please try again.")
        analyze_data_menu()
