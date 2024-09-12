from model.cnn import start_training1
from model.predict import start_prediction
from model.add_photo import handle_new_photo
from model.rf import start_training, predict_image
from analyze.data_analyze import analyze_data_menu
from model.fix import start_training_all

BASE_IMAGE_PATH = "C:\\Users\\zenklai\\data\\Final_Test\\Images\\"

def main_menu():
    print("\nCNN Training Program")
    print("1. Start Training CNN")
    print("2. Predict Image on CNN Wiht Your Link")
    print("3. Predict Image on CNN from Test Folder")
    print("4. Add New Photo and Update Database")
    print("\nFor Random Forest")
    print("5. Start Random Forest Learning")
    print("6. Predict Image on Random forest from Test")
    print("\n7. For graphs and info")
    print("8. Predict all from test folder")
    print('\n9. Exit')


    choice = input("\nPlease enter your choice: ")

    if choice == '1':
        print("Starting the training process...")
        start_training1()
        main_menu()
    elif choice == '2':
        image_path = input(r"Enter the path to the image file: ")
        start_prediction(image_path)
        main_menu()
    elif choice == '3':
        image_number = input(f"Enter the image number for CNN (e.g., 00001 to 10000): ")
        image_path = f"{BASE_IMAGE_PATH}{image_number.zfill(5)}.ppm"
        print(f"Using image path: {image_path}")
        start_prediction(image_path)
        main_menu()
    elif choice == '4':
        handle_new_photo()
        main_menu()
    elif choice == '5':
        print("Starting random forest learning")
        start_training()
        main_menu()
    elif choice == '6':
        image_number = input(f"Enter the image number for random forest(e.g., 00001 to 10000): ")
        image_path = f"{BASE_IMAGE_PATH}{image_number.zfill(5)}.ppm"
        print(f"Using image path: {image_path}")
        predicted_class = predict_image(image_path)
        print(f"Predicted Class: {predicted_class}")
        main_menu()
    elif choice == '7':
        analyze_data_menu()
        main_menu()
    elif choice == '8':
        start_training_all()
        main_menu()
    elif choice == '9':
        print("Exiting the program. Goodbye!")
    else:
        print("Invalid choice. Please try again.")
        main_menu()

if __name__ == "__main__":
    main_menu()