import cv2
import mediapipe as mp
import numpy as np
import os

from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler
from sklearn.svm import SVC
from sklearn.metrics import accuracy_score, classification_report
import joblib

def augment_landmark_data(landmark_array, augmentation_factor=5, max_variation=0.01):
    """
    Augments hand landmark data by randomly altering coordinates, increasing dataset size without
    significant variation
    """
    augmented_data = [landmark_array]  # Start with the original data

    for _ in range(augmentation_factor):
        augmented_sample = landmark_array.copy()  # Create a copy to modify
        for i in range(21):
            for j in range(3):
                # Calculate the random variation
                variation = np.random.uniform(-max_variation, max_variation)
                # Apply the variation to the coordinate
                augmented_sample[i, j] += augmented_sample[i, j] * variation
        augmented_data.append(augmented_sample)
    return augmented_data

def load_images(dir):
    # initialize hand tracking
    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    img_data = {}
    for sub_dir in [x for x in os.listdir(dir) if x != ".DS_Store"]:
        sub_path = os.path.join(dir, sub_dir)
        sub = []
        for img in [x for x in os.listdir(sub_path) if x != ".DS_Store"]:
            image_path = os.path.join(sub_path, img)
            if os.path.isfile(image_path):
                image = cv2.imread(image_path)

                if image is None:
                    print(f"Could not open or find the image: {image_path}")
                    continue
                image_rgb = cv2.cvtColor(image, cv2.COLOR_BGR2RGB)
                results = hands.process(image_rgb)
                landmarks = results.multi_hand_landmarks
                
                if landmarks:
                    for hand_landmarks in landmarks:
                        landmark_array = np.zeros((21, 3))
                        for i, landmark in enumerate(hand_landmarks.landmark):
                            landmark_array[i, 0] = landmark.x
                            landmark_array[i, 1] = landmark.y
                            landmark_array[i, 2] = landmark.z
                        
                        augmented_list = augment_landmark_data(landmark_array)
                        sub.extend(augmented_list)
        img_data[sub_dir] = sub

    hands.close()
    return img_data

def preprocess_data(data):
    """
    Preprocesses the data from the dictionary into a format that can 
    train a machine learning model
    """
    X = []
    y = []

    for gesture_name, gesture_data in data.items():
        for landmark_array in gesture_data:
            # Flatten (21, 3) array into (63,) array
            X.append(landmark_array.flatten())
            y.append(gesture_name)
    return np.array(X), np.array(y)
                
def create_model(X_train, y_train):
    """
    Create and train a SVM model
    """
    # feature scaling
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # create and train SVM model
    model = SVC(kernel="rbf", C=1, gamma='scale')
    model.fit(X_train_scaled, y_train)

    return model, scaler

def evaluate_model(model, scaler, X_test, y_test):
    """
    Evaluate the performance of the SVM model
    """
    # scale test data using same scaler used for training
    X_test_scaled = scaler.transform(X_test)

    # make predictions
    y_pred = model.predict(X_test_scaled)

    accuracy = accuracy_score(y_test, y_pred)
    print(f"Accuracy: {accuracy:.4f}")
    print("Classification Report:")
    print(classification_report(y_test, y_pred))


if __name__ == "__main__":
    dir_path = os.getcwd() + "/HandImages/"
    hand_data = load_images(dir_path)

    X, y = preprocess_data(hand_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model, scaler = create_model(X_train, y_train)

    evaluate_model(model, scaler, X_test, y_test)

    model_filename, scaler_filename = "gesture_model.joblib", "gesture_scaler.joblib"
    joblib.dump(model, model_filename)
    joblib.dump(scaler, scaler_filename)

