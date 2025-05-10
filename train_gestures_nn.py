import cv2
import mediapipe as mp
import numpy as np
import os
import joblib

from tensorflow.keras.models import Sequential # type: ignore
from tensorflow.keras.layers import Dense, Dropout # type: ignore
from tensorflow.keras.utils import to_categorical # type: ignore
from sklearn.preprocessing import LabelEncoder
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

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
    # encode labels
    label_encoder = LabelEncoder()
    y_encoded = label_encoder.fit_transform(y_train)
    y_categorical = to_categorical(y_encoded)

    # scale features
    scaler = StandardScaler()
    X_train_scaled = scaler.fit_transform(X_train)

    # define model
    model = Sequential([
        Dense(128, activation='relu', input_shape=(X_train.shape[1],)),
        Dropout(0.3),
        Dense(64, activation='relu'),
        Dropout(0.3),
        Dense(y_categorical.shape[1], activation='softmax')
    ])

    model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['accuracy'])

    # train model
    model.fit(X_train_scaled, y_categorical, epochs=30, batch_size=32, verbose=1)

    return model, scaler, label_encoder


def evaluate_model(model, scaler, label_encoder, X_test, y_test):
    X_test_scaled = scaler.transform(X_test)
    y_test_encoded = label_encoder.transform(y_test)
    y_test_categorical = to_categorical(y_test_encoded)

    loss, accuracy = model.evaluate(X_test_scaled, y_test_categorical)
    print(f'Accuracy: {accuracy:.4f}')

if __name__ == "__main__":
    dir_path = os.getcwd() + "/HandImages/"
    hand_data = load_images(dir_path)

    X, y = preprocess_data(hand_data)

    X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2)

    model, scaler, label_encoder = create_model(X_train, y_train)

    evaluate_model(model, scaler, label_encoder, X_test, y_test)
    class_names = label_encoder.classes_.tolist()
    print(class_names)

    model.save("gesture_model.keras")
    joblib.dump(scaler, "gesture_scaler.joblib")
    joblib.dump(label_encoder, "gesture_label_encoder.joblib")