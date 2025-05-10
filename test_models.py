import cv2
import os
import numpy as np
import mediapipe as mp

from tensorflow.keras.models import load_model # type: ignore
import joblib

# SVC
gesture_model_svc = joblib.load("gesture_model.joblib")
gesture_scaler_svc = joblib.load("gesture_scaler.joblib")

# neural network
gesture_model_nn = load_model("gesture_model.keras")
gesture_scaler_nn = joblib.load("gesture_scaler.joblib")
label_encoder = joblib.load("gesture_label_encoder.joblib")

def predict_svm(model, scaler, landmark_points):
    """
    predict using SVM
    """
    input_data = landmark_points.reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)
    prediction = model.predict(input_data_scaled)

    return prediction

def predict_nn(model, scaler, label_encoder, landmark_points):
    input_data = landmark_points.reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)

    prediction_probs = model.predict(input_data_scaled)[0] # get softmax probabilites
    predicted_class_index = np.argmax(prediction_probs)
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])

    predicted_probability = prediction_probs[predicted_class_index]

    # print(f"Predicted gesture: {predicted_class_label} ({predicted_probability:.2%} confidence)")
    return predicted_class_label, predicted_probability

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
                        
                        sub.append(landmark_array.flatten())
        img_data[sub_dir] = sub

    hands.close()
    return img_data

def evaluate_models(image_dir):
    data = load_images(image_dir)
    total = 0
    svm_correct = 0
    nn_correct = 0
    for true_label, samples in data.items():
        for sample in samples:
            pred1 = predict_svm(gesture_model_svc, gesture_scaler_svc, sample) # SVC
            pred2, _ = predict_nn(gesture_model_nn, gesture_scaler_nn, label_encoder, sample) # neural network

            print(f"True: {true_label}")
            print(f" Model 1 Prediction: {pred1}")
            print(f" Model 2 Prediction: {pred2}")
            print("---------------")
            if pred1 == true_label:
                svm_correct += 1
            if pred2 == true_label:
                nn_correct += 1
            total += 1
    
    print("Results:")
    print("---------------")
    print(f'SVM: {(svm_correct / total): .2%}')
    print(f'NN: {(nn_correct / total): .2%}')

if __name__ == "__main__":
    evaluate_models("captures")