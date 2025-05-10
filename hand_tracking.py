import cv2
import mediapipe as mp
import numpy as np
import time
from collections import Counter

import joblib
from tensorflow.keras.models import load_model # type: ignore

import socket
server_ip = '127.0.0.1'
server_port = 12345
client_socket = socket.socket(socket.AF_INET, socket.SOCK_STREAM)
client_socket.connect((server_ip, server_port))
client_socket.setblocking(False) # continue if no message received
client_socket.sendall("input".encode("utf-8"))

# def predict_gesture(model, scaler, landmark_points):
#     """
#     predict using SVM
#     """
#     input_data = landmark_points.reshape(1, -1)
#     input_data_scaled = scaler.transform(input_data)
#     prediction = model.predict(input_data_scaled)

#     return prediction

def predict_gesture(model, scaler, label_encoder, landmark_points):
    input_data = landmark_points.reshape(1, -1)
    input_data_scaled = scaler.transform(input_data)

    prediction_probs = model.predict(input_data_scaled)[0] # get softmax probabilites
    predicted_class_index = np.argmax(prediction_probs)
    predicted_class_label = label_encoder.inverse_transform([predicted_class_index])

    predicted_probability = prediction_probs[predicted_class_index]

    # print(f"Predicted gesture: {predicted_class_label} ({predicted_probability:.2%} confidence)")
    return predicted_class_label, predicted_probability


def main():

    # # SVC
    # gesture_model = joblib.load("gesture_model.joblib")
    # gesture_scaler = joblib.load("gesture_scaler.joblib")

    # neural network
    gesture_model = load_model("gesture_model.keras")
    gesture_scaler = joblib.load("gesture_scaler.joblib")
    label_encoder = joblib.load("gesture_label_encoder.joblib")


    mp_hands = mp.solutions.hands
    hands = mp_hands.Hands()

    mp_drawing = mp.solutions.drawing_utils

    # cap = cv2.VideoCapture(0)
    cap = cv2.VideoCapture(1)

    last_time = 0
    predictions = []
    while cap.isOpened():
        ret, frame = cap.read()

        if not ret:
            print("[WARNING] Frame not read")
            time.sleep(0.1)  # Give camera a little break
            continue

        frame_rgb = cv2.cvtColor(frame, cv2.COLOR_BGR2RGB)

        results = hands.process(frame_rgb)

        if results.multi_hand_landmarks:
            for landmark in results.multi_hand_landmarks:
                # print(landmark)
                landmark_array = np.zeros((21, 3))
                for i, l in enumerate(landmark.landmark):
                    landmark_array[i, 0] = l.x
                    landmark_array[i, 1] = l.y
                    landmark_array[i, 2] = l.z

                # TODO: should this be flattened or not?
                # predicted_gesture = predict_gesture(gesture_model, gesture_scaler, landmark_array.flatten()) # SVC
                predicted_gesture, predicted_prob = predict_gesture(gesture_model, gesture_scaler, label_encoder, landmark_array) # neural network
                # print(predicted_gesture)
                predictions.append(predicted_gesture[0])
                
                mp_drawing.draw_landmarks(frame, landmark, mp_hands.HAND_CONNECTIONS)
                cv2.putText(frame, f"{predicted_gesture[0]} {predicted_prob:.2%}", (10, 500), fontFace=cv2.FONT_HERSHEY_SIMPLEX, fontScale=1,
                            color=(255, 255, 255), thickness=1, lineType=2)

        cv2.imshow('Hand Recognition', frame)
        cv2.resizeWindow('Hand Recognition', 640, 480)

        if cv2.waitKey(10) & 0xFF == ord('q'):
            break
    
        # make sure that prediction is only accepted if consistent over a period
        # current configuration requires around 3/4 second to execute command
        num_predictions = 20 # there are around 16 predictions per second
        if len(predictions) > num_predictions:
            predictions.pop(0) # retain only recent ones

            c = Counter(predictions)
            pred, num = c.most_common(1)[0] # gets the most common prediction and its occurences
            if num > int(num_predictions * 0.85):
                # print(pred)
                if (time.time() - last_time) > 0.1:
                    client_socket.send(pred.encode('utf-8'))
                    last_time = time.time()
            
    
    cap.release()
    cv2.destroyAllWindows()
    client_socket.send("exit".encode('utf-8'))
    client_socket.close()

if __name__ == "__main__":
    main()
