import cv2 
import numpy as np 
import mediapipe as mp 
from keras.models import load_model 

# Load pre-trained model and labels
model = load_model("model.h5")
label = np.load("labels.npy")

# Initialize MediaPipe holistic and hand models
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Open webcam
cap = cv2.VideoCapture(0)

while True:
    lst = []

    # Read frame from webcam
    _, frm = cap.read()

    # Flip frame horizontally for better user experience
    frm = cv2.flip(frm, 1)

    # Process frame with MediaPipe Holistic model
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Extract features from detected landmarks
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            for i in range(42):
                lst.append(0.0)

        # Reshape feature array for prediction
        lst = np.array(lst).reshape(1, -1)

        # Predict label using the pre-trained model
        pred = label[np.argmax(model.predict(lst))]

        # Print predicted label
        print(pred)

        # Display predicted label on the frame
        cv2.putText(frm, pred, (50, 50), cv2.FONT_ITALIC, 1, (255, 0, 0), 2)

    # Draw landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display the frame
    cv2.imshow("window", frm)

    # Exit loop if ESC key is pressed
    if cv2.waitKey(1) == 27:
        cv2.destroyAllWindows()
        cap.release()
        break
