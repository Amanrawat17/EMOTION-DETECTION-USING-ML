import mediapipe as mp
import numpy as np
import cv2

# Open the webcam
cap = cv2.VideoCapture(0)

# Get the name of the data from the user
name = input("Enter the name of the data: ")

# Initialize Mediapipe solutions and drawing utils
holistic = mp.solutions.holistic
hands = mp.solutions.hands
holis = holistic.Holistic()
drawing = mp.solutions.drawing_utils

# Initialize an empty list to store the data
X = []
# Initialize a variable to keep track of the number of data points collected
data_size = 0

# Main loop to capture video and process frames
while True:
    lst = []

    # Read frame from the webcam
    _, frm = cap.read()

    # Flip the frame horizontally for a more intuitive view
    frm = cv2.flip(frm, 1)

    # Process the frame with Mediapipe
    res = holis.process(cv2.cvtColor(frm, cv2.COLOR_BGR2RGB))

    # Extract features from face and hands landmarks
    if res.face_landmarks:
        for i in res.face_landmarks.landmark:
            lst.append(i.x - res.face_landmarks.landmark[1].x)
            lst.append(i.y - res.face_landmarks.landmark[1].y)

        # Extract features from left hand landmarks
        if res.left_hand_landmarks:
            for i in res.left_hand_landmarks.landmark:
                lst.append(i.x - res.left_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.left_hand_landmarks.landmark[8].y)
        else:
            # If no left hand landmarks are detected, fill with zeros
            for _ in range(42):
                lst.append(0.0)

        # Extract features from right hand landmarks
        if res.right_hand_landmarks:
            for i in res.right_hand_landmarks.landmark:
                lst.append(i.x - res.right_hand_landmarks.landmark[8].x)
                lst.append(i.y - res.right_hand_landmarks.landmark[8].y)
        else:
            # If no right hand landmarks are detected, fill with zeros
            for _ in range(42):
                lst.append(0.0)

        # Append the extracted features to the data list
        X.append(lst)
        data_size += 1

    # Draw landmarks on the frame
    drawing.draw_landmarks(frm, res.face_landmarks, holistic.FACEMESH_CONTOURS)
    drawing.draw_landmarks(frm, res.left_hand_landmarks, hands.HAND_CONNECTIONS)
    drawing.draw_landmarks(frm, res.right_hand_landmarks, hands.HAND_CONNECTIONS)

    # Display the number of data points collected on the frame
    cv2.putText(frm, str(data_size), (50, 50), cv2.FONT_HERSHEY_SIMPLEX, 1, (0, 255, 0), 2)

    # Display the frame
    cv2.imshow("window", frm)

    # Check for ESC key press to exit the loop or if the desired number of data points is collected
    if cv2.waitKey(1) == 27 or data_size > 99:
        cv2.destroyAllWindows()
        cap.release()
        break

# Save the collected data as a NumPy array
np.save(f"{name}.npy", np.array(X))
print(np.array(X).shape)  # Print the shape of the collected data
#This code captures video from the webcam, processes each frame using Mediapipe, extracts features from face and hand landmarks, and saves the collected data as a NumPy array with the provided name. The number of data points collected is limited to 100 (99 in this case, as indexing starts from 0).