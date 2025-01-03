# EMOTION-DETECTION-USING-OPEN CV
# Emotion Recognition and Song Recommendation

This project is a real-time emotion recognition system that recommends songs based on the user's mood. It leverages MediaPipe for emotion detection and a machine learning model for song recommendations, integrated into a web application using Streamlit.

## Table of Contents

- [Project Overview](#project-overview)
- [Technologies Used](#technologies-used)
- [Installation](#installation)
- [Usage](#usage)
- [Model Training](#model-training)
- [How It Works](#how-it-works)
- [Contributing](#contributing)


## Project Overview

The application uses a webcam to detect emotions based on facial expressions and hand gestures. Depending on the detected emotion, it recommends songs by a specified artist or in a particular language.

## Technologies Used

- Python
- Streamlit
- OpenCV
- Mediapipe
- TensorFlow
- NumPy

## Installation

To run this project, ensure you have Python installed, then follow these steps:

1. Clone the repository:
   ```bash
   git clone https://github.com/emotion song.git
   Usage
Start the Streamlit server:

````
streamlit run app.py
````
Open the provided URL in your web browser.

Allow camera access when prompted.

Input the desired language and singer's name.

Click the "Recommend me songs" button to get song recommendations based on your detected emotion.

Model Training
To train your model, use a dataset that includes features representing emotions and corresponding labels. The model should be saved as model.h5, and labels should be stored in a labels.npy file for predictions.

How It Works
Webcam Feed: The application captures video from your webcam.
Emotion Detection: It processes each frame to detect facial landmarks and hand gestures using MediaPipe.
Prediction: The processed landmarks are fed into a pre-trained model to predict the user's emotion.
Song Recommendation: Based on the predicted emotion, the application recommends songs related to the specified language and singer.
Contributing
Contributions are welcome! If you would like to contribute to this project, please create a pull request or open an issue.
   
