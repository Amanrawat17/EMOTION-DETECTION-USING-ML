import os
import numpy as np
import cv2
from tensorflow.keras.utils import to_categorical
from keras.layers import Input, Dense
from keras.models import Model

# Initialize variables
is_init = False
size = -1
label = []
dictionary = {}
c = 0

# Iterate over files in the current directory
for i in os.listdir():
    # Check if the file is a .npy file and not named 'labels'
    if i.split(".")[-1] == "npy" and not(i.split(".")[0] == "labels"):  
        if not is_init:
            # Load data from the first .npy file
            is_init = True 
            X = np.load(i)
            size = X.shape[0]
            # Create labels for the first data file
            y = np.array([i.split('.')[0]] * size).reshape(-1, 1)
        else:
            # Concatenate data from subsequent .npy files
            X = np.concatenate((X, np.load(i)))
            # Create labels for subsequent data files
            y = np.concatenate((y, np.array([i.split('.')[0]] * size).reshape(-1, 1)))
        
        # Add filename to label list
        label.append(i.split('.')[0])
        # Create dictionary mapping filename to index
        dictionary[i.split('.')[0]] = c  
        c += 1

# Encode labels using one-hot encoding
for i in range(y.shape[0]):
    y[i, 0] = dictionary[y[i, 0]]
y = to_categorical(y)

# Shuffle data and labels
X_new = X.copy()
y_new = y.copy()
counter = 0 
cnt = np.arange(X.shape[0])
np.random.shuffle(cnt)
for i in cnt: 
    X_new[counter] = X[i]
    y_new[counter] = y[i]
    counter += 1

# Define model architecture
ip = Input(shape=(X.shape[1]))
m = Dense(512, activation="relu")(ip)
m = Dense(256, activation="relu")(m)
op = Dense(y.shape[1], activation="softmax")(m) 
model = Model(inputs=ip, outputs=op)

# Compile the model
model.compile(optimizer='rmsprop', loss="categorical_crossentropy", metrics=['acc'])

# Train the model
model.fit(X, y, epochs=50)

# Save the trained model and labels
model.save("model.h5")
np.save("labels.npy", np.array(label))

model.save("model.h5")
np.save("labels.npy", np.array(label))