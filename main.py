import os
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
from sklearn.model_selection import train_test_split
import tensorflow as tf
from tensorflow import keras

with_mask_files = os.listdir("./data/with_mask")
without_mask_files = os.listdir("./data/without_mask")

# 0 --> no mask
# 1 --> mask
labels = [0] * len(without_mask_files) + [1] * len(with_mask_files)

# img = mpimg.imread("./data/with_mask/with_mask_10.jpg")
# plt.imshow(img)
# plt.show()

from PIL import Image

data = []

for file in without_mask_files:
    img = Image.open("./data/without_mask/" + file)
    img = img.resize((128, 128))
    img = img.convert("RGB")
    img = np.array(img)
    data.append(img)

for file in with_mask_files:
    img = Image.open("./data/with_mask/" + file)
    img = img.resize((128, 128))
    img = img.convert("RGB")
    img = np.array(img)
    data.append(img)

X = np.array(data)
y = np.array(labels)


X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# scale data
X_train_scaled = X_train/255
X_test_scaled = X_test/255

model = keras.Sequential()

model.add(keras.layers.Conv2D(32, kernel_size=(3,3), activation="relu", input_shape=(128, 128, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Conv2D(64, kernel_size=(3,3), activation="relu", input_shape=(128, 128, 3)))
model.add(keras.layers.MaxPooling2D(pool_size=(2,2)))

model.add(keras.layers.Flatten())

model.add(keras.layers.Dense(128, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(64, activation='relu'))
model.add(keras.layers.Dropout(0.5))

model.add(keras.layers.Dense(2, activation="sigmoid"))

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["acc"])

model.fit(X_train_scaled, y_train, validation_split=0.1, epochs=10)

loss, accuracy = model.evaluate(X_test_scaled, y_test)
print('Test Accuracy =', accuracy)

# save the model
model.save("mask_model.h5")
print("Model saved")