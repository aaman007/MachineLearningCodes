import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
from tensorflow import keras

data = keras.datasets.fashion_mnist

(train_images, train_labels), (test_images, test_labels) = data.load_data()

class_names = ['T-shirt/top', 'Trouser', 'Pullover', 'Dress', 'Coat',
               'Sandal', 'Shirt', 'Sneaker', 'Bag', 'Ankle boot']

train_images = train_images/255.0
test_images = test_images/255.0


model = keras.Sequential([
    keras.layers.Flatten(input_shape=(28,28)),
    keras.layers.Dense(128, activation="relu"),
    keras.layers.Dense(10, activation="softmax")
])

model.compile(optimizer="adam", loss="sparse_categorical_crossentropy", metrics=['accuracy'])

model.fit(train_images, train_labels, epochs=5)

# Single Image Prediction
# prediction = model.predict([ test_images[5] ])

# Multiple Images Prediction

prediction = model.predict(test_images)

for i in range(10):
    plt.grid(False)
    plt.imshow(test_images[i], cmap=plt.cm.binary)
    plt.xlabel('Actual : ' + class_names[test_labels[i]])
    plt.title('Prediction : ' + class_names[np.argmax(prediction[i])])
    plt.show()
