import tensorflow as tf
from tensorflow import keras
import numpy as np

data = keras.datasets.imdb

(train_data, train_labels), (test_data, test_labels) = data.load_data(num_words=85000)

word_index = data.get_word_index()

word_index = {key: (value+3) for key, value in word_index.items()}
word_index["<PAD>"] = 0
word_index["<START>"] = 1
word_index["<UNK>"] = 2
word_index["<UNUSED>"] = 3

reverse_word_index = dict([(value, key) for (key, value) in word_index.items()])

train_data = keras.preprocessing.sequence.pad_sequences(train_data, value=word_index["<PAD>"], padding="post", maxlen=250)
test_data = keras.preprocessing.sequence.pad_sequences(test_data, value=word_index["<PAD>"], padding="post", maxlen=250)


def decode_review(text):
    return " ".join([reverse_word_index.get(i, "?") for i in text])


def encode_review(words):
    encoded = [1]
    for word in words:
        if word.lower() in word_index:
            encoded.append(word_index[word.lower()])
        else:
            encoded.append(2)
    return encoded


'''
# Creating Model
model = keras.Sequential()
model.add(keras.layers.Embedding(85000, 16))
model.add(keras.layers.GlobalAveragePooling1D())
model.add(keras.layers.Dense(16, activation="relu"))
model.add(keras.layers.Dense(1, activation="sigmoid"))

model.summary()

model.compile(optimizer="adam", loss="binary_crossentropy", metrics=["accuracy"])

x_val = train_data[:10000]
y_val = train_labels[:10000]

x_train = train_data[10000:]
y_train = train_labels[10000:]

model.fit(x_train, y_train, epochs=30, batch_size=512, validation_data=(x_val, y_val), verbose=1)
results = model.evaluate(test_data, test_labels)

model.save("model.h5")
'''

model = keras.models.load_model("model.h5")

# print(model.evaluate(test_data, test_labels))

with open("review.txt") as file:
    for review in file.readlines():
        ori_review = review
        #for char in '.,():':
            #review = review.replace(char, "")

        review = review.strip().split(" ")
        review = encode_review(review)
        review = keras.preprocessing.sequence.pad_sequences([review], value=word_index["<PAD>"], padding="post",
                                                            maxlen=250)
        predict = model.predict(review)
        print("Review : ", ori_review)
        print("Encoded Review : ", review)
        print("Prediction : ", predict[0])
