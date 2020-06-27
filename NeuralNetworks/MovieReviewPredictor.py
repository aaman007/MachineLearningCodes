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


model = keras.models.load_model("model.h5")

while True:
    review = input("Enter Your Review or enter Quit to exit\n")
    if review.lower() == "quit":
        break
    for char in '.,():':
        review = review.replace(char, "")
    review = review.strip().split(" ")
    review = encode_review(review)
    review = keras.preprocessing.sequence.pad_sequences([review], value=word_index["<PAD>"], padding="post", maxlen=250)
    predict = model.predict(review)
    print(predict[0])
    if predict[0] >= 0.5:
        print("Positive Review")
    else:
        print("Negative Review")
