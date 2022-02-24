import tensorflow as tf

from RNN_model import get_model, int2label
from tensorflow.keras.preprocessing.sequence import pad_sequences

import pickle
import numpy as np

SEQ_LEN = 100

tokenizer = pickle.load(open("results/tokenizer.pickle", "rb"))

model = get_model(tokenizer, 128)
model.load_weights("results/spam_classifier_0.06.h5")


def get_predictions(text):
    sequence = tokenizer.texts_to_sequences([text])
    sequence = pad_sequences(sequence, maxlen=SEQ_LEN)
    prediction = model.predict(sequence)[0]
    return int2label[np.argmax(prediction)]


Accurate, Total = 0, 0
with open("data/Mail_Test_Dataset") as f:
    for line in f:
        Total += 1
        split = line.split()
        result = get_predictions(split[1])
        if result == split[0]:
            Accurate += 1
    print((Accurate / Total) * 100, "% / 100%")
