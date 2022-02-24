import tensorflow as tf

gpus = tf.config.experimental.list_physical_devices('GPU')
if gpus:
    tf.config.experimental.set_memory_growth(gpus[0], enable=True)

from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical
from tensorflow.keras.callbacks import ModelCheckpoint, TensorBoard
from sklearn.model_selection import train_test_split
import time
import numpy as np
import pickle

from RNN_model import get_model, SEQUENCE_LENGTH, TEST_SIZE
from RNN_model import BATCH_SIZE, EPOCHS, label2int


def load_data():
    texts, labels = [], []
    with open("data/Mail_Dataset") as f:
        for line in f:
            split = line.split()
            labels.append(split[0].strip())
            texts.append(' '.join(split[1:]).strip())
    return texts, labels


X, y = load_data()

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X)
pickle.dump(tokenizer, open("results/tokenizer.pickle", "wb"))

X = tokenizer.texts_to_sequences(X)
X = np.array(X)
y = np.array(y)
X = pad_sequences(X, maxlen=SEQUENCE_LENGTH)

y = [label2int[label] for label in y]
y = to_categorical(y)

X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=TEST_SIZE, random_state=7)

model = get_model(tokenizer=tokenizer, lstm_units=128)

model_checkpoint = ModelCheckpoint("results/spam_classifier_{val_loss:.2f}.h5", save_best_only=True)

tensorboard = TensorBoard(f"logs/spam_classifier_{time.time()}")

model.fit(X_train, y_train, validation_data=(X_test, y_test),
          batch_size=BATCH_SIZE, epochs=EPOCHS,
          callbacks=[tensorboard, model_checkpoint],
          verbose=1)

result = model.evaluate(X_test, y_test)



