import numpy
import keras
from keras.datasets import reuters
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import LSTM
from keras.layers.embeddings import Embedding
from keras.preprocessing import sequence

top_words = 1000
(X_train, y_train), (X_test, y_test) = reuters.load_data(num_words=top_words)

max_review_length = 200
X_train = sequence.pad_sequences(X_train, maxlen=max_review_length)
X_test = sequence.pad_sequences(X_test, maxlen=max_review_length)
print X_train[0]
# y_train = keras.utils.to_categorical(y_train, 46)
# y_test = keras.utils.to_categorical(y_test, 46)

# # create the model
# embedding_vecor_length = 100
# model = Sequential()
# model.add(Embedding(top_words, embedding_vecor_length, input_length=max_review_length))
# model.add(LSTM(100))
# model.add(Dense(46, activation='sigmoid'))
# model.compile(loss='binary_crossentropy', optimizer='adam', metrics=['accuracy'])
# print(model.summary())
# model.fit(X_train, y_train, validation_data=(X_test, y_test), epochs=5, batch_size=32)

# # Final evaluation of the model
# scores = model.evaluate(X_test, y_test, verbose=0)
# print("Accuracy: %.2f%%" % (scores[1]*100))