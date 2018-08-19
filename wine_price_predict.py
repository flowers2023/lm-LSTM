#!/bin/python
import numpy as np
import pandas as pd
import tensorflow as tf

from sklearn.preprocessing import LabelEncoder
from tensorflow import keras

layers = keras.layers
# print tf version
print("You have TensorFlow version", tf.__version__)

data = pd.read_csv('./data/wine_data_test.csv')
print('----data shape----')
print(data.shape)
print(data.count())

train_size = int(len(data) * .8)

# train features
description_train = data['description'][:train_size]
print(data['description'][:10])
variety_train = data['variety'][:train_size]
print('-----variety-----')
print(variety_train[:10])

# train labels
labels_train = data['price'][:train_size]

# Test features
description_test = data['description'][train_size:]
variety_test = data['variety'][train_size:]

# Test labels
labels_test = data['price'][train_size:]

# feature 1: wine description
vocab_size = 12000
tokenize = keras.preprocessing.text.Tokenizer(num_words=vocab_size, char_level=False)
tokenize.fit_on_texts(data['description'])  # only fit on train

description_bow_train = tokenize.texts_to_matrix(description_train)
print('---------descrition bow train matrix------------')
print(description_bow_train[:10])
description_bow_test = tokenize.texts_to_matrix(description_test)
print('---------descrition bow test matrix------------')
print(description_bow_test[:10])

# feature 2: wine classes
# use sklearn utility to convert label strings to numbered index
encoder = LabelEncoder()
encoder.fit(data['variety'])
variety_train = encoder.transform(variety_train)
print('--------variety train encoder----------')
print(variety_train[:10])
variety_test = encoder.transform(variety_test)
print('---------------variety test------------')
print(variety_test[:10])
num_classes = np.max(variety_train) + 1
print('--------num classes---------')
print(num_classes)

# convert labels to on hot
variety_train = keras.utils.to_categorical(variety_train, num_classes=num_classes)
variety_test = keras.utils.to_categorical(variety_test, num_classes=num_classes)

bow_inputs = layers.Input(shape=(vocab_size,))
variety_inputs = layers.Input(shape=(num_classes,))
merged_layer = layers.concatenate([bow_inputs, variety_inputs])
merged_layer = layers.Dense(256, activation='relu')(merged_layer)
predictions = layers.Dense(1)(merged_layer)
# wide model
wide_model = keras.Model(inputs=[bow_inputs, variety_inputs], outputs=predictions)
keras.Model()
print(wide_model.input_shape)

# compile model
wide_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# depe model
train_embed = tokenize.texts_to_sequences(description_train)
test_embed = tokenize.texts_to_sequences(description_test)

max_seq_length = 170
train_embed = keras.preprocessing.sequence.pad_sequences(train_embed, maxlen=max_seq_length)
test_embed = keras.preprocessing.sequence.pad_sequences(test_embed, maxlen=max_seq_length)

deep_inputs = layers.Input(shape=(max_seq_length,))
embedding = layers.Embedding(vocab_size, 8, input_length=max_seq_length)(deep_inputs)
embedding = layers.Flatten()(embedding)

embed_out = layers.Dense(1, activation='linear')(embedding)
deep_model = keras.Model(inputs=deep_inputs, outputs=embed_out)
deep_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Wide and Deep
merged_out = layers.concatenate([wide_model.output, deep_model.output])
merged_out = layers.Dense(1)(merged_out)
combined_model = keras.Model(wide_model.input + [deep_model.input], merged_out)
combined_model.compile(loss='mse', optimizer='adam', metrics=['accuracy'])

# Training
combined_model.fit([description_bow_train, variety_train] + [train_embed], labels_train, epochs=10, batch_size=128)

# Evaluation
combined_model.evaluate([description_bow_test, variety_test] + [test_embed], labels_test, batch_size=128)

predictions = combined_model.predict([description_bow_test, variety_test] + [test_embed])

print(predictions.shape)
# for i in range(5):
#     val = predictions[i]
#     print(description_test[i])
#     print(val[0], 'Actual:', labels_test.iloc[i], '\n')