# coding: utf-8

# In[16]:


import numpy as np

np.random.seed(42)
import tensorflow as tf

tf.set_random_seed(42)
import matplotlib.pyplot as plt
import pickle
import sys
import re
import data_reader
from pylab import rcParams

rcParams['figure.figsize'] = 12, 5
step = 3
SEQUENCE_LENGTH = 10
lr = 0.001
epochs = 20
batch_size = 128
validation_split = 0.05

# In[17]:
def memery_size(name, obj):
    size = sys.getsizeof(obj)
    kb = float('%.2f' % (size / 1024))
    mb = float('%.2f' % (size / 1024 / 1024))
    gb = float('%.2f' % (size / 1024 / 1024 / 1024))
    print()
    print(f'----------------[{name}]--------------')
    print(f'size(byte):{size}')
    print(f'size(KB):{kb}')
    print(f'size(MB):{mb}')
    print(f'size(GB):{gb}')
    print()


"""判断一个unicode是否是汉字"""
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

# In[18]:
file_type = '.test'
path = './data/news' + file_type + '.txt'
text = open(path).read().lower()
# text = re.sub('(\\s+)', '', text)
print('corpus length:', len(text))
memery_size('raw text', text)

# In[19]:

sentences = []
next_chars = []
for i in range(0, len(text) - SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])

print(f'num training examples:{len(sentences)}')
memery_size('sentences', sentences)
memery_size('nect chars', next_chars)

# In[22]:

chars = data_reader.words
char_indices = data_reader.get_word_index()
indices_char = data_reader.get_index_word()

print(len(chars))
X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
Y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        if char in char_indices:
            X[i, t, char_indices[char]] = 1
    if next_chars[i] in char_indices:
        Y[i, char_indices[next_chars[i]]] = 1
memery_size('X', X)
memery_size('Y', Y)

# In[23]:

sentences[100]

# In[24]:

next_chars[100]

# In[25]:

X[0][0]

# In[26]:

Y[0]

# In[27]:

X.shape

# In[28]:

Y.shape

# In[29]:

model = tf.keras.models.Sequential()
model.add(tf.keras.layers.LSTM(units=128, input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(tf.keras.layers.Dense(len(chars)))
model.add(tf.keras.layers.Activation('softmax'))

# In[30]:

optimizer = tf.keras.optimizers.RMSprop(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer=optimizer, metrics=['accuracy'])
history = model.fit(X, Y, validation_split=validation_split, batch_size=batch_size, epochs=epochs, shuffle=True).history

# In[31]:
model.save('./model/lm_lstm' + file_type + '.h5')
pickle.dump(history, open('./model/history' + file_type + '.p', 'wb'))

print(model.input_shape)
print('model train sucessful')
# In[32]:

# plt.plot(history['acc'])
# plt.plot(history['val_acc'])
# plt.title('model accuracy')
# plt.ylabel('accuracy')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
#
## In[33]:
#
#
# plt.plot(history['loss'])
# plt.plot(history['val_loss'])
# plt.title('model loss')
# plt.ylabel('loss')
# plt.xlabel('epoch')
# plt.legend(['train', 'test'], loc='upper left')
