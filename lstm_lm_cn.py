# coding: utf-8

# In[16]:


import numpy as np
np.random.seed(42)
import tensorflow as tf
tf.set_random_seed(42)
import matplotlib.pyplot as plt
import pickle
import sys
import heapq
import seaborn as sns
from pylab import rcParams

sns.set(style='whitegrid',palette='muted',font_scale=1.5)
rcParams['figure.figsize']=12, 5
step = 3
SEQUENCE_LENGTH = 10
lr=0.001
epochs=20
batch_size=128
validation_split=0.05


# In[17]:


def memery_size(name,obj):
    size = sys.getsizeof(obj)
    kb = float('%.2f'% (size/1024))
    mb = float('%.2f'% (size/1024/1024))
    gb = float('%.2f'% (size/1024/1024/1024))
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


path = './data/news.small.txt'
text = open(path).read().lower()
print('corpus length:',len(text))
memery_size('raw text',text)


# In[19]:


chars = sorted(list(set(text)))
print(type(chars))
memery_size('words',chars)


# In[20]:


char_indices = dict((c,i) for i, c in enumerate(chars))
memery_size('char indeces',char_indices)
indices_char = dict((i,c) for i,c in enumerate(chars))
memery_size('indeces char',indices_char)


# In[21]:


sentences = []
next_chars = []
for i in range(0,len(text)-SEQUENCE_LENGTH, step):
    sentences.append(text[i: i + SEQUENCE_LENGTH])
    next_chars.append(text[i + SEQUENCE_LENGTH])
print(f'num training examples:{len(sentences)}')
memery_size('sentences',sentences)
memery_size('nect chars',next_chars)


# In[22]:


X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
Y = np.zeros((len(sentences), len(chars)), dtype=np.bool)
for i, sentence in enumerate(sentences):
    for t, char in enumerate(sentence):
        if char in char_indices:
            X[i,t,char_indices[char]] = 1
    if next_chars[i] in char_indices:
        Y[i,char_indices[next_chars[i]]] = 1
memery_size('X',X)
memery_size('Y',Y)


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


model = tf.keras.Sequential()
model.add(tf.keras.layers.LSTM(units=128,input_shape=(SEQUENCE_LENGTH, len(chars))))
model.add(tf.keras.layers.Dense(len(chars)))
model.add(tf.keras.layers.Activation('softmax'))


# In[30]:


optimizer = tf.keras.optimizers.RMSprop(lr=lr)
model.compile(loss='categorical_crossentropy', optimizer = optimizer, metrics =['accuracy'])
history = model.fit(X,Y,validation_split=validation_split, batch_size=batch_size,epochs=epochs,shuffle=True).history


# In[31]:


model.save('./model/lm_lstm_cn.h5')
pickle.dump(history,open('history.p','wb'))


# In[32]:


plt.plot(history['acc'])
plt.plot(history['val_acc'])
plt.title('model accuracy')
plt.ylabel('accuracy')
plt.xlabel('epoch')
plt.legend(['train','test'],loc ='upper left')


# In[33]:


plt.plot(history['loss'])
plt.plot(history['val_loss'])
plt.title('model loss')
plt.ylabel('loss')
plt.xlabel('epoch')
plt.legend(['train','test'],loc='upper left')


# In[34]:


def prepare_input(text):
    x = np.zeros((1, SEQUENCE_LENGTH,len(chars)))
    #X = np.zeros((len(sentences), SEQUENCE_LENGTH, len(chars)), dtype=np.bool)
    
    for t, char in enumerate(text):
        if char in char_indices:
            x[0, t, char_indices[char]] = 1
    
    return x
prepare_input('LSTM'.lower())


# In[35]:


def sample(preds, top_n = 3):
    preds = np.asarray(preds).astype('float64')
    preds = np.log(preds)
    exp_preds = np.exp(preds)
    preds = exp_preds / np.sum(exp_preds)
    
    return heapq.nlargest(top_n, range(len(preds)), preds.take)


# In[36]:


def predict_completion(text):
    original_text = text
    generated = text
    completion = ''
    while True:
        x = prepare_input(text)
        preds = model.predict(x,verbose=0)[0]
        next_index = sample(preds,top_n =1)[0]
        next_char = indices_char[next_index]
        
        text = text[1:] + next_char
        completion += next_char
        
        if len(original_text + completion) +2 > len(original_text) and next_char == ' ':
            return completion


# In[ ]:


def predict_completions(text, n =3):
    x = prepare_input(text)
    preds = model.predict(x,verbose=0)[0]
    next_indices = sample(preds, n)
    return [indices_char[idx] + predict_completion(text[1:] + indices_char[idx]) for idx in next_indices]


# In[ ]:


quotes = [
    "造成它",
    "文化艺",
    "就意味",
    "天使的面"
]
for q in quotes:
    seq = q[:10].lower()
    print(seq)
    print(predict_completions(seq, 5))
    print()
