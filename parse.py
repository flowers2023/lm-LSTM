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
import re
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

"""判断一个unicode是否是汉字"""
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False

# In[18]:


dic = set(open('./data/common-words.txt').read())
file_type='.test'
path = './data/news'+file_type+'.txt'
text = open(path).read().lower()
replaced_char = map(lambda c : c if c in dic else 'N',list(text))
replaced_list = list(replaced_char)
with open('./data/news'+file_type+'.parsed.txt','w') as f:
    for i,c in enumerate(replaced_list):
        if i>0 and c == 'N' and c == replaced_list[i-1]:
            continue
        f.write(c)
