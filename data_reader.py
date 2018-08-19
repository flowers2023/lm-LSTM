# coding: utf-8

# In[16]:


import numpy as np

np.random.seed(42)
import tensorflow as tf

tf.set_random_seed(42)

"""判断一个unicode是否是汉字"""
def is_chinese(uchar):
    if uchar >= u'\u4e00' and uchar <= u'\u9fa5':
        return True
    else:
        return False


_word_set = set(open('./data/common-words.txt').read())
_word_set.discard('\n')
words = sorted(list(_word_set))


def get_word_index():
    word_indices = dict((c, i) for i, c in enumerate(words))
    return word_indices


def get_index_word():
    indices_word = dict((i, c) for i, c in enumerate(words))
    return indices_word


def get_corpus_list(line, max_seq_len):
    s_len = len(line)
    list = []
    for i in range(0, s_len - 1):
        sub_sentence = line[i:i + max_seq_len]
        list.append(_get_text_and_label(sub_sentence))

    return list


def _get_text_and_label(text):
    return (text[0:len(text) - 1], text[-1:])
