#!/usr/bin/python
# encoding=utf-8

#########################################################################
# Function: 
# Author: DRUNK
# mail: shuangfu.zhang@xiaoi.com
# Created Time: Mon 30 Jul 2018 04:24:54 PM
#########################################################################

def get_corpus_list(line, max_seq_len):
    s_len = len(line)
    list = []
    for i in range(0, s_len - 1):
        sub_sentence = line[i:i + max_seq_len]
        list.append(get_text_and_label(sub_sentence))

    return list

def get_text_and_label(text):
    return (text[0:len(text) - 1], text[-1:])


list = get_corpus_list('123456789', 5)
print(list)
