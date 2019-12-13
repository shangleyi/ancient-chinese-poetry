#! /usr/bin/env python3
# -*- coding:utf-8 -*-
# embedding from: https://github.com/Embedding/Chinese-Word-Vectors
# Author: Devin Zuo(baseline), Leyi Shang(modification line 35-75)

from char_dict import CharDict
from gensim import models
from numpy.random import uniform
from paths import char2vec_path, check_uptodate, dict_dir
from poems import Poems
from singleton import Singleton
from utils import CHAR_VEC_DIM, is_cn_char
import numpy as np
import os
import re

def _gen_char2vec():
    '''
    Simple example to understand word2vec:
    sentences = [['first', 'sentence'], ['second', 'sentence']]
    # train word2vec on the two sentences
    model = models.Word2Vec(sentences, min_count=1)
    print(model.wv['first'])
    exit()
    '''
    print("Generating char2vec model ...")
    char_dict = CharDict()

    '''
    # poems = Poems()
    '''
    # change this line of code to use another model
    # model = models.Word2Vec(poems, size = CHAR_VEC_DIM, min_count = 5)
    embedding = uniform(-1.0, 1.0, [len(char_dict), CHAR_VEC_DIM])
    vocab_count = 0
    vocab_array = []
    vocab_list = []
    with open(os.path.join(dict_dir, 'sgns.literature.char'), 'r',  encoding='utf-8') as fin:
        split = re.split(' |\n',fin.read())
        # length of a word
        word_length = int(split[1])
        # position 2, 304, 606, 908
        i = 2
        while i < len(split):
            if is_cn_char(split[i]):
                word_representation = []
                vocab_list.append(split[i])
                for j in range(word_length):
                    word_representation.append(split[i+1+j])
                vocab_array.append(word_representation)
            #get to the next character
            i += word_length + 2

    count = 0
    for i, ch in enumerate(char_dict):
        try:
            index = vocab_list.index(ch)
            embedding[i, :] = vocab_array[index]
            # embedding[i, 300:] = [0] * (CHAR_VEC_DIM - 300)
            count += 1
            print("Processing " + ch)
        except:
            continue

    '''
    # the for loop was used by the author
    for i, ch in enumerate(char_dict):
        if ch in model.wv:
            # The print statement here illustrates that only 花 is present in the model
            # print(ch)
            # len(model.wv[ch]) = 512
            embedding[i, :] = model.wv[ch]
    '''
    print("Processed " + str(count) + " words")
    np.save(char2vec_path, embedding)
    # np.save(char2vec_path, one_hot)

class Char2Vec(Singleton):

    def __init__(self):
        if not check_uptodate(char2vec_path):
            _gen_char2vec()
        self.embedding = np.load(char2vec_path)
        self.char_dict = CharDict()

    def get_embedding(self):
        return self.embedding

    def get_vect(self, ch):
        return self.embedding[self.char_dict.char2int(ch)]

    def get_vects(self, text):
        return np.stack(map(self.get_vect, text)) if len(text) > 0 \
                else np.reshape(np.array([[]]), [0, CHAR_VEC_DIM])


# For testing purpose.
if __name__ == '__main__':
    char2vec = Char2Vec()

