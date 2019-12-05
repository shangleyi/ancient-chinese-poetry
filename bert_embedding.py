#! /usr/bin/env python3
# -*- coding:utf-8 -*-
# https://github.com/brightmart/albert_zh

import bert
from bert import run_classifier
from bert import optimization
from bert import tokenization
from utils import CHAR_VEC_DIM

from char_dict import CharDict
from poems import Poems

def gen_bert_embedding():
	embedding = uniform(-1.0, 1.0, [len(char_dict), CHAR_VEC_DIM])
	model =
	char_dict = CharDict()
    for i, ch in enumerate(char_dict):
        if ch in model.wv:
            # print(ch)
            embedding[i, :] = model.wv[ch]
	return embedding