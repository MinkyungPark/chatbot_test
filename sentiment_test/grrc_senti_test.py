import pandas as pd
import numpy as np
from nltk import word_tokenize, pos_tag
from gensim.models import word2vec

data1 = pd.read_csv('C:\\Projects_Python\\chatbot_test\\sentiment_test\\data\\train_intent.csv', names=['question','intent'], encoding='utf-8')
data2 = pd.read_csv('C:\\Projects_Python\\chatbot_test\\sentiment_test\\data\\intent_grrc.csv', names=['question','intent'], encoding='utf-8')
data1 = data1[1:]
data2 = data2[1:]

data = pd.concat([data1, data2], axis=0, ignore_index=True, names=['question','intent'])
# data.info() 4086 object

que_list = list(data.question)
# print(rdw)

from konlpy.tag import Okt
pos_tagger = Okt()

def tokenize(list):
    new_list = []
    for sentence in list: 
        sentence = pos_tagger.pos(sentence, norm=True, stem=True)
        new_list.append(sentence)
        # for word in sentence:
        #     new_list.append(word)
    return new_list

toke_list = tokenize(que_list)
print(toke_list)