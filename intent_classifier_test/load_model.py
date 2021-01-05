from keras.models import load_model
from keras.preprocessing.sequence import pad_sequences
from konlpy.tag import Okt

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import re

data = pd.read_csv('intent_data.csv', names=['question','intent'], encoding='utf-8')
data = data[1:]

ques = data.question
intent = data.intent

# 태그 단어
PAD = "<PADDING>"   # 패딩
OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

# 태그 인덱스
PAD_INDEX = 0
OOV_INDEX = 1

max_sequences = 30

# 정규 표현식 필터
RE_FILTER = re.compile("[.,!?\"':;~()]")

# 형태소 분석
def pos_tag(sentences):
    okt = Okt()
    sentences_pos = []
    for sentence in sentences:
        sentence = re.sub(RE_FILTER, '', sentence)
        sentence = ' '.join(okt.morphs(sentence))
        sentences_pos.append(sentence)

    return sentences_pos

raw_ques = pos_tag(ques)
raw_intent = pos_tag(intent)

ques = pos_tag(ques)
intent = pos_tag(intent)

# print(ques) # ['오랜 만이 야', '잘 있었어',
# print(intent) # ['welcome', 'welcome', 'welcome',

ques_words = []
for sentence in ques:
    for word in sentence.split():
        ques_words.append(word)

# print(ques_words) # ['오랜', '만이', '야', '잘', '있었어',

ques_words = [word for word in ques_words if len(word) > 0 ] # 길이가 0 삭제
ques_words = list(set(ques_words)) # 중복 단어 삭제
intent = list(set(intent))
ques_words[:0] = [PAD, OOV]
intent[:0] = [PAD, OOV]

# print(intent)
# ['weather', 'news', 'jobdam', 'depress', 'weight', 'welcome', 'alcohol', 'health', 'blood', 'diabetes', 'time', 'religion', 'friend', 'family', 'music', 'no', 'yes', 'wiki', 'dust', 'hobby', 'food', 'smoke', 'wisesaying', 'person',
# 'translate', 'issue', 'exercise', 'end', 'date']

ques = ques_words

ques_size = len(ques)+1 # embedding레이어에서 input_dim의 수
intent_size = len(intent)+1

# print(ques_size) # 1725
# print(intent_size) # 31]

ques = sorted(list(ques))
intent = sorted(list(intent))

# 단어, 인덱스 사전생성
ques_to_idx = {word: index for index, word in enumerate(ques)}
idx_to_ques = {index: word for index, word in enumerate(ques)}
intent_to_idx = {word: index for index, word in enumerate(intent)}
idx_to_intent = {index: word for index, word in enumerate(intent)}


model = load_model('cnn1d_v1.h5')


def convert_txt_to_idx(sentences, voca):
    sentences_idx = []

    for sentence in sentences:
        sentence_idx = []

        for word in sentence.split():
            if voca.get(word) is not None:
                # 사전에 있는 단어
                sentence_idx.extend([voca[word]])
            else:
                # 사전에 없는 단어면 OOV 추가
                sentence_idx.extend([voca[OOV]])
        
        if len(sentence_idx) > max_sequences:
            sentence_idx = sentence_idx[:max_sequences]
        
        # # 패딩
        # sentence_idx += (max_sequences - len(sentence_idx)) * [voca[PAD]]
        sentences_idx.append(sentence_idx)
    
    return np.asarray(sentences_idx)


def predict_input(sentence):
    sentences = []
    sentences.append(sentence)
    sentences = pos_tag(sentences)
    input_seq = convert_txt_to_idx(sentences, ques_to_idx)
    input_seq = pad_sequences(input_seq, maxlen=max_sequences)
    # input_seq = input_seq.reshape((input_seq.shape[0], input_seq.shape[1],1))

    return input_seq



def convert_idx_to_txt(indexs, voca):
    indexs = np.argmax(indexs, axis=-1)
    sentence = ''

    for idx in indexs:
        if voca.get(idx) is not None:
            sentence += voca[idx]
        else: # 사전에 없으면 OOV 단어 추가
            sentence.extend([voca[OOV_INDEX]])
        sentence += ' '

    return sentence


def get_intent(speech):
    input_seq = predict_input(speech)
    result_idx = model.predict(input_seq)
    intent = convert_idx_to_txt(result_idx, idx_to_intent)

    return intent


for i in [3, 6, 8, 33, 24, 101, 400, 22, 700, 1001, 1200, 55, 67, 1080, 540, 541, 542, 546, 547]:
    input_seq = predict_input(data.question[i])
    print(data.question[i])
    result_idx = model.predict(input_seq)
    intent = convert_idx_to_txt(result_idx, idx_to_intent)
    print(intent)


