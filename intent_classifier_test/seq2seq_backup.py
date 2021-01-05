from keras import models
from keras import layers
from keras import optimizers, losses, metrics
from keras import preprocessing
from keras.models import load_model

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import os
import re

from konlpy.tag import Okt

data = pd.read_csv('app\data\intent_data.csv', names=['question','intent'], encoding='utf-8')
data = data[1:]

ques = data.question
intent = data.intent

# 태그 단어
PAD = "<PADDING>"   # 패딩
STA = "<START>"     # 시작
END = "<END>"       # 끝
OOV = "<OOV>"       # 없는 단어(Out of Vocabulary)

# 태그 인덱스
PAD_INDEX = 0
STA_INDEX = 1
END_INDEX = 2
OOV_INDEX = 3

# 데이터 타입
ENCODER_INPUT  = 0
DECODER_INPUT  = 1
DECODER_TARGET = 2

# 한 문장에서 단어 시퀀스의 최대 개수
max_sequences = 30

# 임베딩 벡터 차원
embedding_dim = 100

# LSTM 히든레이어 차원
lstm_hidden_dim = 128

# 정규 표현식 필터
RE_FILTER = re.compile("[.,!?\"':;~()]")



### DATA PREPROCESSING ###

# 형태소 분석
def pos_tag(sentences):
    okt = Okt()
    sentences_pos = []
    for sentence in sentences:
        sentence = re.sub(RE_FILTER, '', sentence)
        sentence = ' '.join(okt.morphs(sentence))
        sentences_pos.append(sentence)

    return sentences_pos

ques = pos_tag(ques)
intent = pos_tag(intent)


# question + intent
sentences = []
sentences.extend(ques)
sentences.extend(intent)

words = []

for sentence in sentences:
    for word in sentence.split():
        words.append(word)

words = [word for word in words if len(word) > 0] # 길이가 0인것 삭제
words = list(set(words)) # 중복된 단어 삭제
words[:0] = [PAD, STA, END, OOV]

# print(len(words)) # 1756

ques = sorted(list(ques))
intent = sorted(list(intent))

# 단어, 인덱스 사전생성
word_to_idx = {word: index for index, word in enumerate(words)}
idx_to_word = {index: word for index, word in enumerate(words)}

# print(dict(list(word_to_idx.items())[:10])) # '맛있죠': 7, 'ne': 8, '엔플라잉': 9
# print(dict(list(idx_to_word.items())[:10])) #  7:'맛있죠', 8:'ne', 9:'엔플라잉'

def convert_txt_to_idx(sentences, voca, type):
    sentences_idx = []

    for sentence in sentences:
        sentence_idx = []

        # 디코더 입력이면 <START> 추가
        if type == DECODER_INPUT:
            sentence_idx.extend([voca[STA]])

        for word in sentence.split():
            if voca.get(word) is not None:
                # 사전에 있는 단어
                sentence_idx.extend([voca[word]])
            else:
                # 사전에 없는 단어면 OOV 추가
                sentence_idx.extend([voca[OOV]])
        
        if type == DECODER_TARGET:
            if len(sentence_idx) >= max_sequences:
                sentence_idx = sentence_idx[:max_sequences-1] + [voca[END]]
            else:
                sentence_idx += [voca[END]]
        else:
            if len(sentence_idx) > max_sequences:
                sentence_idx = sentence_idx[:max_sequences]
        
        # 패딩
        sentence_idx += (max_sequences - len(sentence_idx)) * [voca[PAD]]
        sentences_idx.append(sentence_idx)
    
    return np.asarray(sentences_idx)

x_encoder = convert_txt_to_idx(ques, word_to_idx, ENCODER_INPUT)
x_decoder = convert_txt_to_idx(intent, word_to_idx, DECODER_INPUT)
y_decoder = convert_txt_to_idx(intent, word_to_idx, DECODER_TARGET)

one_hot_data = np.zeros((len(y_decoder), max_sequences, len(words)))

# train시 입력 index, 출력 one-hot
for i, seq in enumerate(y_decoder):
    for j, idx in enumerate(seq):
        one_hot_data[i, j, idx] = 1

y_decoder = one_hot_data


### MODELING ###

## TRAIN MODEL
## ENCODER
encoder_inputs = layers.Input(shape=(None,))
encoder_outputs = layers.Embedding(len(words), embedding_dim)(encoder_inputs)
encoder_outputs, state_h, state_c = layers.LSTM(lstm_hidden_dim, dropout=0.1, recurrent_dropout=0.5, return_state=True)(encoder_outputs)

encoder_states = [state_h, state_c]

## DECODER
decoder_inputs = layers.Input(shape=(None,))
decoder_embedding = layers.Embedding(len(words), embedding_dim)
decoder_outputs = decoder_embedding(decoder_inputs)
decoder_lstm = layers.LSTM(lstm_hidden_dim, dropout=0.1, recurrent_dropout=0.5, return_state=True, return_sequences=True)
decoder_outputs, _, _ = decoder_lstm(decoder_outputs, initial_state=encoder_states)
decoder_dense = layers.Dense(len(words), activation='softmax')
decoder_outputs = decoder_dense(decoder_outputs)

## TRAIN MODEL
model = models.Model([encoder_inputs, decoder_inputs], decoder_outputs)

model.compile(optimizer='rmsprop', loss='categorical_crossentropy', metrics=['accuracy'])

## PREDICT MODEL
## ENCODER
encoder_model = models.Model(encoder_inputs, encoder_states)

## DECODER
decoder_state_input_h = layers.Input(shape=(lstm_hidden_dim,))
decoder_state_input_c = layers.Input(shape=(lstm_hidden_dim,))
decoder_state_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs = decoder_embedding(decoder_inputs)
decoder_outputs, state_h, state_c = decoder_lstm(decoder_outputs, initial_state=decoder_state_inputs)
decoder_states = [state_h, state_c]
decoder_outputs = decoder_dense(decoder_outputs)

decoder_model = models.Model([decoder_inputs] + decoder_state_inputs, [decoder_outputs] + decoder_states)


### TRAIN ###

def convert_idx_to_txt(indexs, voca):
    sentence = ''

    for idx in indexs:
        if idx == END_INDEX:
            break
        if voca.get(idx) is not None:
            sentence += voca[idx]
        else: # 사전에 없으면 OOV 단어 추가
            sentence.extend([voca[OOV_INDEX]])
        sentence += ' '

    return sentence

for epoch in range(1):
    print('Total Epoch : ', epoch+1)
    history = model.fit([x_encoder, x_decoder], y_decoder, epochs=100, batch_size=32) # verbose=0

    print('정확도(acc) : ', history.history['accuracy'][-1])
    print('손실(loss) : ', history.history['loss'][-1])

    # input_encoder = x_encoder[2].reshape(1, x_encoder[2].shape[0])
    # input_decoder = x_decoder[2].reshape(1, x_decoder[2].shape[0])
    # results = model.predict([input_encoder, input_decoder])

    # # result : one-hot -> index / 1축을 기준으로 가장 높은 값의 위치
    # indexs = np.argmax(results[0], 1)

    # # index -> sentence
    # sentence = convert_idx_to_txt(indexs, idx_to_word)
    # print(ques[2])
    # print(sentence)


### PREDICT ###

def predict_input(sentence):
    sentences = []
    sentences.append(sentence)
    sentences = pos_tag(sentences)
    input_seq = convert_txt_to_idx(sentences, word_to_idx, ENCODER_INPUT)

    return input_seq

def generate_text(input_seq):
    states = encoder_model.predict(input_seq)
    target_seq = np.zeros((1, 1))
    target_seq[0,0] = STA_INDEX
    indexs = []

    while 1:
        decoder_outputs, state_h, state_c = decoder_model.predict([target_seq] + states)

        # result : one-hot -> index
        index = np.argmax(decoder_outputs[0, 0, :])
        indexs.append(index)

        if index ==END_INDEX or len(indexs) >= max_sequences:
            break

        target_seq = np.zeros((1, 1))
        target_seq[0, 0] = index

        states = [state_h, state_c]
    
    sentence = convert_idx_to_txt(indexs, idx_to_word)

    return sentence

for i in [3, 6, 8, 33, 24, 101, 400, 22, 700, 1001, 1200, 55, 67, 1080]:
    print(ques[i])
    input_seq = predict_input(ques[i])
    intent = generate_text(input_seq)
    print(intent)