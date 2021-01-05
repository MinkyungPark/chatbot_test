
from keras.models import load_model
from keras.callbacks import EarlyStopping

import numpy as np
import pandas as pd
import tensorflow as tf
import matplotlib.pyplot as plt
import os
import re

seed = 0
np.random.seed(seed)

from konlpy.tag import Okt

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

ques_size = len(ques)+1
intent_size = len(intent)+1

# print(ques_size) # 1725
# print(intent_size) # 31

ques = sorted(list(ques))
intent = sorted(list(intent))

# 단어, 인덱스 사전생성
ques_to_idx = {word: index for index, word in enumerate(ques)}
idx_to_ques = {index: word for index, word in enumerate(ques)}
intent_to_idx = {word: index for index, word in enumerate(intent)}
idx_to_intent = {index: word for index, word in enumerate(intent)}


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

x = convert_txt_to_idx(raw_ques, ques_to_idx)
y = convert_txt_to_idx(raw_intent, intent_to_idx)

# print(x) # list([1138, 486, 355]) 오랜 만이 야 list([65, 808]) 잘 있었어
# print(y) # [[11][11]....]

# 입력은 패딩 출력은 원핫인코딩
from keras.preprocessing.sequence import pad_sequences
from keras.utils import to_categorical

x = pad_sequences(x, maxlen=max_sequences)
y = to_categorical(y)

# print(x.shape) # (4085, 30)
# print(y.shape) # (4085, 31)

x = x.reshape((x.shape[0], x.shape[1], 1))

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x, y, random_state=66, test_size=0.2)


from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM

model = Sequential()
model.add(LSTM(50, activation = 'relu', input_shape=(30,1)))
model.add(Dense(5))
model.add(Dense(31, activation='softmax'))


model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
history = model.fit(X_train, y_train, batch_size=32, epochs=100, validation_data=(X_test, y_test))

print('테스트 정확도 : %.4f' % (model.evaluate(X_test,y_test)[1]))


model.save('lstm_v1.h5')

def predict_input(sentence):
    sentences = []
    sentences.append(sentence)
    sentences = pos_tag(sentences)
    input_seq = convert_txt_to_idx(sentences, ques_to_idx)
    input_seq = pad_sequences(input_seq, maxlen=max_sequences)
    input_seq = input_seq.reshape((input_seq.shape[0], input_seq.shape[1],1))

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



input_seq = predict_input('뉴스 알려줘')
print(input_seq)

result_idx = model.predict(input_seq)
print(result_idx)

intent = convert_idx_to_txt(result_idx, idx_to_intent)

print(intent)


for i in [3, 6, 8, 33, 24, 101, 400, 22, 700, 1001, 1200, 55, 67, 1080]:
    input_seq = predict_input(data.question[i])
    print(data.question[i])
    result_idx = model.predict(input_seq)
    intent = convert_idx_to_txt(result_idx, idx_to_intent)
    print(intent)


# _________________________________________________________________
# Layer (type)                 Output Shape              Param #
# =================================================================
# lstm_1 (LSTM)                (None, 50)                10400
# _________________________________________________________________
# dense_1 (Dense)              (None, 5)                 255
# _________________________________________________________________
# dense_2 (Dense)              (None, 31)                186
# =================================================================
# Total params: 10,841
# Trainable params: 10,841
# Non-trainable params: 0
# _________________________________________________________________