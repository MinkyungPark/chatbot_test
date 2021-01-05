import numpy as np
seed = 0
np.random.seed(seed)

import pandas as pd

df = pd.read_csv('C:\Projects_Python\chatbot_test\extend_data\data\corpus.csv', encoding='utf-8', names=['sentences'])
sentences = df['sentences']

words = []
for sentence in sentences:
    for word in sentence.split(' '):
        words.append(word)

print(words[:20])

from keras.preprocessing.text import Tokenizer

t = Tokenizer()
t.fit_on_texts(df['sentences'])
voca_size = len(t.word_index) + 1

print(t.word_index)
print(voca_size) # 776


# 훈련데이터 생성
sequences = list()
for line in sentences:
    encoded = t.texts_to_sequences([line])[0]
    for i in range(1, len(encoded)):
        sequence = encoded[:i+1]
        sequences.append(sequence)

print(sequences[:10])
print('Train Sample 개수 : ', len(sequences)) # 1357

max_len = max(len(l) for l in sequences) # 가장 긴 샘플 길이
print('샘플의 최대 길이 : {}'.format(max_len)) # 11

# 11로 패딩
from keras.preprocessing.sequence import pad_sequences
sequences = pad_sequences(sequences, maxlen=max_len, padding='pre')
print(sequences[:10])

import numpy as np
sequences = np.array(sequences)
X = sequences[:,:-1]
y = sequences[:,-1]

print(X[:10])
print(y[:10])

# y의 원핫인코딩
from keras.utils import to_categorical
y = to_categorical(y, num_classes=voca_size)

print(y[:10])
print(y.shape) # 1357, 776

from keras.models import Sequential
from keras.layers import Embedding, Dense, LSTM

model = Sequential()
model.add(Embedding(voca_size, 10, input_length=max_len-1))
model.add(LSTM(64))
model.add(Dense(voca_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X, y, epochs=300, verbose=2, validation_split=0.2, batch_size=11)

def sentence_generation(model, t, current_word, n): # 모델, 토크나이저, 현재 단어, 반복할 횟수
    init_word = current_word # 처음 들어온 단어도 마지막에 같이 출력하기위해 저장
    sentence = ''
    for _ in range(n): # n번 반복
        encoded = t.texts_to_sequences([current_word])[0] # 현재 단어에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=max_len-1, padding='pre') # 데이터에 대한 패딩
        result = model.predict_classes(encoded, verbose=0)
    
        for word, index in t.word_index.items(): 
            if index == result: # 만약 예측한 단어와 인덱스와 동일한 단어가 있다면
                break # 해당 단어가 예측 단어이므로 break
        current_word = current_word + ' '  + word # 현재 단어 + ' ' + 예측 단어를 현재 단어로 변경
        sentence = sentence + ' ' + word # 예측 단어를 문장에 저장
    
    sentence = init_word + sentence
    return sentence

print(sentence_generation(model, t, '요즘', 4))
print(sentence_generation(model, t, '요즘', 4))
print(sentence_generation(model, t, '요즘', 5))
print(sentence_generation(model, t, '요즘', 6))
print(sentence_generation(model, t, '요즘', 7))



# LSTM
# epoch 300 batch_size = 1
# Epoch 183/300
# - 13s - loss: 0.3230 - accuracy: 0.8802 - val_loss: 8.6776 - val_accuracy: 0.5294

# epoch 300 batch_size = 11
# 위와 비슷...

# 요즘 어디 아프신 곳은 없으세요
# 요즘 어디 아프신 곳은 없으세요
# 요즘 어디 아프신 곳은 없으세요 있으신가요
# 요즘 어디 아프신 곳은 없으세요 있으신가요 있으신가요
# 요즘 어디 아프신 곳은 없으세요 있으신가요 있으신가요 있으신가요