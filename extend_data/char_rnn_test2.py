import numpy as np
seed = 0
np.random.seed(seed)

import pandas as pd

df = pd.read_csv('C:\Projects_Python\chatbot_test\extend_data\data\corpus.csv', encoding='utf-8', names=['sentences'])
sentences = df['sentences']

# 하나의 문자열로 합치기
text = ' '.join(sentences)
# print('총 문자열의 길이 :', len(text)) # 7293
# print(text[:100])

char_voca = sorted(list(set(text)))
voca_size = len(char_voca)
# print('글자 집합 크기 : {}'.format(voca_size)) #389

char_to_idx = dict((c,i) for i,c in enumerate(char_voca))
# print(char_to_idx)

# 모든 샘플의 길이가 10이 되도록
length = 11
sequences = []
for i in range(length, len(text)):
    seq = text[i-length:i]
    sequences.append(seq)

# print('총 훈련 샘플 수 : %d' % len(sequences)) # 7282
# print(sequences[:20])
# ['평소에 앓고 계시는 ', '소에 앓고 계시는 지', '에 앓고 계시는 지병', ' 앓고 계시는 지병이', '앓고 계시는 지병이 ', '고 계시는 지병이 있', ' 계시는 지병이 있으', '계시는 지병이 있으신', '시는 지병이 있으신가', '는 지
# 병이 있으신가요', ' 지병이 있으신가요?', '지병이 있으신가요? ', '병이 있으신가요? 지', '이 있으신가요? 지금', 

# 하나씩 정수 인코딩
X_train = []
for line in sequences:
    tmp = [char_to_idx[char] for char in line]
    X_train.append(tmp)

sequences = np.array(X_train)

X_train = sequences[:,:-1]
y_train = sequences[:,-1]

from keras.utils import to_categorical
sequences = [to_categorical(x, num_classes=voca_size) for x in X_train]
X_train = np.array(sequences)
y_train = to_categorical(y_train, num_classes=voca_size)

from keras.models import Sequential
from keras.layers import Dense, LSTM
from keras.preprocessing.sequence import pad_sequences

model = Sequential()
model.add(LSTM(128, input_shape=(X_train.shape[1], X_train.shape[2])))
model.add(Dense(voca_size, activation='softmax'))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=200, verbose=2, validation_split=0.2, batch_size=10)

def sentence_generation(model, char_to_idx, seq_length, seed_text, n): # 모델, 인덱스 정보, 문장 길이, 초기 시퀀스, 반복 횟수
    init_text = seed_text # 문장 생성에 사용할 초기 시퀀스
    sentence = ''

    for _ in range(n): # n번 반복
        encoded = [char_to_idx[char] for char in seed_text] # 현재 시퀀스에 대한 정수 인코딩
        encoded = pad_sequences([encoded], maxlen=seq_length, padding='pre') # 데이터에 대한 패딩
        encoded = to_categorical(encoded, num_classes=len(char_to_idx))
        result = model.predict_classes(encoded, verbose=0)
        # 입력한 X(현재 시퀀스)에 대해서 y를 예측하고 y(예측한 글자)를 result에 저장.
        for char, index in char_to_idx.items(): # 만약 예측한 글자와 인덱스와 동일한 글자가 있다면
            if index == result: # 해당 글자가 예측 글자이므로 break
                break
        seed_text=seed_text + char # 현재 시퀀스 + 예측 글자를 현재 시퀀스로 변경
        sentence=sentence + char # 예측 글자를 문장에 저장

    sentence = init_text + sentence
    return sentence

print(sentence_generation(model, char_to_idx, 10, '요즘', 80))


# Epoch 200/200
#  - 7s - loss: 0.0400 - accuracy: 0.9768 - val_loss: 3.4133 - val_accuracy: 0.7227
# 요즘 잘 있으신가요? 지금도 지속적으로 관리해줘야 하는 병이 있으세요? 식사하시고 바로 누우시면 소화에 안좋을 수도 있어요.알고계시죠? 규칙적인 식