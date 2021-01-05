# %%
import numpy as np
seed = 0
np.random.seed(seed)

import pandas as pd

df = pd.read_csv('C:\Projects_Python\chatbot_test\extend_data\data\corpus.csv', encoding='utf-8', names=['sentences'])
sentences = df['sentences']

text = ' '.join(sentences)
print('총 문자열의 길이 :', len(text)) # 7293
print(text[:100])

# %%
char_voca = sorted(list(set(text)))
voca_size = len(char_voca)
print('글자 집합 크기 : {}'.format(voca_size)) #389

# %%
char_to_idx = dict((c,i) for i,c in enumerate(char_voca))
print(char_to_idx)


# %%
idx_to_char = {}
for key, value in char_to_idx.items():
    idx_to_char[value] = key


# %%
seq_length = 20 # 문장의 길이를 20
n_samples = int(np.floor((len(text) - 1) / seq_length)) # 문자열을 20 등분 => 총 샘플 개수
print('문장 샘플의 수 : {}'.format(n_samples)) # 364


# %%
X_train = []
y_train = []

for i in range(n_samples):
    X_sample = text[i*seq_length: (i+1)*seq_length]
    X_encoded = [char_to_idx[c] for c in X_sample]
    X_train.append(X_encoded)

    y_sample = text[i*seq_length+1: (i+1)*seq_length+1]
    y_encoded = [char_to_idx[c] for c in y_sample]
    y_train.append(y_encoded)

print(X_train[:2])
print(y_train[:2])

# y_train은 X_train에서 오른쪽 한칸 쉬프트 된 문장임

# %%
from keras.utils import to_categorical
X_train = to_categorical(X_train)
y_train = to_categorical(y_train)

print('X_trian shape : {}'.format(X_train.shape)) # (364, 20, 389)
print('y_trian shape : {}'.format(y_train.shape)) # (364, 20, 389)

# %%
# 모델링
from keras.models import Sequential
from keras.layers import Dense, LSTM, TimeDistributed

model = Sequential()
model.add(LSTM(256, input_shape=(None, X_train.shape[2]), return_sequences=True))
model.add(LSTM(256, return_sequences=True))
model.add(TimeDistributed(Dense(voca_size, activation='softmax')))

model.compile(loss='categorical_crossentropy', optimizer='adam', metrics=['accuracy'])
model.fit(X_train, y_train, epochs=100, verbose=2, batch_size=20, validation_split=0.2)

# %%
def sentence_generation(model, length):
    ix = [np.random.randint(voca_size)] # 글자에 대한 랜덤 인덱스 생성
    y_char = [idx_to_char[ix[-1]]] # 랜덤 익덱스로부터 글자 생성
    print(ix[-1],'번 글자',y_char[-1],'로 예측을 시작!')
    X = np.zeros((1, length, voca_size)) # (1, length, 55) 크기의 X 생성. 즉, LSTM의 입력 시퀀스 생성

    for i in range(length):
        X[0][i][ix[-1]] = 1 # X[0][i][예측한 글자의 인덱스] = 1, 즉, 예측 글자를 다음 입력 시퀀스에 추가
        print(idx_to_char[ix[-1]], end="")
        ix = np.argmax(model.predict(X[:, :i+1, :])[0], 1)
        y_char.append(idx_to_char[ix[-1]])
    return ('').join(y_char)

sentence_generation(model, 100)



# Epoch 100/100
#  - 1s - loss: 1.0030 - accuracy: 0.7675 - val_loss: 3.3787 - val_accuracy: 0.4541
# 81 번 글자 닿 로 예측을 시작!
# 닿는 어아세요? 요즘 재밌으신가 아요. 요즘 불편하신가요? 즐겨하시죠? 최근에눈이 자주 보시고 계시죠? 주칙적이있으세요? 요즘 재밌으신가요? 적이 있으신가요? 자주에 남는 많으
# '닿는 어아세요? 요즘 재밌으신가 아요. 요즘 불편하신가요? 즐겨하시죠? 최근에 눈이 자주 보시고 계시죠? 주칙적이 있으세요? 요즘 재밌으신가요? 적이 있으신가요? 자주에 남는 많으세'