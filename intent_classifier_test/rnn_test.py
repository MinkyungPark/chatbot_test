# %%
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences
from keras.utils import np_utils

seed = 0
np.random.seed(seed)

data = pd.read_csv('C:\Projects_Python\chatbot_test\intent_classifier_test\data\intent_data.csv', names=['question','intent'], encoding='utf-8')

print(data.shape)
data.info()
# question    3918 non-null object
# intent      3918 non-null object
# null 확인

X_data = data['question']
y_data = data['intent']

tokenizer = Tokenizer()
tokenizer.fit_on_texts(X_data) # 토큰화
sequences = tokenizer.texts_to_sequences(X_data) # word -> num, index

tokenizer_y = Tokenizer()
tokenizer_y.fit_on_texts(y_data) # 토큰화
sequences_y = tokenizer_y.texts_to_sequences(y_data)

word_to_index = tokenizer.word_index # 각 단어 별 인덱스 부여
print(word_to_index)
voca_size = len(word_to_index)+1
print(voca_size) # x 데이터의 단어 수 1893

X_data = sequences
y_data = pd.DataFrame(sequences_y)
y_data = np.asarray(y_data)
y_data = np_utils.to_categorical(y_data)

print('문장의 최대 길이 : %d' % max(len(l) for l in X_data)) # 7

max_len = 7
X_data = pad_sequences(X_data, maxlen=max_len) # x의 길이 전부 7로 맞추기 위해 패딩

from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(X_data, y_data, test_size=0.2, random_state=42)

from keras.layers import SimpleRNN, Dense, Embedding
from keras.models import Sequential

# model 구성
model = Sequential()
model.add(Embedding(voca_size, 10)) # embedding벡터차원 10
model.add(SimpleRNN(128)) # RNN레이어 128개
model.add(Dense(32))
model.add(Dense(y_data.shape[1], activation='softmax'))

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
hist = model.fit(X_train, y_train, epochs=500, batch_size=2, validation_split=0.1, callbacks=[early_stop])

# SimpleRNN(바닐라 RNN) 성능문제.. LSTM셀 또는 GRU로 인코더 디코더 구성

# %%

# plot 그리기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(hist.history['loss'], 'y', label='train loss')
loss_ax.plot(hist.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 3.0])

acc_ax.plot(hist.history['acc'], 'b', label='train acc')
acc_ax.plot(hist.history['val_acc'], 'g', label='val acc')
acc_ax.set_ylim([0.0, 1.0])

loss_ax.set_xlabel('epoch')
loss_ax.set_ylabel('loss')
acc_ax.set_ylabel('accuray')

loss_ax.legend(loc='upper left')
acc_ax.legend(loc='lower left')

plt.show()

# 모델 평가
loss_and_metrics = model.evaluate(X_test, y_test, batch_size=64)
print('## evaluation loss and_metrics ##')
print(loss_and_metrics)

# %%
