# %%
import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

df = pd.read_csv('C:\Projects_Python\chatbot_test\intent_classifier_test\data\intent_data.csv', names=['question','intent'], encoding='utf-8')

question = list(df['question'][1:])
intent = list(df['intent'][1:])

# 토큰화
from konlpy.tag import Okt
from keras.preprocessing.text import Tokenizer
from keras.preprocessing.sequence import pad_sequences

okt = Okt()

for i, sentence in enumerate(question):
    tmp = okt.morphs(sentence) # ['오랜','만이','야']
    question[i] = ' '.join(tmp) # ['오랜 만이 야', ...]

# Tokenizer의 파라미터 num_words= 지정하면 적은 빈도수 단어 생략
tokenizer = Tokenizer()
tokenizer.fit_on_texts(question)
x_data = tokenizer.texts_to_sequences(question)
word_index = tokenizer.word_index
# , '이수일': 1718, '심': 1719, '순애': 1720, '이상': 1721, '이중섭': 1722, '중섭': 1723, '잡담': 1724}
# print(len(tokenizer.word_index)) # 토큰의 갯수 1724
# print(x_data[:10])

# 가장 긴 문장 길이 확인
max_len = max([len(sentence) for sentence in x_data])
print(max_len) # 가장 단어가 많은 문장 한 문장당 10개의 벡터(토큰 갯수10)

x_data = pad_sequences(x_data, maxlen=max_len)

# intent 정수 인덱싱
intent_sort = sorted(list(intent))
intent_set = list(set(intent_sort))
intent_to_idx = {word: index for index, word in enumerate(intent_set)}

y_data = []

for word in intent:
    y_data.append(intent_to_idx[word])

from keras.utils import np_utils

y_data = np.asarray(y_data)
y_data = np_utils.to_categorical(y_data)

print(x_data[:10])
print(y_data[:10])
print(x_data.shape) # (4095, 10)
print(y_data.shape) # (4095, 29)

# 데이터셋 섞기
from sklearn.model_selection import train_test_split
X_train, X_test, y_train, y_test = train_test_split(x_data, y_data, random_state=66, test_size=0.2)


# %%
import gensim.models as g

model = g.Doc2Vec.load('C:\Projects_Python\chatbot_test\intent_classifier_test\model\w2v_10features_5window_iter200')

vocab = list(model.wv.vocab) # 9861 tokens(words)
vector = model[vocab] # 10차원

max_words = len(vocab)
embedding_dim = 10

print(len(vocab)) # 9861
print(vocab[:10])
print(vector[:10])

# %%
# embedding_index = {단어:[단어벡터], ...}
embedding_index = {}
for i in range(len(vocab)):
    embedding_index.setdefault(vocab[i], list(vector[i]))

print(list(embedding_index.keys())[0])
print(list(embedding_index.values())[0])
print(list(word_index.keys())[0])
print(list(word_index.values())[0])
print(embedding_index['알려줘'])

# %%
# (max_words, embedding_dim) 크기인 임베딩 행렬을 임베딩 층에 주입.

embedding_matrix = np.zeros((max_words, embedding_dim))

for word, i in word_index.items():
    embedding_vector = embedding_index.get(word)
    if i < max_words:
        if embedding_vector is not None:
            # 임베딩 인덱스에 없는 단어는 0
            embedding_matrix[i-1] = embedding_vector
            # word_index의 index는 1부터 시작

print(embedding_matrix[0])

# %%
from keras.models import Sequential
from keras.layers import Embedding, Dense, Flatten, SimpleRNN
from keras.layers.embeddings import Embedding

# model 구성
model = Sequential()
model.add(Embedding(max_words, embedding_dim, input_length=max_len))
model.add(SimpleRNN(128))
model.add(Dense(32))
model.add(Dense(29, activation='softmax')) # 왜 그런데 29개지.. 30개여야 하는데

# 사전 훈련된 word2vec 단어 임베딩을 Embdding 층에 로드하기
model.layers[0].set_weights([embedding_matrix])
model.layers[0].trainable = False

# or Embedding(max_words, embedding_dim, input_length=max_len, weights=[embedding_matrix], trainable=Flase)

from keras.callbacks import EarlyStopping
early_stop = EarlyStopping(monitor='loss', patience=10, mode='auto')

model.compile(optimizer='adam', loss='categorical_crossentropy', metrics=['acc'])
history = model.fit(X_train, y_train, epochs=500, batch_size=2, callbacks=[early_stop], validation_split=0.1)
# model.save_weights('pre_trained_glove_model.h5')


# %%
# plot 그리기
import matplotlib.pyplot as plt

fig, loss_ax = plt.subplots()

acc_ax = loss_ax.twinx()

loss_ax.plot(history.history['loss'], 'y', label='train loss')
loss_ax.plot(history.history['val_loss'], 'r', label='val loss')
loss_ax.set_ylim([0.0, 3.0])

acc_ax.plot(history.history['acc'], 'b', label='train acc')
acc_ax.plot(history.history['val_acc'], 'g', label='val acc')
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
