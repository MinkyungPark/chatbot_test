## 데이터 두개 합친거에, 데이터셋 랜덤으로 ##

import pandas as pd
import numpy as np

data1 = pd.read_csv('intent_classifier_test\data\intent_train.csv', names=['question','intent'], encoding='utf-8')
data2 = pd.read_csv('intent_classifier_test\data\intent_grrc.csv', names=['question','intent'], encoding='utf-8')
data1 = data1[1:]
data2 = data2[1:]

data = pd.concat([data1, data2], axis=0, ignore_index=True, names=['question','intent'])
# data.info() 4086 object
data.intent = data.intent.astype('str')

### <sos> \t, <eos> \n
data.intent = data.intent.apply(lambda x : '\t '+ x +' \n') # \t 번역 \n

### 글자 집합 생성
que_vocab = set()
for line in data.question: # 1줄씩
    for char in line: # 1글자씩
        que_vocab.add(char)

intent_vocab = set()
for line in data.intent:
    for char in line:
        intent_vocab.add(char)

que_vocab_size = len(que_vocab)+1 # 651
intent_vocab_size = len(intent_vocab)+1

### 인덱스 지정
que_vocab = sorted(list(que_vocab))
intent_vocab = sorted(list(intent_vocab))

que_to_index = dict([(word, i+1) for i, word in enumerate(que_vocab)])
intent_to_index = dict([(word, i+1) for i, word in enumerate(intent_vocab)])
# print(que_to_index)

### 인코더 정수 인코딩
encoder_input = []
for line in data.question:
    temp_X = []
    for w in line:
        temp_X.append(que_to_index[w])
    encoder_input.append(temp_X)

### 디코더 정수 인코딩
decoder_input = []
for line in data.intent:
    temp_X = []
    for w in line:
        temp_X.append(intent_to_index[w])
    decoder_input.append(temp_X)

### 실제값에는 <sos> 있을 필요 없다. 제거.
decoder_target = []
for line in data.intent:
    t = 0
    temp_X = []
    for w in line:
        if t > 0:
            temp_X.append(intent_to_index[w])
        t = t+1
    decoder_target.append(temp_X)

### 패딩 작업
max_que_len = max([len(line) for line in data.question]) # 27
max_intent_len = max([len(line) for line in data.intent]) # 10

from keras.preprocessing.sequence import pad_sequences
encoder_input = pad_sequences(encoder_input, maxlen=max_que_len, padding='post')
decoder_input = pad_sequences(decoder_input, maxlen=max_intent_len, padding='post')
decoder_target = pad_sequences(decoder_target, maxlen=max_intent_len, padding='post')

### word2vec 말고 원핫인코딩 사용, 입력 출력 둘다
from keras.utils import to_categorical
encoder_input = to_categorical(encoder_input)
decoder_input = to_categorical(decoder_input)
decoder_target = to_categorical(decoder_target)

### train, test set 분리
from sklearn.model_selection import train_test_split
encoder_input_train, encoder_input_test = train_test_split(encoder_input, random_state=66, test_size=0.2)
decoder_input_train, decoder_input_test = train_test_split(decoder_input, random_state=66, test_size=0.2)
decoder_target_train, decoder_target_test = train_test_split(decoder_target, random_state=66, test_size=0.2)


### 훈련모델 ###
from keras.layers import Input, LSTM, Embedding, Dense
from keras.models import Model

## ENCODER
encoder_inputs = Input(shape=(None, que_vocab_size))
encoder_lstm = LSTM(units=256, return_state=True) 
# 은닉 크기 256, 인코더 내부상태를 디코더로 넘겨줌 return_state=T
encoder_outputs, state_h, state_c = encoder_lstm(encoder_inputs)
# state_h 은닉상태, state_c 셀상태
encoder_states = [state_h, state_c] 
# encoder_outputs 여기서는 필요 없음
# encoder_states가 context vector 인 것

## DECODER
decoder_inputs = Input(shape=(None, intent_vocab_size))
decoder_lstm = LSTM(units=256, return_sequences=True, return_state=True)
decoder_outputs, _, _ = decoder_lstm(decoder_inputs, initial_state=encoder_states) # 디코더의 첫 상태를 인코더의 은닉,셀 상태로
decoder_softmax_layer = Dense(intent_vocab_size, activation='softmax')
decoder_outputs = decoder_softmax_layer(decoder_outputs)

model = Model([encoder_inputs, decoder_inputs], decoder_outputs)

### 훈련
model.compile(optimizer='rmsprop', loss='categorical_crossentropy')
model.fit(x=[encoder_input_train, decoder_input_train], y=decoder_target_train, batch_size=4, epochs=200, validation_split=0.2)


### 동작모델 ###
# ENCODER
encoder_model = Model(inputs=encoder_inputs, outputs=encoder_states)

# DECODER
decoder_state_input_h = Input(shape=(256,))
decoder_state_input_c = Input(shape=(256,))
decoder_states_inputs = [decoder_state_input_h, decoder_state_input_c]
decoder_outputs, state_h, state_c = decoder_lstm(decoder_inputs, initial_state=decoder_states_inputs) # 초기상태를 이전상태로 사용
decoder_states = [state_h, state_c] # 은닉상태와 셀상태 버리지 X
decoder_outputs = decoder_softmax_layer(decoder_outputs)
decoder_model = Model(inputs=[decoder_inputs] + decoder_states_inputs, outputs=[decoder_outputs] + decoder_states)

# 인덱스로부터 단어를 얻는다.
index_to_que = dict(
    (i, char) for char, i in que_to_index.items()
)
index_to_intent = dict(
    (i, char) for char, i in intent_to_index.items()
)

def decode_sequence(input_seq):
    states_value = encoder_model.predict(input_seq) # 입력으로부터 인코더 상태 얻음
    target_seq = np.zeros((1, 1, intent_vocab_size)) # <sos>에 해당하는 원핫벡터 새성
    target_seq[0, 0, intent_to_index['\t']] = 1.

    stop_condition = False
    decoded_sentence = ''
    while not stop_condition:
        output_tokens, h, c = decoder_model.predict([target_seq] + states_value)
        sampled_token_index = np.argmax(output_tokens[0, -1, :])
        sampled_char = index_to_intent[sampled_token_index]
        decoded_sentence += sampled_char

        if(sampled_char == '\n' or len(decoded_sentence) > max_intent_len):
            stop_condition = True
        
        # 길이가 1인 target sequence update
        target_seq = np.zeros((1, 1, intent_vocab_size))
        target_seq[0, 0, sampled_token_index] = 1

        states_value = [h, c] # state update
    
    return decoded_sentence

if __name__ == "__main__":
    for seq_index in [3, 5, 100, 300, 1001, 1200,1500, 1600, 1901,2500, 2600, 3000,3200,3300,3500,3700,4000,4070]:
        input_seq = encoder_input[seq_index: seq_index + 1]
        decoded_sentence = decode_sequence(input_seq)
        print(35 * '-')
        print('Question : ', data.question[seq_index])
        print('Intent : ', data.intent[seq_index][1:len(data.intent[seq_index])-1])
        print('Predict : ', decoded_sentence[:len(decoded_sentence)-1])