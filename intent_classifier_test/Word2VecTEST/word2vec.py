# %%
import numpy as np
import pandas as pd

seed = 0
np.random.seed(seed)

df = pd.read_csv('C:\Projects_Python\chatbot_test\intent_classifier_test\data\intent_data.csv', names=['question','intent'], encoding='utf-8')

# print(data.shape)
# question    3918 non-null object
# intent      3918 non-null object
# null 확인

df2 = pd.read_csv('C:\Projects_Python\chatbot_test\intent_classifier_test\data\ChatbotData.csv', names=['Q','A','label'], encoding='utf-8')


from konlpy.tag import Okt

okt = Okt()

def parse(df):
    parsed = []
    for line in df:
        tmp = okt.morphs(line)
        parsed.append(tmp)
    return parsed


question1 = parse(df['question'][1:])
question2 = parse(df2['Q'][1:])

question = question1 + question2

print(question1[:10])
print(question2[:10])
print(question[:100])
print(question[:-100])



# %%
## voca dictionary

# ques_dic = []
# int_dic = []

# for line in question:
#     for word in line:
#         ques_dic.append(word)

# for word in df['intent']:
#     int_dic.append(word)

# ques_dic = [word for word in ques_dic if len(word) > 0 ] # 길이가 0 삭제
# ques_dic = list(set(ques_dic)) # 중복 단어 삭제
# int_dic = list(set(int_dic))

# print(len(ques_dic)) # 1726 단어
# print(len(int_dic)) # 30 종류

# total_voca = sorted(ques_dic) + sorted(int_dic)

# print(total_voca)


# %%
from gensim.models.word2vec import Word2Vec
import logging

# logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)

model = Word2Vec(question, size=10, window=5, iter=200, workers=4, min_count=1, sg=1,)
model.init_sims(replace=True)

# print(model)

print(model.wv.similarity('노래','음악'))
print(model.wv.similarity('혈당','혈압'))
print()
print(model.wv.most_similar('알려줘'))
print()
print(model.wv.most_similar('지냈니'))



# %%
# 학습이 끝나면 필요 없는 메모리 unload
model.init_sims(replace=True)

model_name = 'w2v_10features_5window_iter200'
model.save(model_name)


# %%
model.wv.doesnt_match('노래 음악 혈압'.split()) # 혈압
# %%
model.wv.doesnt_match('담배 가족 친구'.split()) # 담배
# %%
model.wv.doesnt_match('안좋아 우울해 울적해 보고싶어 알려줘'.split()) # 알려줘


# %%
# 시각화
from sklearn.manifold import TSNE # 차원축소 T-분포 확률임베딩
import matplotlib as mpl
import matplotlib.pyplot as plt
import gensim 
import gensim.models as g
import pandas as pd

model = g.Doc2Vec.load('C:\Projects_Python\chatbot_test\intent_classifier_test\model\w2v_10features_5window_iter200')

vocab = list(model.wv.vocab)
X = model[vocab]

print(len(X)) # 9861
print(X[0][:10])
tsne = TSNE(n_components=2)

# 300개의 단어에 대해서만 시각화
X_tsne = tsne.fit_transform(X[:300,:])
# X_tsne = tsne.fit_transform(X)

df_tsne = pd.DataFrame(X_tsne, index=vocab[:300], columns=['x', 'y'])
print(df_tsne.shape) # 100개만 지정했으므로 (100, 2)
print(df_tsne.head(10))

#             x         y
# 오랜   3.180843  3.267357
# 만이   3.810435  5.062474
# 야    3.957183  6.299223
# 잘   -3.054624  1.714211
# 있었어 -5.841204  3.220338
# 반가워 -5.383442 -2.584780
# 오랜만  0.298631 -1.766186
# 어떻게 -1.429489  0.523382
# 지냈어 -1.709383  0.669397
# 지냈니 -3.312621  1.885789



# %%

# 한글 깨짐 때문에 폰트지정
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname='C:\Windows\Fonts\H2GTRM.ttf').get_name()
rc('font', family=font_name)
# 마이너스 폰트 깨짐 해결
mpl.rcParams['axes.unicode_minus'] = False

fig = plt.figure()
fig.set_size_inches(40, 20)
ax = fig.add_subplot(1, 1, 1)

ax.scatter(df_tsne['x'], df_tsne['y'])

for word, pos in df_tsne.iterrows():
    ax.annotate(word, pos, fontsize=30)
plt.show()

# %%
