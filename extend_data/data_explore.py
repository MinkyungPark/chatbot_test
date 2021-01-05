# %%
import pandas as pd
import itertools as it
from konlpy.tag import Kkma

df = pd.read_csv('C:\Projects_Python\chatbot_test\extend_data\data\corpus.csv', encoding='utf-8', names=['sentences'])

# print(df.info()) # 1 column, 311 sentences
sentences = df['sentences']

kkma = Kkma()

# 품사별로 분류, 중복제거, 명사(NNG), 동사만(VV, VA) 추출
def parse(sentence):   
    groups = {word: set(i[0] for i in tag) 
              for word, tag in it.groupby(sorted(kkma.pos(sentence), key=lambda item:item[1]), lambda item:item[1])} 
    return [*groups.get('NNG', []), *[f'{word}다' for word in groups.get('VV', [])], *[f'{word}다' for word in groups.get('VA', [])]]

df['nlp'] = sentences.apply(parse)

print(df['nlp'][0])


# %%
# Word2Vec Model로 유사도
from gensim.models.word2vec import Word2Vec

model = Word2Vec(df['nlp'].values, sg=1, window=5, min_count=1, workers=4, iter=100)
model.init_sims(replace=True)

print(model.wv.similarity('식사','밥'))
print(model.wv.most_similar('식사'))


# %%
# df['nlp']는 한 문장, 한 로우 -> 리스트 하나로 합치기
all_nlp = []
for sentence in df['nlp']:
    for word in sentence:
        all_nlp.append(word)

# all_nlp = list(set(all_nlp)) # count 위해 일단 중복제거 배제
print(all_nlp)


# %%
# 빈도수

count = dict( (l, all_nlp.count(l) ) for l in set(all_nlp))
# print(count) # dict count.keys(), count.values()

sorted_count = sorted(count.items(), key=lambda x:x[1], reverse=True)
print(sorted_count[:10])

# %%
