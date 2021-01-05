# simple version 

import csv
from konlpy.tag import Okt
from gensim.models import word2vec

f = open('.\\data\\ratings_train.txt', encoding='utf-8') # pandas쓸경우 ""처리를 해줘야... csv.QUOTE_NOTE(3)
rdr = csv.reader(f, delimiter='\t')
rdw= list(rdr)
f.close()

okt = Okt() # pos_tagger

result = []
# 한줄 씩
for line in rdw:
    malist = okt.pos(line[1], norm=True, stem=True)
    r=[]
    for word in malist:
        # 조사, 어미, 문장부호 제외
        if not word[1] in ['Josa', 'Eomi', 'Punctuation']:
            r.append(word[0])
    # 각 형태소 사이에 공백 넣고 양쪽 공백 제거
    rl = (' '.join(r)).strip()
    result.append(rl)

# 각 형태소 저장
with open('.\\model\\sentiment_test.nlp', 'w', encoding='utf=8') as fp:
    fp.write('\n'.join(result))

# 결과 모델 생성
wData = word2vec.LineSentence('.\\model\\sentiment_test.nlp')
wModel = word2vec.Word2Vec(wData, size=200, window=10, hs=1, min_count=2, sg=1)
wModel.save('.\\model\\sentiment_test.model')

print('model save finished')

model = word2vec.Word2Vec.load('.\\model\\sentiment_test.model')

print(model.most_similar(positive=['재밌다']))
print(model.most_similar(positive=['즐겁다']))
print(model.most_similar(positive=['슬프다']))
