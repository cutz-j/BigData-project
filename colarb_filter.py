### 협업 필터링 ###
## 코사인 유사도 기반 ##
## 최하 클러스터 유사 어린이집 찾기 ##
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import MinMaxScaler

## 데이터 전처리 ##
all_data = pd.read_csv("d:/project_data/KK_k150_2021.csv", sep=",", encoding='cp949')

# 필요 데이터 벡터화 #
data = pd.concat((all_data['predK25'], all_data['center_access'], all_data['center_access_2'],
                                  all_data['people_access'], all_data['people_access_2']), axis=1)

data.index = all_data['Name'] # 인덱스 첨부

mm = MinMaxScaler()
data_scale = mm.fit_transform(data)
ana = cosine_similarity(data_scale)

data_259 = pd.DataFrame(ana[259], index=all_data['Name'], columns=['봄빛'])
#data_259 = data_259.sort_values(by='봄빛', ascending=False)
data_261 = pd.DataFrame(ana[261], index=all_data['Name'], columns=['상일'])
#data_261 = data_261.sort_values(by='상일', ascending=False)
data_270 = pd.DataFrame(ana[270], index=all_data['Name'], columns=['한마을'])
#data_270 = data_270.sort_values(by='한마을', ascending=False)
data_824 = pd.DataFrame(ana[824], index=all_data['Name'], columns=['늘사랑'])
#data_824 = data_824.sort_values(by='늘사랑', ascending=False)
data_686 = pd.DataFrame(ana[686], index=all_data['Name'], columns=['노원'])
#data_686 = data_686.sort_values(by='노원', ascending=False)

cos_sim = pd.concat((data_259, data_261, data_270, data_824, data_686), axis=1)
cos_sim = cos_sim[cos_sim > 0.9]
cos_sim = cos_sim.dropna(axis=0)

#cos_sim.to_csv("d:/project_data/cos_sim.csv", encoding="cp949")
#