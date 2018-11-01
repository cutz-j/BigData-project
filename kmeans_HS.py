'''
K-means는 접근성 분석을 위해서 사용됩니다
'''
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist


## 어린이집 데이터 전처리 ##
all_center = pd.read_csv("d:/project_data/all_center9.csv", sep=",", encoding="euc-kr")
x_test = all_center[all_center['Type'] == "국공립"] # 국공립만 선택

## train_test split ## --> train: x_test / predict: x_test (cluster)

## k-means ##
k = 424 # 서울시 행정 동 개수

# n_cluster = 424, max_iter=3000 #
k_means = KMeans(n_clusters=k, max_iter=3000)
k_means.fit(x_test.iloc[:, 14:16])
k_cluster = k_means.predict(x_test.iloc[:, 14:16])
x_test['k_cluster'] = k_cluster

ss = silhouette_score(x_test.iloc[:, 14:16], k_means.labels_, metric='euclidean')
print(ss) # 0.4312


# 한글 폰트 깨지는 문제 #
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# 시각화 그래프 # --> 구별 평균 추정값 (TEST)
fig = plt.figure()
for i in range(k):
    scat = plt.scatter(x_test[x_test['k_cluster']==i].iloc[:, 14], x_test[x_test['k_cluster']==i].iloc[:, 15],
                       s=10)
    

fig.show()


