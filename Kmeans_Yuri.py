### k-means & 접근성 분석 ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist


## 어린이집 데이터 전처리 ##
all_center = pd.read_csv("d:/project_data/KNN_data.csv", sep=",", encoding="euc-kr")
x_test = all_center[all_center['Type'] == "국공립"] # 국공립만 선택
X = x_test.iloc[:, 14:16]

## train_test split ## --> train: x_test / predict: x_test (cluster)

## k-means ##
# 최적 클러스터 k 찾기 # -> elbow graph
K = 150

def k_search():
    ## 최적의 k를 엘보 그래프로 찾는 함수 ##
    K = range(25,200,5)
    KM = [KMeans(n_clusters=k).fit(X) for k in K] # 각각의 k(25~300까지 5단위), k-means 명령어
    ss = [silhouette_score(X, k.labels_, metric='euclidean') for k in KM]
    centroids = [k.cluster_centers_ for k in KM] # 각 k-means마다 클러스터별 center 거리
    D_k = [cdist(X, centrds, 'euclidean') for centrds in centroids] # 센터와 X데이터간의 거리
    cIdx = [np.argmin(D, axis=1) for D in D_k] # 최소 거리
    dist = [np.min(D, axis=1) for D in D_k] # 최소 거리
    avgWithinSS = [sum(d) / X.shape[0] for d in dist] # 클러스터 내 제곱 평균 (sum of sq)
    wcss = [sum(d**2) for d in dist] # sq 계산
    tss = sum(pdist(X)**2 / X.shape[0]) # X각각의 거리 제곱 / m --> 평균
    bss = tss - wcss

    fig, axs = plt.subplots(2,1, constrained_layout=True)
    axs[0].plot(K, avgWithinSS, 'o-')
    axs[0].set_title('Average within-cluster sum of squares')
    axs[0].set_xlabel('Number of clusters')
    axs[0].set_ylabel('avgWithinSS')
    fig.suptitle('Elbow Curve for finding K value', fontsize=16)

    ## 분산 ## 
    axs[1].plot(K, bss/tss*100, '--')
    axs[1].set_title('Analysis of variance')
    axs[1].set_xlabel('Number of clusters')
    axs[1].set_ylabel('variance explained(%)')
    plt.show()
    return ss

ss = k_search() # k -- > 구별 25 / 100로 진행

# n_cluster = 424, max_iter=3000 #
k_means = KMeans(n_clusters=K, max_iter=3000, random_state=77)
k_means.fit(X)
k_cluster = k_means.predict(X)
x_test['k_cluster'] = k_cluster

ss = silhouette_score(X, k_means.labels_, metric='euclidean')
print(ss) # 0.40


# 한글 폰트 깨지는 문제 #
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# 시각화 그래프 # --> 구별 평균 추정값 (TEST) // 그냥 거리만 나타낸 시각화
# fig = plt.figure()
# for i in range(K):
#     scat = plt.scatter(x_test[x_test['k_cluster']==i].iloc[:, 14], x_test[x_test['k_cluster']==i].iloc[:, 15])
#
# fig.show()


### 센터 접근성 분석 ###
### k-means 클러스터를 이용해, 예측한 인구에 따른 '센터'의 접근성 분석 ###
center = k_means.cluster_centers_ # 200개의 클러스터
groupby = x_test.sort_values(['k_cluster'])

def distance(a, b):
    ## 좌표계 사이의 거리를 km 계산으로 ##
    lon1, lat1 = a[0], a[1]
    lon2, lat2 = float("%.6f" %b[0]), float("%.6f" %b[1])
    R = 6378.137 #// radius of the earth in km
    dlat = (lat2 - lat1) * (np.pi / 180)
    dlon = (lon2 - lon1) * (np.pi / 180)
    a = np.sin((dlat/2))**2 + np.cos(lat1 * np.pi / 180) * np.cos(lat2 * np.pi / 180) * (np.sin(dlon/2))**2
    c = 2 * np.math.atan2(np.sqrt(a), np.sqrt(1-a))
    d = R * c
    return d

def center_access():
    global k_means, center, K, groupby
#    pop = groupby[['201809']] # (1412, 1)
    groupby['center_access'] = 0.01
    for j in range(len(groupby)):
        tmpList = []
        for i in range(len(center)):
            gb = groupby[groupby['k_cluster'] == i]
            e = np.int(np.mean(gb['201809']))
            dist = distance(groupby.iloc[j, 14:16].values, center[i])
            tmpList.append(e * (dist*1000) ** -1)
        groupby.iloc[j, -1] = np.sum(tmpList)
#    groupby['mean'] = groupby['center_access'] / K

def people_access():
    global k_means, center, K, groupby
    center = pd.DataFrame(center)
    center['people_access'] = 0.01
    for j in range(len(center)):
        tmpList = []
        for i in range(len(groupby)):
            center_acc = groupby['center_access'].iloc[i]
            limit = groupby['Max'].iloc[i]
            dist = distance(groupby.iloc[i, 14:16].values, center.iloc[j, :-1].values)
            tmpList.append((limit * (dist*1000) ** -1) / (center_acc))
        center.iloc[j, -1] = np.sum(tmpList)
        
        
center_access()
people_access()

groupby['people_access'] = 0
for i in range(len(groupby)):
    groupby.iloc[i, -1] = center['people_access'][groupby['k_cluster'].iloc[i]]
    
    
#groupby.to_csv("d:/project_data/KK_k150.csv", encoding="euc-kr")
print(groupby)
