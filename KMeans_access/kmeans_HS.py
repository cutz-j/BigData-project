### k-means & 접근성 분석 ###
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.cluster import KMeans
from sklearn.metrics import silhouette_score
from scipy.spatial.distance import cdist, pdist

np.random.seed(777)

## 어린이집 데이터 전처리 ##
all_center = pd.read_csv("d:/project_data/test/all_test_1.csv", sep=",", encoding="euc-kr")
x_test = all_center[all_center['Type'] == "국공립"] # 국공립만 선택
X = x_test.iloc[:-15, 15:17]
X_test = x_test.iloc[:, 15:17]


## train_test split ## --> train: x_test / predict: x_test (cluster)

## k-means ##
# 최적 클러스터 k 찾기 # -> elbow graph
K = 150

def k_search():
    ## 최적의 k를 엘보 그래프로 찾는 함수 ##
    global K
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
    ## 엘보 곡선 ## ----> 시각화 하세요~!
    fig = plt.figure()
    ax = fig.add_subplot(111)
    ax.plot(K, avgWithinSS, 'b*-')
    plt.grid(True)
    plt.xlabel("# of clusters")
    plt.ylabel("Avg of SS")
    ## 분산 ## ----> 이것도 ~!
    fig2 = plt.figure()
    ax = fig2.add_subplot(111)
    ax.plot(K, bss/tss*100, 'b*-')
    plt.grid(True)
    plt.xlabel("# of cluster")
    plt.ylabel("var")
    plt.show()
    return ss

#ss = k_search() # k -- > 구별 25 / 100로 진행

# n_cluster = 150, max_iter=3000 #
k_means = KMeans(n_clusters=K, max_iter=3000, random_state=77)
k_means.fit(X)
k_cluster = k_means.predict(X_test)
x_test['k_cluster'] = k_cluster

ss = silhouette_score(X, k_means.labels_, metric='euclidean')
#print(ss) # 0.40


# 한글 폰트 깨지는 문제 #
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)

# 시각화 그래프 # --> 구별 평균 추정값 (TEST) // 그냥 거리만 나타낸 시각화
#fig = plt.figure()
#for i in range(K):
#    scat = plt.scatter(x_test[x_test['k_cluster']==i].iloc[:, 14], x_test[x_test['k_cluster']==i].iloc[:, 15],
#                       s=10)
#    
#
#fig.show()

### 
### 센터 접근성 분석 ###
### k-means 클러스터를 이용해, 예측한 인구에 따른 '센터'의 접근성 분석 ###
center = k_means.cluster_centers_ # 200개의 클러스터
center = pd.DataFrame(center)
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

def center_access(center_col, pop):
    global k_means, center, K, groupby
#    pop = groupby[['201809']] # (1412, 1)
    groupby[center_col] = 0.01
    xy = np.array(groupby.iloc[:, 15:17])
    center_xy = np.array(center.iloc[:, 0:2])
    tmp = np.zeros_like(groupby[center_col])
    for j in range(len(groupby)):
        if j % 100 == 0: print("center continue..")
        tmpList = []
        for i in range(len(center)):
            gb = groupby[groupby['k_cluster'] == i]
            e = np.int(np.mean(gb[pop]))
            dist = distance(xy[j], center_xy[i])
            tmpList.append(e * (dist*1000) ** -1)
        tmp[j] = np.sum(tmpList)
        groupby[center_col] = tmp
#    groupby['mean'] = groupby['center_access'] / K

def people_access(people_col, center_col):
    global k_means, center, K, groupby
    center[people_col] = 0.01
    xy = np.array(groupby.iloc[:, 15:17])
    center_xy = np.array(center.iloc[:, 0:2])
    tmp = np.zeros_like(center[people_col])
    for j in range(len(center)):
        if j % 100 == 0: print("people continue..")
        tmpList = []
        for i in range(len(groupby)):
            center_acc = groupby[center_col].iloc[i]
            limit = groupby['Max'].iloc[i]
            dist = distance(xy[i], center_xy[j])
            tmpList.append((limit * (dist*1000) ** -1) / (center_acc))
        tmp[j] = np.sum(tmpList)
    center[people_col] = tmp
    for i in range(len(groupby)):
        groupby[people_col].iloc[i] = center[people_col][groupby['k_cluster'].iloc[i]]       
        
center_access('center_access', '201809')
people_access('people_access', 'center_access')
center_access('center_access_2', '202104')
people_access('people_access_2', 'center_access_2')

groupby.to_csv("d:/project_data/test/test_1.csv", encoding="euc-kr", index=0)
#