### k-means & 접근성 분석 ###
import numpy as np
import pandas as pd
from sklearn.cluster import KMeans
from scipy.spatial.distance import cdist, pdist

np.random.seed(777)

## 어린이집 데이터 전처리 ##
all_center = pd.read_csv("d:/project_data/all_data_edit.csv", sep=",", encoding="euc-kr")
x_test = all_center[all_center['Type'] == "국공립"] # 국공립만 선택
X = x_test.iloc[:, 14:16]

## k-means ##
K = 150

# n_cluster = 150, max_iter=3000 #
k_means = KMeans(n_clusters=K, max_iter=3000, random_state=77)
k_means.fit(X)
k_cluster = k_means.predict(X)
x_test['k_cluster'] = k_cluster
 
### 센터 접근성 분석 ###
### k-means 클러스터를 이용해, 예측한 인구에 따른 '센터'의 접근성 분석 ###
center = k_means.cluster_centers_ # 150개의 클러스터
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
    groupby[center_col] = 0.01
    xy = np.array(groupby.iloc[:, 14:16])
    center_xy = np.array(center.iloc[:, 0:2])
    tmp = np.zeros_like(groupby[center_col])
    for j in range(len(groupby)):
        if j % 100 == 0: print("center continue.. %d / %d" %(j, len(groupby)))
        tmpList = []
        for i in range(len(center)):
            gb = groupby[groupby['k_cluster'] == i]
            e = np.int(np.mean(gb[pop]))
            dist = distance(xy[j], center_xy[i])
            tmpList.append(e * (dist*1000) ** -1)
        tmp[j] = np.sum(tmpList)
        groupby[center_col] = tmp

def people_access(people_col, center_col):
    global k_means, center, K, groupby
    center[people_col] = 0.01
    groupby[people_col] = 0.01
    xy = np.array(groupby.iloc[:, 14:16])
    center_xy = np.array(center.iloc[:, 0:2])
    tmp = np.zeros_like(center[people_col])
    for j in range(len(center)):
        if j % 100 == 0: print("people continue..%d / %d" %(j, len(center)))
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

groupby.to_csv("d:/project_data/test.csv", encoding="euc-kr", index=None)
