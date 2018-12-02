### PCA ###
import numpy as np
import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.decomposition import TruncatedSVD
from sklearn.decomposition import PCA
from sklearn.preprocessing import MinMaxScaler

## 데이터 불러오기 ##
rent = pd.read_csv("d:/project_data/KNN_data_rent.csv", encoding='euc-kr')
all_data = pd.read_csv("d:/project_data/KK_k150_2021.csv", sep=",", encoding='cp949')

# 전월세 데이터 추가 #
rent_name = rent[['predK25']]

all_data['rent'] = 0
for i in range(len(rent_name)):
    all_data.iloc[all_data[all_data['Name'] == rent_name.iloc[i,0]].index[0], -1] = rent_name.iloc[i, 1]

## X data ##
X = all_data[['predK25', 'rent', 'center_access', 'people_access', 'center_access_2', 'people_access_2']]
y = all_data[['Name']]

## scaling ##
ss = MinMaxScaler()
X_scale = ss.fit_transform(X)

### SVD ###
svd = TruncatedSVD(n_components=3)
X_svd = svd.fit_transform(X_scale, y=y)
#print(np.sum(svd.explained_variance_ratio_))

X_sq = np.dot(X_scale.T, X_scale) / X_scale.shape[0]
U, S, V = np.linalg.svd(X_sq)
svd_vec = np.dot(X_scale, U[:,:3])

X_new = np.mean(X_svd, axis=1)
y['new'] = X_new
