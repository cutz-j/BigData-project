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
rent_price = rent[['predK25']]
all_data['rent'] = rent_price

## X data ##
X = all_data[['predK25', 'rent', 'center_access', 'people_access', 'center_access_2', 'people_access_2']]
y = all_data[['Name']]

## scaling ##
ss = StandardScaler()
X_scale = ss.fit_transform(X)

mm = MinMaxScaler()
X_scale = mm.fit_transform(X)

### SVD ###
svd = TruncatedSVD(n_components=3)
X_svd = svd.fit_transform(X_scale, y=y)
#print(np.sum(svd.explained_variance_ratio_))

X_sq = np.dot(X_scale.T, X_scale) / X_scale.shape[0]
U, S, V = np.linalg.svd(X_sq)
svd_vec = np.dot(X_scale, U[:,:3])

X_new = np.mean(X_svd, axis=1)
y['new'] = X_new
