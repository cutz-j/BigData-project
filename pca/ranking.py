### PCA ###
import numpy as np
import pandas as pd
from sklearn.decomposition import TruncatedSVD
from sklearn.preprocessing import MinMaxScaler

## 데이터 불러오기 ##
rent = pd.read_csv("d:/project_data/KNN_data_rent.csv", encoding='euc-kr')
all_data = pd.read_csv("d:/project_data/KK_k150_2021.csv", sep=",", encoding='cp949')
data_new = pd.read_csv("d:/project_data/test.csv", sep=",", encoding='euc-kr')

all_data['add_name'] = all_data['Name'] + all_data['Add']
data_new['add_name'] = data_new['Name'] + data_new['Add']

# 전월세 데이터 추가 #

def data_concat(data1, data2):
    ## 함수화 ##
    rent_name = data2[['Name','Add','predK25']]
    rent_name['add_name'] = rent_name['Name'] + rent_name['Add']
    rent_name1 = rent_name[['add_name', 'predK25']]
    rent_name1.columns = ['add_name', 'rent']
    data1 = pd.merge(data1, rent_name1, on='add_name')
    X = data1[['predK25', 'rent', 'center_access', 'people_access', 'center_access_2', 'people_access_2']]
    y = data1[['Name']]
    return X, y

def scale_svd(X, Y):
    ## 함수화 ##
    ss = MinMaxScaler()
    X_scale = ss.fit_transform(X)
    ### SVD ###
    svd = TruncatedSVD(n_components=3)
    X_svd = svd.fit_transform(X_scale, y=Y)
    #print(np.sum(svd.explained_variance_ratio_))
    
    X_new = np.mean(X_svd, axis=1)
    Y['new'] = X_new
    return Y

def sort_rank(y):
    y_old_sort = y.sort_values(['new'])
    y_old_sort['rank'] = y_old_sort['new'].rank() / len(y_old_sort)
    return y_old_sort

X_old, y = data_concat(all_data, rent)
y_old = scale_svd(X_old, y)
X_new, y_new = data_concat(data_new, rent)
y_new_svd = scale_svd(X_new, y_new)
y_new_svd = y_new_svd.drop_duplicates()
y_old_rank = sort_rank(y_old)
y_new_rank = sort_rank(y_new_svd)

y_new_rank.to_csv("d:/project_data/y_new.csv", encoding='euc-kr', index=None)
y_old_rank.to_csv("d:/project_data/y_old.csv", encoding='euc-kr', index=None)
