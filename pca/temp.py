import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler
from sklearn.decomposition import TruncatedSVD

## data load ##
rent = pd.read_csv("d:/data/KNN_data_rent.csv", 
                   encoding='euc-kr')
all_data = pd.read_csv("d:/data/KK_k150_2021.csv",
                       encoding='euc-kr')
data_new = pd.read_csv("d:/data/test.csv", 
                       encoding='euc-kr')
all_data['add_name'] = all_data['Name'] + all_data['Add']
data_new['add_name'] = data_new['Name'] + data_new['Add']

def data_concat(data1, data2):
    ''' 기존 데이터와 전월세 추정 데이터를 mapping해서 concat '''
    rent_name = data2[['Name','Add','predK25']]
    rent_name['add_name'] = rent_name['Name'] + rent_name['Add']
    rent_name1 = rent_name[['add_name', 'predK25']]
    rent_name1.columns = ['add_name', 'rent']
    data1 = pd.merge(data1, rent_name1, on='add_name')
    X = data1[['predK25', 'center_access', 'people_access',
                 'center_access_2', 'people_access_2', 'rent']]
    y = data1[['Name']]
    return X, y

def scale_svd(X, y):
    '''## scaling + svd + mean ##'''
    ms = MinMaxScaler()
    X_scale = ms.fit_transform(X)
    
    ## svd ##
    svd = TruncatedSVD(n_components=3, random_state=77)
    X_svd = svd.fit_transform(X_scale)
    np.sum(svd.explained_variance_ratio_)
    
    X_svd = pd.DataFrame(X_svd)
    
    avg = np.mean(X_svd, axis=1)
    y['new'] = avg
    return y

X_old, y_old = data_concat(all_data, rent)
y_old_svd = scale_svd(X_old, y_old)

def sort_rank(y):
    y_old_sort = y.sort_values(by=['new'])
    y_old_sort['rank'] = y_old_sort['new'].rank() / len(y_old_sort) # rank / 1412
    return y_old_sort

y_old_rank = sort_rank(y_old_svd)

X_new, y_new = data_concat(data_new, rent)
y_new_svd = scale_svd(X_new, y_new)
y_new_rank = sort_rank(y_new_svd)













