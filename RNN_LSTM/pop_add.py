import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.set_random_seed(777)

## 데이터 전처리 ##
# shape(54, 423)

all_data = pd.read_csv("d:/project_data/people_2021.csv", sep=",", encoding='cp949')
all_center = pd.read_csv("d:/project_data/KK_k150.csv", sep=",", encoding="euc-kr")
#
# 202104 인구데이터 뽑아오기 #
recent_data = all_data.iloc[-1, :]
index_list = []
for i in recent_data.index:
    index_list.append(i.split()[1])

recent_data.index = index_list
all_center['old_add'][all_center['old_add']=='공릉1동'] = '공릉1.3동'
all_center['old_add'][all_center['old_add']=='위례동'] = '장지동'


## center data에 2019년 4월 인구 데이터 붙이기 ##
all_center['201904'] = 0
for i in range(len(all_center['201904'])):
    try: all_center['201904'].iloc[i] = recent_data[all_center['old_add'].iloc[i]]
    except: print("error: ", all_center['old_add'].iloc[i])

#all_center.to_csv("d:/project_data/all_data_pp.csv", encoding="euc-kr")
