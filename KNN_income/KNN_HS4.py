### 공시지가/아파트실거래가/연립주택실거래가 K-NN ###
import numpy as np
import pandas as pd
from sklearn.metrics import accuracy_score, classification_report
import sklearn.neighbors as neg
import matplotlib.pyplot as plt
import json
import sklearn.preprocessing as pp

import warnings
warnings.filterwarnings('ignore')

## 데이터 전처리 ##  --> 이상치 제거, 표준화 필요 ##
all_data = pd.read_csv("d:/project_data/house_clean02.csv", dtype=np.str, encoding='euc-kr') # encodig: 'euc-kr'

# 면적 당 공시지가 추가 # --> string type이므로 astype을 통해 타입 변경
all_data['y_price'] = all_data['공시지가'].astype(np.float32) / all_data['면적'].astype(np.float32)

# X: (x, y) / y: (면적 당 공시지가) #
X = all_data.iloc[:, 9:11].astype(np.float32) # shape (28046, 2)
y = all_data['y_price'] # shape (28046, )

## Robust scaling ## --> 이상치를 반영한 정규화(min-max)
rs = pp.RobustScaler()
y_scale = rs.fit_transform(np.array(y).reshape(-1, 1))

## 실거래가 아파트 데이터 전처리 ## --> shape (281684, 7)
all_data_apt = pd.read_csv("d:/project_data/total_Apt.csv", sep=",", encoding='euc-kr')
all_data_apt['price_big'] = all_data_apt['Price'] / all_data_apt['Howbig']
X_apt = all_data_apt.iloc[:, -3:-1] # shape (281684, 2)
y_apt_scale = rs.fit_transform(np.array(all_data_apt['price_big']).reshape(-1, 1)) # shape(281684, 1)

## 실거래가 연립 데이터 전처리 ## --> shape ()
all_data_town = pd.read_csv("d:/project_Data/total_Townhouse01.csv", sep=",", encoding="cp949")
all_data_town['price_big'] = all_data_town['Price'] / all_data_town['Howbig']
X_town = all_data_town.iloc[:, -3:-1] # shape (281684, 2)
y_town_scale = rs.fit_transform(np.array(all_data_town['price_big']).reshape(-1, 1)) # shape(281684, 1)

## 어린이집 데이터 전처리 ##
all_center = pd.read_csv("d:/project_data/all_center9.csv", encoding="euc-kr")

# 특정 열만 선택 #
x_test = all_center[all_center['Type'] == "국공립"] # 국공립만 선택

## KNN regressor##
k_list = [25]

# minkowski --> p = 2  // 평균 회귀 --> regressor #
knn_fit = neg.KNeighborsRegressor(n_neighbors=k_list[0], p=2, metric='minkowski')
knn_fit.fit(X, y_scale)
knn_fit.fit(X_apt, y_apt_scale)

## predict --> 평균가 적용 ##
#pred = knn_fit.predict(x_test.iloc[:, 1:3])
#x_test['income'] = pred
for i in range(len(x_test['Gue'])):
    x_test['Gue'].values[i] = x_test['Gue'].values[i][:-1] # '구' 빼기
    
## groupby를 통해 구별 평균 소득 추정 ##
mean = x_test.groupby(['Gue'], as_index=False).mean()

# 한글 폰트 깨지는 문제 #
from matplotlib import font_manager, rc
font_name = font_manager.FontProperties(fname="c:/Windows/Fonts/malgun.ttf").get_name()
rc('font', family=font_name)


## k마다 평균추정치 비교 ## main 코드 --> 구별 평균치 추정
# 공시지가 & 아파트 실거래가 & 연립주택 실거래가에 따라 순위가 바뀜 #
plt.figure()
sortList = []
for i in range(len(k_list)):
    knn_fit = neg.KNeighborsRegressor(n_neighbors=k_list[i], p=2, metric='minkowski')
    knn_fit.fit(X, y_scale)
    knn_fit.fit(X_apt, y_apt_scale)
    knn_fit.fit(X_town, y_town_scale)
    x_test["predK%i" %k_list[i]] = knn_fit.predict(x_test.iloc[:, 14:16])
    mean = x_test.groupby(['Gue'], as_index=False).mean()
    price_pred = pd.DataFrame(mean.iloc[:, -1])
    price_pred.index = mean['Gue']
    sortList.append(price_pred)
    plt.plot(price_pred)
plt.legend(k_list)
plt.rcParams['axes.grid'] = True
plt.rcParams["figure.figsize"] = (16,4)
plt.show()

test1 =knn_fit.predict(np.array([[126.835069, 37.488801]])) # array([[-0.3405183]])
test2 = knn_fit.predict(np.array([[126.820568, 37.481623]])) # array([[0.04430948]])
test3 = knn_fit.predict(np.array([[127.173252, 37.566563]])) # array([[0.28846714]])
test4 = knn_fit.predict(np.array([[127.178065, 37.567504]])) # array([[0.11665704]])
knn_fit.predict(np.array([[127.051929,	37.678604]])) # array([[0.11665704]])


#x_test.to_csv("d:/project_data/KNN_data.csv", encoding='euc-kr', index=False)

