### 동별 랭킹과 %를 구하는 모듈 ###
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import MinMaxScaler

all_data = pd.read_csv("d:/project_data/test/test_11.csv", encoding='euc-kr')
#scale = all_data.iloc[:, -5:]

mm = MinMaxScaler()
scale_m = mm.fit_transform(all_data.iloc[:, -5:])
summation = pd.DataFrame(np.mean(scale_m, axis=1))
data = pd.concat((all_data['Name'], all_data['old_add'], summation), axis=1)
mean = data.groupby(['old_add'], as_index=False).mean() # (4, 13, )
mean.columns = ['old_add', 'ranking']
mean = mean.sort_values(by=['ranking'])
mean['rank'] = mean.iloc[:,[-1]].rank() / len(mean) * 100

#mean.to_csv("d:/project_data/test/ranking.csv", encoding="euc-kr")
