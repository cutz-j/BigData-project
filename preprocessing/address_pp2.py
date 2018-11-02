### 구주소를 행정동으로 변환 ###
import pandas as pd
import numpy as np

## 데이터 load ##
all_data = pd.read_csv("d:/project_data/all_data2.csv", encoding="euc-kr", index_col=0)
all_ad = pd.read_csv("d:/project_data/address_ad.txt", sep="|", encoding="cp949", header=None)

# 휴지 상태 제거 #
all_data = all_data[all_data['Open'] == '정상']
all_data['Code'] = '0' + all_data['Code'].astype(np.str)

# 중복 행 제거 #
all_ad[3] = '0' + all_ad[3].astype(np.str) # 0 추가
idx = all_ad[3].duplicated()
all_ad = all_ad[idx==False]

# 구주소를 행정동 주소로 변환 ##
for i in range(len(all_data['old_add'])):
    try:
        all_data['old_add'].iloc[i] = all_ad[2][all_ad[3] == all_data['Code'].iloc[i]].iloc[0]
        print(all_data['old_add'].iloc[i])
    except: 
        print("error: ", all_data['old_add'].iloc[i])
        continue


all_data.to_csv("d:/project_data/all_data3.csv", encoding="euc-kr")
