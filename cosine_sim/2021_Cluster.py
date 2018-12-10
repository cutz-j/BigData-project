import pandas as pd
import numpy as np

all_data = pd.read_csv("d:/project_data/test/test_1.csv", sep=",", encoding='cp949')

def income_cluster(col):
    quarter = range(25,101,25)
    k = 0
    p = np.min(col)
    for i in quarter:
        q = np.percentile(col, i)
        idx = all_data[all_data['predK25'] <= q][all_data['predK25'] > p].index
        for j in idx:
            all_data.iloc[j, -1] = k
        k += 1
        p = q

def center_cluster(all_data, colname, new_col):
    mean = all_data[colname].groupby(all_data['old_add']).mean()
    mean = mean.sort_values()
    for i in range(len(mean)):
        mean[i] = i
    for i in range(len(all_data)):
        all_data.iloc[i, -1] = int(mean[all_data['old_add'][i]])
        
def people_cluster(colname, new_col):
    global all_data
    sort = all_data[colname].sort_values().index
    k = 0
    j = all_data[colname][sort[0]]
    for i in sort:
        if all_data[colname][i] == j:
            all_data.iloc[i, -1] = k
        else:
            k += 1
            all_data.iloc[i, -1] = k
        j = all_data[colname][i]
        

all_data['income_cluster_test'] = 0
income_cluster(all_data['predK25'])
all_data['center_cluster1_test'] = 0
center_cluster(all_data, 'center_access', 'center_cluster1_test')
all_data['people_cluster1_test'] = 0
people_cluster('people_access', 'people_cluster1_test')
all_data['center_cluster2_test'] = 0
center_cluster(all_data, 'center_access_2', 'center_cluster2_test')
all_data['people_cluster2_test'] = 0
people_cluster('people_access_2', 'people_cluster2_test')

#all_data = all_data.sort_values(by='income_cluster')
#income_bot30 = all_data.iloc[:30, :]
#income_name = income_bot30['Name']
#all_data = all_data.sort_values(by='center_cluster1')
#cc1_bot30 = all_data.iloc[:30, :]
#cc1_name = cc1_bot30['Name']
#all_data = all_data.sort_values(by='people_cluster1')
#pc1_bot30 = all_data.iloc[:30, :]
#pc1_name = pc1_bot30['Name']
#all_data = all_data.sort_values(by='center_cluster2')
#cc2_bot30 = all_data.iloc[:30, :]
#cc2_name = cc2_bot30['Name']
#all_data = all_data.sort_values(by='people_cluster2')
#pc2_bot30 = all_data.iloc[:30, :]
#pc2_name = pc2_bot30['Name']
#
#bot30_name = pd.concat((income_name, cc1_name, pc1_name, cc2_name, pc2_name), axis=1)
#
#
#
#bot30_name.to_csv("d:/project_data/bot30_name.csv", encoding='cp949')
#pc1_bot30.to_csv("d:/project_data/pc1_bot30.csv", encoding='cp949')
#cc1_bot30.to_csv("d:/project_data/cc1_bot30.csv", encoding='cp949')
#pc2_bot30.to_csv("d:/project_data/pc2_bot30.csv", encoding='cp949')
#cc2_bot30.to_csv("d:/project_data/cc2_bot30.csv", encoding='cp949')
#
#
#all_data.to_csv("d:/project_data/test/test2(상계1동, 강일동).csv", encoding='cp949')
