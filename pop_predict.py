### 0~5세 서울시 동별 인구 학습 & 예측 ###
### 딥러닝 기반 예측 모델 ###
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt

# shape(54, 423)
all_data = pd.read_csv("d:/project_data/people_data_all.csv", sep=",", encoding='euc-kr')

