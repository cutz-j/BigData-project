### 0~5세 서울시 동별 인구 학습 & 예측 ###
### 딥러닝 기반 예측 모델 ###
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import StandardScaler

tf.set_random_seed(777)

## 데이터 전처리 ##
# shape(54, 423)

all_data = pd.read_csv("d:/project_data/peopleDataAll01.csv", sep=",", encoding='cp949')
#all_center = pd.read_csv("d:/project_data/all_data3.csv", sep=",", encoding="euc-kr")
#
## 201809 인구데이터 뽑아오기 #
#recent_data = all_data.iloc[-1, :]
#index_list = []
#for i in recent_data.index:
#    index_list.append(i.split()[1])
#
#recent_data.index = index_list
#all_center['old_add'][all_center['old_add']=='공릉1동'] = '공릉1.3동'
#all_center['old_add'][all_center['old_add']=='위례동'] = '장지동'
#
#
### center data에 2018년 9월 인구 데이터 붙이기 ##
#all_center['201809'] = 0
#for i in range(len(all_center['201809'])):
#    try: all_center['201809'].iloc[i] = recent_data[all_center['old_add'].iloc[i]]
#    except: print("error: ", all_center['old_add'].iloc[i])

#all_center.to_csv("d:/project_data/all_center9.csv", encoding="euc-kr")

# X, Y 데이터 전처리 #
x_data = np.array(all_data.index, dtype=np.float32).reshape(105, 1) + 1.001 # shape(105, 1)
Y_data = np.array(all_data, dtype=np.float32) # shape (105, 423)

# standard scaling #
ss = StandardScaler()
ss_x = StandardScaler()
Y_scale = ss.fit_transform(Y_data) # fit 데이터를 이용해 복원 --> ss.inverse_transform(Y_scale)
x_scale = ss_x.fit_transform(x_data)
# train_test split 8:2 #
train_size = int(len(x_data) * 0.8)
x_train, x_test, y_train, y_test = x_scale[:84], x_scale[84:], Y_scale[:84, :], Y_scale[84:, :]

## hyper paramter ##
learning_rate = 0.003
l2norm = 0.0001
epochs = 10000
batch_size = 42
is_training = True  # 배치 정규화를 위한 boollean
keep_prob = 0.7

## tf building ## + ## tensor graph ##
#tf.reset_default_graph()
X = tf.placeholder(dtype=tf.float32, shape=[None, 1])
Y = tf.placeholder(dtype=tf.float32, shape=[None, 423])

## batch-normalization ##  --> wx+B의 계산마다 정규화가 이루어져, 학습 개선
# 다중신경망 구성 #
# drop-out # 학습 노드를 임의로 제외하여, 오버피팅 방지 --> 앙상블 학습 구현
init = tf.contrib.layers.xavier_initializer(seed=77)
W1 = tf.Variable(init([1, 2115]), name='weight1')
b1 = tf.Variable(init([2115]), name='bias1')
layer1 = tf.matmul(X, W1) + b1
l1 = tf.contrib.layers.batch_norm(layer1, center=True, scale=True,
                                  is_training=is_training)
L1 = tf.nn.tanh(l1, name='relu1')
L1 = tf.nn.dropout(L1, keep_prob=keep_prob)

W2 = tf.Variable(init([2115, 1269]), name='weight2')
b2 = tf.Variable(init([1269]), name='bias2')
layer2 = tf.matmul(L1, W2) + b2
l2 = tf.contrib.layers.batch_norm(layer2, center=True, scale=True,
                                  is_training=is_training)
L2 = tf.nn.tanh(l2, name='relu2')
L2 = tf.nn.dropout(L2, keep_prob=keep_prob)

W3 = tf.Variable(init([1269, 846]), name='weight3')
b3 = tf.Variable(init([846]), name='bias3')
layer3 = tf.matmul(L2, W3) + b3
l3 = tf.contrib.layers.batch_norm(layer3,  center=True, scale=True,
                                  is_training=is_training)
L3 = tf.nn.tanh(l3, name='relu3')
L3 = tf.nn.dropout(L3, keep_prob=keep_prob)

W4 = tf.Variable(init([846, 423]), name='weight4')
b4 = tf.Variable(init([423]), name='bias4')
hypothesis = tf.matmul(L3, W4) + b4

## l2 정규화 = 리지회귀(W값 패널티를 cost에서 더해준다.)
var = tf.trainable_variables()
l2reg = tf.add_n([tf.nn.l2_loss(v) for v in var if 'bias' not in v.name]) * l2norm

# cost&opt #
cost = tf.reduce_mean(tf.square(hypothesis - Y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # batch_norm
with tf.control_dependencies(update_ops):
    train = optimizer.minimize(cost)

gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.333)
sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
sess.run(tf.global_variables_initializer())
#sess.run(is_training, feed_dict={is_training: True})

for i in range(epochs):
    cost_val, _ = sess.run([cost, train], feed_dict={X: x_train, Y: y_train})
    if i % 50 == 0:
        print(cost_val)

is_training = False
keep_prob = 1.0
y_hat = sess.run(hypothesis, feed_dict={X: x_test})
print("test cost: ", sess.run([cost], feed_dict={X: x_test, Y:y_test}))  
        
#coord = tf.train.Coordinator()
#threads = tf.train.start_queue_runners(sess=sess, coord=coord)

#shuffle_batch
#min_after_dequeue = 10000
#capacity = min_after_dequeue + 3 * batch_size
#train_x_batch, train_y_batch = tf.train.shuffle_batch([x, Y], batch_size=batch_size,
#                                                      capacity=capacity, min_after_dequeue=min_after_dequeue,
#                                                      shapes=[(84, 1), (84, 423)])

#for epoch in range(epochs):
#    avg_cost = 0
#    total_batch = int(84 / batch_size) # batch 학습
#    for i in range(total_batch):
#        x_batch, y_batch = sess.run([train_x_batch, train_y_batch])
#        cost_val, _ = sess.run([cost, train], feed_dict={x: x_train, Y: y_train, is_training: True})
#        avg_cost += cost_val / total_batch
#        print(avg_cost)
#        
#    print('Epoch:', '%04d' % (epoch + 1), 'cost =', '{:.9f}'.format(avg_cost))
#
#coord.request_stop()
#coord.join(threads)
sess.close()
