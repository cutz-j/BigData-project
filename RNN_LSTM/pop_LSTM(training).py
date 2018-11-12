### prop predict RNN-LSTM ###
## tensor board ##
import tensorflow as tf
import pandas as pd
import numpy as np
import matplotlib.pyplot as plt
from sklearn.preprocessing import StandardScaler
from tensorflow.contrib import rnn
import warnings
warnings.filterwarnings('ignore')

tf.set_random_seed(777)
tf.reset_default_graph()

time = ["%i"%(i) + "-%i"%(j) for i in range(2010, 2022) for j in range(3, 12, 4)]

## parameter ##
seq_length = 10 # 데이터의 시퀀스 length (연관된 데이터)  -> output row
data_dim = 1 # 입력 차원 --> 인구수 1 (동별)
output_dim = 1 # 출력 차원 --> 예측치 1
#hidden_size = 20 # 셀 연산 후 나오는 output col
learning_rate = 0.1
iteration = 8000
m = 105 # --> None
MSE_list = []
predict_list = []
is_training = True
l2norm = 0.0001

### 데이터 전처리 ###
all_data = pd.read_csv("d:/project_data/peopleDataAll01.csv", sep=",", encoding='cp949')

## 청운효자동 LSTM ##
for k in range(1,10):
    tf.reset_default_graph()
    keep_prob = tf.placeholder(dtype=tf.float32)
    test1 = all_data.iloc[:, [k]] # shape(105,1) m = 105
    # train scaling #
    mm1 = StandardScaler()
    test1 = mm1.fit_transform(test1)
    
    ## split ## --> 시계열(시간순)
    train_size = int(len(test1) * 0.8)
    train_set = test1[:train_size, :] # shape(512, 5)
    test_set = test1[train_size:, :] # test(220, 5)

    # RNN data building #
    def build(time_series, seq_length):
        x_data = []
        y_data = []
        for i in range(0, len(time_series) - seq_length):
            x_tmp = time_series[i: i + seq_length, :]
            y_tmp = time_series[i + seq_length, [-1]]
            x_data.append(x_tmp)
            y_data.append(y_tmp)
        return np.array(x_data), np.array(y_data)
    
    x_train, y_train = build(train_set, seq_length)
    x_test, y_test = build(test_set, seq_length)
    predict_x = test_set[-seq_length:].reshape(1, seq_length, 1)
    
    ## RNN building ##
    # cell #
    def lstm_cell(hidden_size):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, activation=tf.tanh)
        return cell

    cell1 = rnn.DropoutWrapper(lstm_cell(15), input_keep_prob=keep_prob, output_keep_prob=keep_prob, seed=77)
    cell2 = rnn.DropoutWrapper(lstm_cell(10), input_keep_prob=keep_prob, output_keep_prob=keep_prob, seed=77)
#    cell3 = rnn.DropoutWrapper(lstm_cell(10), input_keep_prob=keep_prob, output_keep_prob=keep_prob, seed=77)
#    cell4 = rnn.DropoutWrapper(lstm_cell(), input_keep_prob=keep_prob, output_keep_prob=keep_prob, seed=77)
    #cell5 = rnn.DropoutWrapper(lstm_cell(), input_keep_prob=keep_prob, output_keep_prob=keep_prob, seed=77)
    
    #cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True, activation=tf.tanh)
    cell = rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True) # dropout cell 5개
    #
    X = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, data_dim])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    #
    ## 초기화 #
    output, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32) 
    Y_pred = tf.contrib.layers.fully_connected(output[:, -1], output_dim, activation_fn=None) # last cell output --> 15일 뒤
    
#    # config #
#    config = tf.ConfigProto(log_device_placement=True)
#    config.gpu_options.allow_growth = True
    init = tf.contrib.layers.xavier_initializer(seed=77)
    W1 = tf.Variable(init([1, 100]), name='weight1')
    b1 = tf.Variable(init([100]), name='bias1')
    layer1 = tf.matmul(Y_pred, W1) + b1
    l1 = tf.contrib.layers.batch_norm(layer1, center=True, scale=True,
                                      is_training=is_training)
    L1 = tf.nn.relu(l1, name='relu1')
    L1 = tf.nn.dropout(L1, keep_prob=keep_prob)
    
    W2 = tf.Variable(init([100, 1]), name='weight2')
    b2 = tf.Variable(init([1]), name='bias2')
    hypothesis = tf.matmul(L1, W2) + b2
        
    # cost #
    cost = tf.reduce_sum(tf.square(Y_pred - y)) # sum of sq --> 수치 예측이기 때문에 sq loss가 필요 없다.
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = opt.minimize(cost)
    
    ## tf.trainable --> l2 norm ##
    var = tf.trainable_variables()
    l2reg = tf.add_n([tf.nn.l2_loss(v) for v in var if 'bias' not in v.name]) * l2norm
    
    # cost #
    cost = tf.reduce_sum(tf.square(Y_pred - y)) # sum of sq --> 수치 예측이기 때문에 sq loss가 필요 없다.
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    update_ops = tf.get_collection(tf.GraphKeys.UPDATE_OPS) # batch_norm
    with tf.control_dependencies(update_ops):
        train = opt.minimize(cost)
    
    # MSE # --> mean squared error
    targets= tf.placeholder(tf.float32, [None, 1])
    predicts = tf.placeholder(tf.float32, [None, 1])
    MSE = tf.sqrt(tf.reduce_mean(tf.square(predicts - targets)))
    
    ## session ##
    # training#
    gpu_options = tf.GPUOptions(per_process_gpu_memory_fraction=0.7)
    sess = tf.Session(config=tf.ConfigProto(gpu_options=gpu_options))
    sess.run(tf.global_variables_initializer())
    for i in range(iteration):
        cost_val, _, out= sess.run([cost, train, output], feed_dict={X: x_train, y: y_train, keep_prob:0.7})
        if i % 1000 == 0:
            print(cost_val)
        
    # predict #
    is_training = False
    y_hat_train = sess.run(Y_pred, feed_dict={X: x_train, keep_prob:1.0})
    y_hat = sess.run(Y_pred, feed_dict={X: x_test, keep_prob:1.0})
#    y_hat = mm1.inverse_transform(y_hat)
#    y_test = mm1.inverse_transform(y_test)
    y_hat = mm1.inverse_transform(y_hat)
    y_test = mm1.inverse_transform(y_test)
    RMSE_train = sess.run(MSE, feed_dict={targets: y_train, predicts: y_hat_train, keep_prob:1.0})
    RMSE = sess.run(MSE, feed_dict={targets: y_test, predicts: y_hat, keep_prob:1.0})
    print("RMSE_train: ", RMSE_train)
    print("RMSE: ", RMSE)
    predict_hat = sess.run(Y_pred, feed_dict={X: predict_x, keep_prob:1.0})
    
    MSE_list.append(RMSE)
    predict_list.append(mm1.inverse_transform(predict_hat)[0,0])
    plt.figure(figsize=(8,3))
    plt.plot(y_train, 'r-')
    plt.plot(y_hat_train, 'b-')
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.show()
    
    plt.figure(figsize=(8,3))
    plt.plot(y_test, 'r-')
    plt.plot(y_hat, 'b-')
    plt.xlabel("Time")
    plt.ylabel("Population")
    plt.show()
    sess.close()
    sess.close()
        
#plt.figure()
#plt.plot(y_train, 'r-')
#plt.plot(y_hat_train, 'b-')
#plt.show()
#
#plt.figure()
#plt.plot(y_test, 'r-')
#plt.plot(y_hat, 'b-')
#plt.show()

mse = pd.DataFrame(MSE_list)





