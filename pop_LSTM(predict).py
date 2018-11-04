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

## parameter ##
seq_length = 5 # 데이터의 시퀀스 length (연관된 데이터)  -> output row
data_dim = 1 # 입력 차원 --> 인구수 1 (동별)
output_dim = 1 # 출력 차원 --> 예측치 1
#hidden_size = 30 # 셀 연산 후 나오는 output col
learning_rate = 0.1
iteration = 5000
m = 105 # --> None
MSE_list = []
predict_list = []

### 데이터 전처리 ###
all_data = pd.read_csv("d:/project_data/peopleDataAll01.csv", sep=",", encoding='cp949')

## LSTM ##
for k in range(1, 5):
    tf.reset_default_graph()
    test1 = all_data.iloc[:, [k]] # shape(105,1) m = 105
    keep_prob = tf.placeholder(tf.float32)
    # train scaling #
    mm1 = StandardScaler()
    test1 = mm1.fit_transform(test1)
    
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
        
    x_train, y_train = build(test1, seq_length)
    predict_x = test1[-seq_length*2+1:-seq_length+1].reshape(1, seq_length, 1)
    
    ## RNN building ##
    # cell #
    def lstm_cell(hidden_size):
        cell = tf.nn.rnn_cell.LSTMCell(num_units=hidden_size, activation=tf.tanh)
        return cell
    
    cell1 = lstm_cell(10)


#    cell2 = rnn.DropoutWrapper(lstm_cell(5), input_keep_prob=keep_prob, output_keep_prob=keep_prob, seed=77)
#    cell3 = rnn.DropoutWrapper(lstm_cell(1), input_keep_prob=keep_prob, output_keep_prob=keep_prob, seed=77)
    
    
    #cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True, activation=tf.tanh)
    cell = rnn.MultiRNNCell([cell1], state_is_tuple=True) # dropout cell 5개
    X = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, data_dim])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    #
    ## 초기화 #
    output, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32) 
    Y_pred = tf.contrib.layers.fully_connected(output[:, -1], output_dim, activation_fn=None) # last cell output --> 15일 뒤
    Y_predict = tf.contrib.layers.fully_connected(output, output_dim, activation_fn=None)
    
    # cost #
    cost = tf.reduce_sum(tf.square(Y_pred - y)) # sum of sq --> 수치 예측이기 때문에 sq loss가 필요 없다.
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = opt.minimize(cost)
    
    # MSE # --> mean squared error
    targets= tf.placeholder(tf.float32, [None, 1])
    predicts = tf.placeholder(tf.float32, [None, 1])
    MSE = tf.sqrt(tf.reduce_mean(tf.square(predicts - targets)))
    
    ## session ##
    # training#
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    for i in range(iteration):
        cost_val, _, out= sess.run([cost, train, output], feed_dict={X: x_train, y: y_train, keep_prob: 1.0})
#        if i % 100 == 0:
#            print(cost_val)
    
    # predict #
    predict_1810 = sess.run(Y_pred, feed_dict={X: predict_x, keep_prob: 1.0}).reshape(1, 1, 1)
    predict_con = np.concatenate((predict_x[:, 1:, :], predict_1810), axis=1)
    predict_1810 = predict_1810.reshape(1, 1)
    test1 = np.concatenate((test1, predict_1810))
    predict_1811 = sess.run(Y_pred, feed_dict={X: predict_con, keep_prob: 1.0}).reshape(1, 1, 1)
    predict_con = np.concatenate((predict_con[:, 1:, :], predict_1811), axis=1)
    predict_1811 = predict_1811.reshape(1, 1)
    test1 = np.concatenate((test1, predict_1811))
    predict_1812 = sess.run(Y_pred, feed_dict={X: predict_con, keep_prob: 1.0}).reshape(1, 1, 1)
    predict_con = np.concatenate((predict_con[:, 1:, :], predict_1812), axis=1)
    predict_1812 = predict_1812.reshape(1, 1)
    test1 = np.concatenate((test1, predict_1812))
    predict_1901 = sess.run(Y_pred, feed_dict={X: predict_con, keep_prob: 1.0}).reshape(1, 1, 1)
    predict_con = np.concatenate((predict_con[:, 1:, :], predict_1901), axis=1)
    predict_1901 = predict_1901.reshape(1, 1)
    test1 = np.concatenate((test1, predict_1901))
    predict_1902 = sess.run(Y_pred, feed_dict={X: predict_con, keep_prob: 1.0}).reshape(1, 1, 1)
    predict_con = np.concatenate((predict_con[:, 1:, :], predict_1902), axis=1)
    predict_1902 = predict_1902.reshape(1, 1)
    test1 = np.concatenate((test1, predict_1902))
    predict_1903 = sess.run(Y_pred, feed_dict={X: predict_con, keep_prob: 1.0}).reshape(1, 1, 1)
    predict_con = np.concatenate((predict_con[:, 1:, :], predict_1903), axis=1)
    predict_1903 = predict_1903.reshape(1, 1)
    test1 = np.concatenate((test1, predict_1903))
    predict_1904 = sess.run(Y_pred, feed_dict={X: predict_con, keep_prob: 1.0}).reshape(1, 1, 1)
    predict_con = np.concatenate((predict_con[:, 1:, :], predict_1904), axis=1)
    predict_1904 = predict_1904.reshape(1, 1)
    test1 = np.concatenate((test1, predict_1904))
    predict_1905 = sess.run(Y_pred, feed_dict={X: predict_con, keep_prob: 1.0}).reshape(1, 1, 1)
    predict_con = np.concatenate((predict_con[:, 1:, :], predict_1905), axis=1)
    predict_1905 = predict_1905.reshape(1, 1)
    test1 = np.concatenate((test1, predict_1905))
#    predict_x2 = predict_1909.reshape(seq_length, 1)
#    predict_2002 = sess.run(Y_predict, feed_dict={X: predict_1909, keep_prob: 1.0})
#    predict_x3 = predict_2002.reshape(seq_length, 1)
#    predict_2007 = sess.run(Y_predict, feed_dict={X: predict_2002, keep_prob: 1.0})
#    predict_x4 = predict_2007.reshape(seq_length, 1)
#    predict_2012 = sess.run(Y_predict, feed_dict={X: predict_2007, keep_prob: 1.0})
#    predict_x5 = predict_2012.reshape(seq_length, 1)
#    predict_2105 = sess.run(Y_pred, feed_dict={X: predict_2012, keep_prob: 1.0})
#    predict_x6 = predict_2105.reshape(seq_length, 1)
    
    sess.close()
#    plt.figure()
#    plt.plot(y_train, 'r-')
#    plt.plot(y_hat_train, 'b-')
#    plt.show()
#   
    if k % 1 == 0:
        data_concat = mm1.inverse_transform(test1)
        plt.figure()
        plt.plot(data_concat, 'r-')
        plt.show()

#plist = pd.DataFrame(predict_list)
#plist.to_csv("d:/project_data/pop_test1.csv")