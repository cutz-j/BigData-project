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

time = ["%i"%(i) + "-%i"%(3) for i in range(2010, 2022)]

## parameter ##
seq_length = 5 # 데이터의 시퀀스 length (연관된 데이터)  -> output row
data_dim = 1 # 입력 차원 --> 인구수 1 (동별)
output_dim = 1 # 출력 차원 --> 예측치 1
#hidden_size = 30 # 셀 연산 후 나오는 output col
learning_rate = 0.07
iteration = 8000
m = 105 # --> None
MSE_list = []
pop_2103 = []

### 데이터 전처리 ###
all_data = pd.read_csv("d:/project_data/peopleDataAll01.csv", sep=",", encoding='cp949')

## LSTM ##
for k in [-18]:
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
    
    cell1 = lstm_cell(30)
    cell2 = lstm_cell(20)
    cells = []
    cells.append(cell1)
    cells.append(cell2)
#    cell3 = rnn.DropoutWrapper(lstm_cell(1), input_keep_prob=keep_prob, output_keep_prob=keep_prob, seed=77)
    
    
    #cell = rnn.BasicLSTMCell(num_units=hidden_size, state_is_tuple=True, activation=tf.tanh)
    cell = rnn.MultiRNNCell([cell1, cell2], state_is_tuple=True) # dropout cell 5개
    
    ## tensor board ##
    for one_lstm_cell in cells:
        one_kernel = one_lstm_cell.variables
        tf.summary.histogram("Kernel", one_kernel)
#        tf.summary.histogram("Bias", one_bias)
    
    X = tf.placeholder(dtype=tf.float32, shape=[None, seq_length, data_dim])
    y = tf.placeholder(dtype=tf.float32, shape=[None, 1])
    #
    ## 초기화 #
    output, _state = tf.nn.dynamic_rnn(cell, X, dtype=tf.float32) 
    Y_pred = tf.contrib.layers.fully_connected(output[:, -1], output_dim, activation_fn=None) # last cell output --> 15일 뒤
    
    # cost #
    cost = tf.reduce_sum(tf.square(Y_pred - y)) # sum of sq --> 수치 예측이기 때문에 sq loss가 필요 없다.
    opt = tf.train.AdamOptimizer(learning_rate=learning_rate)
    train = opt.minimize(cost)
    
    # MSE # --> mean squared error
    targets= tf.placeholder(tf.float32, [None, 1])
    predicts = tf.placeholder(tf.float32, [None, 1])
    MSE = tf.sqrt(tf.reduce_mean(tf.square(predicts - targets)))
    
    summary_op = tf.summary.merge_all()
    ## session ##
    # training#
    sess = tf.Session()
    sess.run(tf.global_variables_initializer())
    train_writer = tf.summary.FileWriter("d:/project_data/logdir/", graph=tf.get_default_graph())
    for i in range(iteration):
        cost_val, _, out, step_summary= sess.run([cost, train, output, summary_op], feed_dict={X: x_train, y: y_train, keep_prob: 0.7})
#        if i % 100 == 0:
#            print(cost_val)
        train_writer.add_summary(step_summary)

    # predict # --> 201809 30개월 후 --> 202103
    for t in range(30):
        tmp_arr = sess.run(Y_pred, feed_dict={X: predict_x, keep_prob: 1.0})
        test1 = np.concatenate((test1, tmp_arr))
        predict_x = np.concatenate((predict_x[:, 1:, :], tmp_arr.reshape(1,1,1)), axis=1)

    sess.close()

## 시각화 ##   
    if k % 1 == 0:
        data_concat = mm1.inverse_transform(test1)
        data_concat = pd.DataFrame(data_concat)
        plt.figure(figsize=(16,8))
        plt.plot(data_concat.iloc[:106, :], 'r-')
        plt.plot(data_concat.iloc[105:, :].index, data_concat.iloc[105:, :], 'b-')
        plt.xlabel("Time")
        plt.ylabel("Population")
        plt.xticks(ticks=np.arange(0, 135, 12), labels=list(time))
        plt.show()
    
    pop_2103.append(int(data_concat.iloc[-1][0]))
    
    
    

plist = pd.DataFrame(pop_2103).T
#plist.to_csv("d:/project_data/pop_2103.csv")