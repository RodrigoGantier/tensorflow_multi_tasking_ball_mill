'''
Created on 15 feb. 2017

@author: yasushishibe
'''
'''
Created on 14 feb. 2017

@author: yasushishibe
'''
from sklearn.metrics import mean_squared_error
# from sklearn.manifold import TSNE
import numpy as np
import scipy.io as sio
import tensorflow as tf
import matplotlib.pyplot as plt

def import_data():
    train_x = sio.loadmat("/Users/yasushishibe/Documents/MATLAB/multi_tasking_ball_mill/data/1_data/train_x.mat")
    train_x = np.asarray(train_x['train_x'], dtype = np.float32)
    train_y1 = sio.loadmat("/Users/yasushishibe/Documents/MATLAB/multi_tasking_ball_mill/data/1_data/train_y.mat")
    train_y1 = np.asarray(train_y1['train_y'], dtype = np.float32)
    train_y2 = sio.loadmat("/Users/yasushishibe/Documents/MATLAB/multi_tasking_ball_mill/data/1_data/train_lqby.mat")
    train_y2 = np.asarray(train_y2['train_y'], dtype = np.float32)
    train_y3 = sio.loadmat("/Users/yasushishibe/Documents/MATLAB/multi_tasking_ball_mill/data/1_data/train_ndy.mat")
    train_y3 = np.asarray(train_y3['train_y'], dtype = np.float32)
    train_y4 = sio.loadmat("/Users/yasushishibe/Documents/MATLAB/multi_tasking_ball_mill/data/1_data/train_wly.mat")
    train_y4 = np.asarray(train_y4['train_y'], dtype = np.float32)
    #test data
    test_x = sio.loadmat("/Users/yasushishibe/Documents/MATLAB/multi_tasking_ball_mill/data/1_data/test_x.mat")
    test_x = np.asarray(test_x['test_x'], dtype = np.float32)
    test_y1 = sio.loadmat("/Users/yasushishibe/Documents/MATLAB/multi_tasking_ball_mill/data/1_data/test_y.mat")
    test_y1 = np.asarray(test_y1['test_y'], dtype = np.float32)
    test_y2 = sio.loadmat("/Users/yasushishibe/Documents/MATLAB/multi_tasking_ball_mill/data/1_data/test_lqby.mat")
    test_y2 = np.asarray(test_y2['test_y'], dtype = np.float32)
    test_y3 = sio.loadmat("/Users/yasushishibe/Documents/MATLAB/multi_tasking_ball_mill/data/1_data/test_ndy.mat")
    test_y3 = np.asarray(test_y3['test_y'], dtype = np.float32)
    test_y4 = sio.loadmat("/Users/yasushishibe/Documents/MATLAB/multi_tasking_ball_mill/data/1_data/test_wly.mat")
    test_y4 = np.asarray(test_y4['test_y'], dtype = np.float32)
    #duplicate data
    #train_x data
    train_x = np.append(train_x, train_x, 0) 
    test_x = np.append(test_x, test_x, 0)
    #train_y data
    train_y1 = np.append(train_y1, train_y1, 0)
    train_y2 = np.append(train_y2, train_y2, 0)
    train_y3 = np.append(train_y3, train_y3, 0)
    train_y4 = np.append(train_y4, train_y4, 0)
    #test_y data
    test_y1 = np.append(test_y1, test_y1, 0)
    test_y2 = np.append(test_y2, test_y2, 0)
    test_y3 = np.append(test_y3, test_y3, 0)
    test_y4 = np.append(test_y4, test_y4, 0)
    return train_x, train_y1, train_y2, train_y3, train_y4, test_x, test_y1, test_y2, test_y3, test_y4

#import data
#train data
train_x, train_y1, train_y2, train_y3, train_y4, test_x, test_y1, test_y2, test_y3, test_y4 = import_data()

#hiperparametros
hidden1 = 5
hidden2 = 15
lstm_hidden = hidden2
hidden4 = 10
out_size=1
lr = 0.0015
n_epochs = 10000
displayer = 10
batch_size = 250
drop=0.5 
n_steps = 6
ventana = 11
n_input = train_x.shape[1]
n_output = 1
batch_init = 0
batch_init_test = 0



def sequencesFromTrainingData(batch_size, n_steps): 
    global batch_init
    X = []
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    
    data_x = np.split(train_x[batch_init:batch_init+ventana*batch_size, :], batch_size, axis=0)
    data_y1 = np.split(train_y1[batch_init:batch_init+ventana*batch_size, :], batch_size, axis=0)
    data_y2 = np.split(train_y2[batch_init:batch_init+ventana*batch_size, :], batch_size, axis=0)
    data_y3 = np.split(train_y3[batch_init:batch_init+ventana*batch_size, :], batch_size, axis=0)
    data_y4 = np.split(train_y4[batch_init:batch_init+ventana*batch_size, :], batch_size, axis=0)
    for p in range(batch_size):
        sample = data_x[p]
        lavel1 = data_y1[p]
        lavel2 = data_y2[p]
        lavel3 = data_y3[p]
        lavel4 = data_y4[p]
        for i in range(ventana-n_steps):
            X.append(sample[i:i+n_steps,:])
            Y1.append(lavel1[i+n_steps,:])
            Y2.append(lavel2[i+n_steps,:])
            Y3.append(lavel3[i+n_steps,:])
            Y4.append(lavel4[i+n_steps,:])
    batch_init = batch_init + ventana*batch_size  
    if (batch_init+ventana*batch_size)>=train_x.shape[0]:
        batch_init = 0
    
    return np.asarray(X), np.asarray(Y1), np.asarray(Y2), np.asarray(Y3), np.asarray(Y4)


def sequencesFromTestData(n_steps): 
    global batch_init_test
    X = []
    Y1 = []
    Y2 = []
    Y3 = []
    Y4 = []
    batch_size = test_x.shape[0]/ventana
    data_x = np.split(test_x[:-2, :], batch_size, axis=0)
    data_y1 = np.split(test_y1[:-2, :], batch_size, axis=0)
    data_y2 = np.split(test_y2[:-2, :], batch_size, axis=0)
    data_y3 = np.split(test_y3[:-2, :], batch_size, axis=0)
    data_y4 = np.split(test_y4[:-2, :], batch_size, axis=0)
    for p in range(batch_size):
        sample = data_x[p]
        lavel1 = data_y1[p]
        lavel2 = data_y2[p]
        lavel3 = data_y3[p]
        lavel4 = data_y4[p]
        for i in range(ventana-n_steps):
            X.append(sample[i:i+n_steps,:])
            Y1.append(lavel1[i+n_steps,:])
            Y2.append(lavel2[i+n_steps,:])
            Y3.append(lavel3[i+n_steps,:])
            Y4.append(lavel4[i+n_steps,:])

    
    return np.asarray(X), np.asarray(Y1), np.asarray(Y2), np.asarray(Y3), np.asarray(Y4)


def main():
    
    
    #set up tensorflow imputs 
    x = tf.placeholder("float", [None, n_steps, n_input])
    y1 = tf.placeholder("float", [None, 1])
    y2 = tf.placeholder("float", [None, 1])
    y3 = tf.placeholder("float", [None, 1])
    y4 = tf.placeholder("float", [None, 1])
    keep_prob = tf.placeholder("float")
    # reshape x
    x_internal = tf.transpose(x,[1, 0, 2])
    x_internal = tf.reshape(x_internal, [-1, n_input]) # (n_steps*batch_size, n_input)
    #lstm cells
    lstm_cell = tf.nn.rnn_cell.BasicLSTMCell(lstm_hidden, forget_bias=1.0)
    istate = lstm_cell.zero_state(batch_size*(ventana-n_steps), tf.float32)
    #inicializate the weights and bias
    w0 = tf.Variable(tf.random_normal([train_x.shape[1], hidden1]))
    b0 = tf.Variable(tf.random_normal([hidden1]))
    #output weights
    w_tk1 = tf.Variable(tf.random_normal([hidden2, out_size]))
    w_tk2 = tf.Variable(tf.random_normal([hidden2, out_size]))
    w_tk3 = tf.Variable(tf.random_normal([hidden2, out_size]))
    w_tk4 = tf.Variable(tf.random_normal([hidden2, out_size]))
    #output bias
    b_tk1 = tf.Variable(tf.random_normal([out_size]))
    b_tk2 = tf.Variable(tf.random_normal([out_size]))
    b_tk3 = tf.Variable(tf.random_normal([out_size]))
    b_tk4 = tf.Variable(tf.random_normal([out_size]))
    #graph
    layer1 = tf.nn.relu(tf.add(tf.matmul(x_internal, w0), b0))
    layer1 = tf.nn.dropout(layer1, keep_prob)
    layer1 = tf.split(0, n_steps, layer1) # n_steps * (batch_size, n_hidden)
    outputs, states = tf.nn.rnn(lstm_cell, layer1, dtype=tf.float32)
    #outputs
    out_tk1 = tf.add(tf.matmul(outputs[-1], w_tk1), b_tk1)
    out_tk2 = tf.add(tf.matmul(outputs[-1], w_tk2), b_tk2)
    out_tk3 = tf.add(tf.matmul(outputs[-1], w_tk3), b_tk3)
    out_tk4 = tf.add(tf.matmul(outputs[-1], w_tk4), b_tk4)
    #loss function
    loss1 = tf.reduce_mean(tf.square(tf.add(out_tk1, -y1)))
    loss2 = tf.reduce_mean(tf.square(tf.add(out_tk2, -y2)))
    loss3 = tf.reduce_mean(tf.square(tf.add(out_tk3, -y3)))
    loss4 = tf.reduce_mean(tf.square(tf.add(out_tk4, -y4)))
    #the final loss
    loss = tf.reduce_mean([loss1, loss2, loss3, loss4])
    #optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    main_optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss1)
    #graph init
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        error_y = 1000
        while error_y > 1:
            batch = sequencesFromTrainingData(batch_size, n_steps)
            _, error_y, output_1 = sess.run([main_optimizer, loss1, out_tk1], feed_dict={x: batch[0], y1: batch[1], keep_prob: drop})
            print "RMSE : {0}, loss: {1}".format(mean_squared_error(output_1, batch[1]), error_y)
            
        
        for _ in range(n_epochs):
            for ii in range((train_x.shape[0]/(ventana*batch_size))-1):
                batch = sequencesFromTrainingData(batch_size, n_steps)
                sess.run(optimizer, feed_dict={x: batch[0], 
                                               y1: batch[1], y2: batch[2], y3: batch[3], y4: batch[4], keep_prob: drop})
                if (ii%displayer)==0:
                    individual_loss = sess.run([loss1, loss2, loss3, loss4], 
                                               feed_dict={x: batch[0], y1: batch[1], y2: batch[2], y3: batch[3], y4: batch[4], keep_prob: 1})
                    batch = sequencesFromTestData(n_steps)
                    total_loss = sess.run(loss, 
                                          feed_dict={x: batch[0], y1: batch[1], y2: batch[2], y3: batch[3], y4: batch[4], keep_prob: 1})
                    
                    print "Individual training loss1: {0}, loss2: {1}, loss3: {2}, loss4: {3}".format(individual_loss[0], individual_loss[1], individual_loss[2], individual_loss[3])
                    print "Test loss: {0}".format(total_loss)
                
        batch = sequencesFromTestData(n_steps)
        total_loss = sess.run(loss, feed_dict={x: batch[0], y1: batch[1], y2: batch[2], y3: batch[3], y4: batch[4],keep_prob: 1})
        print "Test loss: {0}".format(total_loss)
        
        out1, out2, out3, out4 = sess.run([out_tk1, out_tk2, out_tk3, out_tk4], feed_dict={x: batch[0], keep_prob: 1})
        print "error task 1: {0}, error task 2:, {1}, error task 3: {2}, error task 4: {3}".format(mean_squared_error(out1, test_y1), 
                                                                                                    mean_squared_error(out2, test_y2), 
                                                                                                    mean_squared_error(out3, test_y3), 
                                                                                                    mean_squared_error(out4, test_y4))
        _, ax = plt.subplots(2,2)
        ax[0,0].plot(out1, "--r", test_y1, "b")
        ax[0,1].plot(out2, "--r", test_y2, "b")
        ax[1,0].plot(out3, "--r", test_y3, "b")
        ax[1,1].plot(out4, "--r", test_y4, "b")
    
if __name__=="__main__":
    main()