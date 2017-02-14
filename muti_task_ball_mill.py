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
#hiperparametros
hiden1 = 5
hiden2 = 20
hiden3 = 15
hiden4 = 10
out_size=1
lr = 0.00001
n_epochs = 10000
displayer = 100
batch_size = 10
drop=0.5 

def main():
    
    #import data
    #train data
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
    
    #set up tensorflow imputs 
    x = tf.placeholder("float", [None, train_x.shape[1]])
    y1 = tf.placeholder("float", [None, 1])
    y2 = tf.placeholder("float", [None, 1])
    y3 = tf.placeholder("float", [None, 1])
    y4 = tf.placeholder("float", [None, 1])
    keep_prob = tf.placeholder("float")
    #inicializate the weigths and bias
    w0 = tf.Variable(tf.random_normal([train_x.shape[1], hiden1]))
    w1 = tf.Variable(tf.random_normal([hiden1, hiden2]))
    w2 = tf.Variable(tf.random_normal([hiden2, hiden3]))
    w3 = tf.Variable(tf.random_normal([hiden3, hiden4]))
    w_tk1 = tf.Variable(tf.random_normal([hiden4, out_size]))
    w_tk2 = tf.Variable(tf.random_normal([hiden4, out_size]))
    w_tk3 = tf.Variable(tf.random_normal([hiden4, out_size]))
    w_tk4 = tf.Variable(tf.random_normal([hiden4, out_size]))
    #crate bias
    b0 = tf.Variable(tf.random_normal([hiden1]))
    b1 = tf.Variable(tf.random_normal([hiden2]))
    b2 = tf.Variable(tf.random_normal([hiden3]))
    b3 = tf.Variable(tf.random_normal([hiden4]))
    b_tk1 = tf.Variable(tf.random_normal([out_size]))
    b_tk2 = tf.Variable(tf.random_normal([out_size]))
    b_tk3 = tf.Variable(tf.random_normal([out_size]))
    b_tk4 = tf.Variable(tf.random_normal([out_size]))
    #graph
    layer1 = tf.nn.relu(tf.add(tf.matmul(x, w0), b0))
    layer1 = tf.nn.dropout(layer1, keep_prob)
    layer2 = tf.nn.relu(tf.add(tf.matmul(layer1, w1), b1))
    layer2 = tf.nn.dropout(layer2, keep_prob)
    layer3 = tf.nn.relu(tf.add(tf.matmul(layer2, w2), b2))
    layer3 = tf.nn.dropout(layer3, keep_prob)
    layer4 = tf.nn.relu(tf.add(tf.matmul(layer3, w3), b3))
    layer4 = tf.nn.dropout(layer4, keep_prob)
    #multitask layers
    out_tk1 = tf.nn.relu(tf.add(tf.matmul(layer4, w_tk1), b_tk1))
#     out_tk1 = tf.nn.dropout(out_tk1, keep_prob)
    out_tk2 = tf.nn.relu(tf.add(tf.matmul(layer4, w_tk2), b_tk2))
#     out_tk2 = tf.nn.dropout(out_tk2, keep_prob)
    out_tk3 = tf.nn.relu(tf.add(tf.matmul(layer4, w_tk3), b_tk3))
#     out_tk3 = tf.nn.dropout(out_tk3, keep_prob)   
    out_tk4 = tf.nn.relu(tf.add(tf.matmul(layer4, w_tk4), b_tk4))
#     out_tk4 = tf.nn.dropout(out_tk4, keep_prob)
    #loss function
    loss1 = tf.reduce_mean(tf.square(tf.add(out_tk1, -y1)))
    loss2 = tf.reduce_mean(tf.square(tf.add(out_tk2, -y2)))
    loss3 = tf.reduce_mean(tf.square(tf.add(out_tk3, -y3)))
    loss4 = tf.reduce_mean(tf.square(tf.add(out_tk4, -y4)))
    #the final loss
    loss = tf.reduce_mean([loss1, loss2, loss3, loss4])
    #optimizer
    optimizer = tf.train.AdamOptimizer(learning_rate=lr).minimize(loss)
    #graph init
    with tf.Session() as sess:
        sess.run(tf.initialize_all_variables())
        for i in range(n_epochs):
            for batch_init, batch_end in zip(range(0,train_x.shape[0], batch_size), range(batch_size,train_x.shape[0], batch_size)):
                sess.run(optimizer, feed_dict={x: train_x[batch_init:batch_end,:], y1: train_y1[batch_init:batch_end,:], 
                                                                                   y2: train_y2[batch_init:batch_end,:], 
                                                                                   y3: train_y3[batch_init:batch_end,:],
                                                                                   y4: train_y4[batch_init:batch_end,:], 
                                                                                   keep_prob: drop})
            if (i%displayer)==0:
                total_loss = sess.run(loss, feed_dict={x: test_x, 
                                                       y1: test_y1, 
                                                       y2: test_y1, 
                                                       y3: test_y1,
                                                       y4: test_y1,
                                                       keep_prob: 1})
                individual_loss = sess.run([loss1, loss2, loss3, loss4], feed_dict={x: train_x, y1: train_y1, y2: train_y2, y3: train_y3, y4: train_y4, keep_prob: 1})
                print "Individual task training loss1: {0}, loss2: {1}, loss3: {2}, loss4: {3}".format(individual_loss[0], individual_loss[1], individual_loss[2], individual_loss[3])
                print "Test loss: {0}".format(total_loss)
                
        total_loss = sess.run(loss, feed_dict={x: test_x, 
                                               y1: test_y1, 
                                               y2: test_y1, 
                                               y3: test_y1,
                                               y4: test_y1,
                                               keep_prob: 1})
        print "Test loss: {0}".format(total_loss)
        
        out1, out2, out3, out4 = sess.run([out_tk1, out_tk2, out_tk3, out_tk4], feed_dict={x: test_x, keep_prob: 1})
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
    