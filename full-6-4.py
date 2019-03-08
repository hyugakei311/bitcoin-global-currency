from __future__ import print_function
from sklearn.preprocessing import MinMaxScaler
from pandas import read_csv

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf

# Parameters
epoch = 4000
learning_rate = 0.01
batch_size = 100

# Network Parameters
n_hidden_1 = 30 # 1st layer number of neurons
num_input = 183 # 
num_output = 183 # 

### TODO: Read data from file ###
dataframe = read_csv('n-transactions.csv', usecols=[1], engine='python', skipfooter=3)
dataset = dataframe.values
dataset = dataset.astype('float32')
### TODO: Scale data from 0 -> 1
scaler = MinMaxScaler(feature_range=(0, 1))
dataset = scaler.fit_transform(dataset)
dataset = dataset.reshape(dataset.size)
### TODO: Get train_set, train_in_batches, train_out_batches
# train_set: [1, 3, 4, 6, 8, 23, 5, 2]
train_size = int(len(dataset) * 0.6)
test_size = len(dataset) - train_size
train_set = dataset[:train_size]

train_in_batches = []
train_ou_batches = []

n = len(train_set)
for out_index in range(num_input, n - num_output + 1):
    train_in_batches.append(train_set[out_index - num_input: out_index])
    train_ou_batches.append(train_set[out_index: out_index + num_output])

# Generates all possible train_in_batches from train_set and num_input

### TODO: Get test_set, test_x, test_y
# Above
test_set = dataset[train_size:len(dataset)]
test_x, test_y = [], []
n = len(test_set)

if n != 0:
    for out_index in range(num_input, n - num_output + 1):
        test_x.append(test_set[out_index - num_input: out_index])
        test_y.append(test_set[out_index: out_index + num_output])

    test_x = np.array(test_x).reshape(len(test_x),-1)
    test_y = np.array(test_y).reshape(len(test_y),-1)
#
### TODO: Implements get_batch() function.
def get_batch(train_in_batches, train_ou_batches, batch_ptr, batch_size):
    # get_batch returns train_x and train_y.
    train_x = train_in_batches[batch_ptr:min(batch_ptr + batch_size,len(train_in_batches))]
    train_y = train_ou_batches[batch_ptr:min(batch_ptr + batch_size,len(train_ou_batches))]
    return np.array(train_x).reshape(len(train_x), -1), np.array(train_y).reshape(len(train_y),-1)
    # train_x contains batch_size batches, starting from batch_ptr, each of them 
    # is an input vector. 

    # train_y contains batch_size batches, starting from batch_ptr, each of them 
    # is an output vector for the corresponding input.
    #
    # For example: 
    # train_in_batches: [[1, 3, 4], [3, 4, 8], [4, 6, 8]]
    # train_ou_batches: [[6, 8 ,23], [8,23, 5], [23, 5, 2]]
    # batch_ptr = 1, batch_size = 2
    # 
    # => train_x: [[3, 4, 8], [4, 6, 8]], train_y: [[8, 23, 5], [23, 5, 2]]
##################################

# tf Graph input
X = tf.placeholder("float", [None, num_input])
Y = tf.placeholder("float", [None, num_output])

# Store layers weight & bias
weights = {
    'h1': tf.Variable(tf.random_normal([num_input, n_hidden_1])),
    'out': tf.Variable(tf.random_normal([n_hidden_1, num_output]))
}
biases = {
    'b1': tf.Variable(tf.random_normal([n_hidden_1])),
    'out': tf.Variable(tf.random_normal([num_output]))
}


# Create model
def neural_net(x):
    # Hidden fully connected layer with 256 neurons
    layer_1 = tf.add(tf.matmul(x, weights['h1']), biases['b1'])
    # Output fully connected layer with a neuron for each class
    out_layer = tf.matmul(layer_1, weights['out']) + biases['out']
    return out_layer

### Construct model
##logits = neural_net(X)
##prediction = tf.nn.softmax(logits)
##
### Define loss and optimizer
##loss_op = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(
##    logits=logits, labels=Y))
logits = neural_net(X)
prediction = tf.sigmoid(logits)

# Define loss and optimizer
loss_op = tf.losses.mean_squared_error(Y, prediction)
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate)
train_op = optimizer.minimize(loss_op)

# Initialize the variables (i.e. assign their default value)
init = tf.global_variables_initializer()

# Start training
with tf.Session() as sess:

    # Run the initializer
    sess.run(init)

    for i in range(epoch):
        batch_ptr = 0
        while batch_ptr < len(train_in_batches):
            # TODO: Implement get_batch() function
            train_x, train_y = get_batch(train_in_batches, train_ou_batches, batch_ptr, batch_size)
            batch_ptr += batch_size
            # Run optimization op (backprop)
            sess.run(train_op, feed_dict={X: train_x, Y: train_y})
            
            if batch_ptr > len(train_in_batches) :
                # Calculate batch loss and accuracy
                loss = sess.run(loss_op, feed_dict={X: train_x,
                                                                     Y: train_y})
                print("Step " + str(i) + ", Minibatch Loss= " + \
                      "{:.4f}".format(loss))
                

    print("Optimization Finished!")

    plt.plot(dataset)

    # Calculate train loss
    pred, loss = sess.run([prediction, loss_op], feed_dict={X: train_in_batches, Y: train_ou_batches})
    print("Train loss: " + str(loss))

    # Calculate test loss
    if len(test_x) != 0:
        pred, loss = sess.run([prediction, loss_op], feed_dict={X: test_x, Y: test_y})
        print("Test loss: " + str(loss))

    pred = sess.run(prediction, feed_dict={X: [dataset[len(dataset) - num_output:]]})

    plot_pred = [np.nan for _ in range(len(dataset) + num_output)]
    plot_pred[len(dataset) + 1:] = pred[0]
    plt.plot(plot_pred)

    result = scaler.inverse_transform(pred)
    print(result[0])

    plt.show()

