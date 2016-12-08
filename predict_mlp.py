from __future__ import print_function

from projutils import load_project_data
import tensorflow as tf
import numpy as np

trX, trY, teX, teY = load_project_data()

# Parameters
learning_rate = 0.5
training_epochs = 100
batch_size = 100
display_step = 1

# Network Parameters
n_input = 7
n_output = 1
layers = [n_input, 512, 512, 256, 128, n_output]

# tf Graph input
x = tf.placeholder("float", [None, n_input])
y = tf.placeholder("float", [None, n_output])

# Create model
def multilayer_perceptron(x, weights, biases, afn):
    n = len(weights)
    for i in range(n - 1):
        x = afn(tf.matmul(x, weights[i]) + biases[i])
    return tf.matmul(x, weights[n - 1]) + biases[n - 1]

def init_vars(layers):
    weights, biases = [], []
    n = len(layers)
    for i in range(n - 1):
        lyrstd = np.sqrt(1.0 / layers[i])
        cur_w = tf.Variable(tf.random_normal([layers[i], layers[i + 1]], stddev=lyrstd))
        weights.append(cur_w)
        cur_b = tf.Variable(tf.random_normal([layers[i + 1]], stddev=lyrstd))
        biases.append(cur_b)
    return (weights, biases)

# Construct model
weights, biases = init_vars(layers)
pred = multilayer_perceptron(x, weights, biases, tf.tanh)

# Define loss and optimizer
cost = tf.reduce_sum(tf.nn.l2_loss(pred - y))
optimizer = tf.train.AdamOptimizer(learning_rate=learning_rate).minimize(cost)

# Initializing the variables
init = tf.global_variables_initializer()

# Launch the graph
with tf.Session() as sess:
    sess.run(init)

    # Training cycle
    for epoch in range(training_epochs):
        avg_cost = 0.
        total_batch = int(len(trX)/batch_size)
        # Loop over all batches
        for i in range(total_batch):
            lo = i*batch_size
            hi = (i+1)*batch_size
            batch_x, batch_y = trX[lo:hi], trY[lo:hi].reshape(batch_size,n_output)

            # Run optimization op (backprop) and cost op (to get loss value)
            sess.run(optimizer, feed_dict={x: batch_x, y: batch_y})
            c = np.sqrt(sess.run(cost, feed_dict={x: batch_x, y: batch_y}) * 2.0 / total_batch)
            # Compute average loss
            #print(c)
            avg_cost += np.sum(c) / total_batch

        # Display logs per epoch step
        if epoch % display_step == 0:
            print("Epoch:", '%04d' % (epoch+1), "cost=", \
                "{:.9f}".format(avg_cost))
