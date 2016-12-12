from tweetreader import TweetReader
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data_set = [i for i in TweetReader("data/tweet_data.txt")][:10000]
data_x = map(lambda r: map(float, r[:-1]), data_set)
data_y = map(lambda r: [r[-1]], data_set)

mid_pt = len(data_set)/2
trX, trY, teX, teY = np.array(data_x[:mid_pt]), np.array(data_y[:mid_pt]), np.array(data_x[mid_pt:]), np.array(data_y[mid_pt:])

print (trX.shape, trY.shape, teX.shape, teY.shape)

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))

def model(X, w_h, w_o):
    h = tf.nn.tanh(tf.matmul(X, w_h))
    return tf.matmul(h, w_o)

X = tf.placeholder("float", [None, len(trX[0])])
Y = tf.placeholder("float", [None, 1])

w_h = init_weights([len(trX[0]), 500]) # create symbolic variables
w_o = init_weights([500, len(trY[0])])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    inc = 50
    for i in range(500):
        for start, end in zip(range(0, len(trX), inc), range(inc, len(trX)+1, inc)):
            sess.run(train_op, feed_dict={X: trX[start:end], Y: trY[start:end]})
        print(i, np.mean(teY - sess.run(predict_op, feed_dict={X: teX})))
