from tweetreader import TweetReader
import tensorflow as tf
import numpy as np
from tensorflow.examples.tutorials.mnist import input_data

data_set = [i for i in TweetReader("test.txt")][:200]
data_x = map(lambda r: map(float, r[:-1]), data_set)
data_y = map(lambda r: [r[-1]], data_set)

raw_input(data_y)

mid_pt = len(data_set)/2
trX, trY, teX, teY = np.array(data_x[:mid_pt]), np.array(data_y[:mid_pt]), np.array(data_x[mid_pt:]), np.array(data_y[:mid_pt])

print trX.shape, trY.shape, teX.shape, teY.shape

def init_weights(shape):
    return tf.Variable(tf.random_normal(shape, stddev=0.01))


def model(X, w_h, w_o):
    h = tf.nn.sigmoid(tf.matmul(X, w_h)) # this is a basic mlp, think 2 stacked logistic regressions
    return tf.matmul(h, w_o) # note that we dont take the softmax at the end because our cost fn does that for us

# trX, trY, teX, teY

X = tf.placeholder("float", [None, 5])
Y = tf.placeholder("float", [None, 1])

w_h = init_weights([5, len(trX)]) # create symbolic variables
w_o = init_weights([len(trX), 1])

py_x = model(X, w_h, w_o)

cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(py_x, Y)) # compute costs
train_op = tf.train.GradientDescentOptimizer(0.05).minimize(cost) # construct an optimizer
predict_op = tf.argmax(py_x, 1)

# Launch the graph in a session
with tf.Session() as sess:
    # you need to initialize all variables
    tf.initialize_all_variables().run()

    for i in range(100):
        for start, end in zip(range(0, 100, 5), range(5, 101, 5)):
            sess.run(train_op, feed_dict={X: trX, Y: trY})
        print(i, np.mean(np.argmax(teY, axis=1) ==
                         sess.run(predict_op, feed_dict={X: teX})))
