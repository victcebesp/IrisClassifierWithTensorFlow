import gzip
import cPickle

import tensorflow as tf
import numpy as np

import os
os.environ['TF_CPP_MIN_LOG_LEVEL'] = '2'

def get_hidden_units(path):
    hidden_units = {
        "./../models/H5B20E20/model.ckpt" : 5,
        "./../models/H5B50E20/model.ckpt": 5,
        "./../models/H20B20E25/model.ckpt": 20,
        "./../models/H20B50E25/model.ckpt": 20
    }

    if hidden_units.has_key(path):
        return hidden_units[path]
    else:
        return -1

def test_accuracy(path):

    hidden = get_hidden_units(path)
    if hidden == -1:
        print("Bad path")
        return

    f = gzip.open('./../data/mnist.pkl.gz', 'rb')
    train_set, valid_set, test_set = cPickle.load(f)
    f.close()

    test_x, test_y = test_set

    #learning_rate = 0.01

    # ---------------------------#
    #           model           #
    # ---------------------------#
    x = tf.placeholder("float", [None, 784])
    y_ = tf.placeholder("float", [None, 10])

    W1 = tf.Variable(np.float32(np.random.rand(784, hidden)) * 0.1)
    b1 = tf.Variable(np.float32(np.random.rand(hidden)) * 0.1)

    W2 = tf.Variable(np.float32(np.random.rand(hidden, 10)) * 0.1)
    b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

    h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
    y = tf.nn.softmax(tf.matmul(h, W2) + b2)

    loss = tf.reduce_sum(tf.square(y_ - y))

    train = tf.train.GradientDescentOptimizer(0.01).minimize(loss)
    saver = tf.train.Saver()
    with tf.Session() as sess:
        saver.restore(sess, path)
        test_output = sess.run(y, feed_dict={x: test_x})
        test_output = [np.argmax(e) for e in test_output]
        confusion_matrix = sess.run(tf.confusion_matrix(test_y, np.array(test_output), 10))
        print(confusion_matrix)
        print("Accuracy: ", np.diagonal(confusion_matrix).sum() * 100 / len(test_output), "%")

test_accuracy("./../models/H5B20E20/model.ckpt")
#test_accuracy("./../models/H5B50E20/model.ckpt")
#test_accuracy("./../models/H20B20E25/model.ckpt")
#test_accuracy("./../models/H20B50E25/model.ckpt")