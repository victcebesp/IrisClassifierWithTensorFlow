import gzip
import cPickle

import tensorflow as tf
import numpy as np


# Translate a list of labels into an array of 0's and one 1.
# i.e.: 4 -> [0,0,0,0,1,0,0,0,0,0]
def one_hot(x, n):
    """
    :param x: label (int)
    :param n: number of bits
    :return: one hot code
    """
    if type(x) == list:
        x = np.array(x)
    x = x.flatten()
    o_h = np.zeros((len(x), n))
    o_h[np.arange(len(x)), x] = 1
    return o_h

def checkThreshold(threshold, error, epoch):
    if len(error) >= 2:
        print("THRESHOLD: ", (abs(error[epoch - 2] - error[epoch - 3])))
        return abs(error[epoch - 2] - error[epoch - 3]) > threshold
    return True


f = gzip.open('./../data/mnist.pkl.gz', 'rb')
train_set, valid_set, test_set = cPickle.load(f)
f.close()

valid_x, valid_y = valid_set

valid_y = one_hot(valid_y, 10)

train_x, train_y = train_set

train_y = one_hot(train_y, 10)

test_x, test_y = test_set

#---------------------------#
#      hyperparameter       #
#---------------------------#

batch_size = 50
max_epochs = 25
hidden_layer_neurons = 20
threshold = 10
learning_rate = 0.01

#---------------------------#
#           model           #
#---------------------------#



x = tf.placeholder("float", [None, 784])
y_ = tf.placeholder("float", [None, 10])

W1 = tf.Variable(np.float32(np.random.rand(784, hidden_layer_neurons)) * 0.1)
b1 = tf.Variable(np.float32(np.random.rand(hidden_layer_neurons)) * 0.1)

W2 = tf.Variable(np.float32(np.random.rand(hidden_layer_neurons, 10)) * 0.1)
b2 = tf.Variable(np.float32(np.random.rand(10)) * 0.1)

h = tf.nn.sigmoid(tf.matmul(x, W1) + b1)
y = tf.nn.softmax(tf.matmul(h, W2) + b2)

loss = tf.reduce_sum(tf.square(y_ - y))

train = tf.train.GradientDescentOptimizer(learning_rate).minimize(loss)

init = tf.initialize_all_variables()

sess = tf.Session()
sess.run(init)

epoch = 1
error = []
epochs = []

import matplotlib.pyplot as plt

while checkThreshold(threshold, error, epoch) and epoch <= max_epochs:
    for jj in xrange(len(train_x) / batch_size):
        batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
        batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
        sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    batch_xs = train_x[jj * batch_size: jj * batch_size + batch_size]
    batch_ys = train_y[jj * batch_size: jj * batch_size + batch_size]
    sess.run(train, feed_dict={x: batch_xs, y_: batch_ys})

    print "Epoch #:", epoch, "Error: ", sess.run(loss, feed_dict={x: batch_xs, y_: batch_ys})
    each_error = sess.run(loss, feed_dict={x: valid_x, y_: valid_y.astype(np.float32)})
    result = sess.run(y, feed_dict={x: batch_xs})
    error.append(each_error)
    epoch += 1
    for b, r in zip(batch_ys, result):
        print b, "-->", r
    print "----------------------------------------------------------------------------------"

saver = tf.train.Saver()
saver.save(sess, "./../models/H20B50E25/model.ckpt")

plt.plot(error)
figure = plt.gcf()
figure.savefig("./../models/H20B50E25/chart.png")
plt.show()

test_output = sess.run(y, feed_dict={x: test_x})
test_output = [np.argmax(e) for e in test_output]
print(sess.run(tf.confusion_matrix(test_y, np.array(test_output), 10)))

# ---------------- Visualizing some element of the MNIST dataset --------------

import matplotlib.cm as cm
import matplotlib.pyplot as plt

#plt.imshow(train_x[57].reshape((28, 28)), cmap=cm.Greys_r)
#plt.show()  # Let's see a sample
#print train_y[57]
