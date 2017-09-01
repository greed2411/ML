import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# check https://stackoverflow.com/a/43555840
# temporary solution https://github.com/tensorflow/tensorflow/issues/7778
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

#### Computation Graph ####

mnist = input_data.read_data_sets('/tmp/data', one_hot = True)
# one_hot returns the output as [1,0,0,0,0,0,0,0,0,0] if the image was '0'

# output classes 0-9
n_classes = 10

# batch size for loading '128' features (here images) to the network at a time.
batch_size = 128

# place holding variables
# x is a matrix = height x width
# but we flatten 28 x 28 into 784 string kind of thing.
x = tf.placeholder('float', [None, 784]) # input data
y = tf.placeholder('float')

keep_rate = 0.8
keep_prob = tf.placeholder(tf.float32)

def conv2d(x, W):
    return tf.nn.conv2d(x, W, strides = [1,1,1,1], padding = 'SAME')

def maxpool2d(x):
    #                           size of the window        movement of window
    return tf.nn.max_pool(x, ksize = [1,2,2,1], strides = [1,2,2,1], padding = 'SAME')

def convolutional_neural_network(x):

    # (input data * weights) + biases, used when input data = 0, mostly for making it dynamic
    #                                    5 x 5 convolution , 1 input, 32 features/outputs
    weights = {'W_conv1':tf.Variable(tf.random_normal([5,5,1,32])),
               'W_conv2':tf.Variable(tf.random_normal([5,5,32,64])),
               'W_fc':tf.Variable(tf.random_normal([7*7*64, 1024])),
               'out':tf.Variable(tf.random_normal([1024, n_classes]))}

    biases = {'b_conv1':tf.Variable(tf.random_normal([32])),
               'b_conv2':tf.Variable(tf.random_normal([64])),
               'b_fc':tf.Variable(tf.random_normal([1024])),
               'out':tf.Variable(tf.random_normal([n_classes]))}

    x = tf.reshape(x, shape = [-1, 28, 28, 1])

    conv1 = tf.nn.relu(conv2d(x, weights['W_conv1']) + biases['b_conv1'])
    conv1 = maxpool2d(conv1)

    conv2 =  tf.nn.relu(conv2d(conv1, weights['W_conv2']) + biases['b_conv2'])
    conv2 = maxpool2d(conv2)

    fc =  tf.reshape(conv2, [-1, 7*7*64])
    fc = tf.nn.relu(tf.matmul(fc,weights['W_fc'])+biases['b_fc'])
    fc = tf.nn.dropout(fc, keep_rate)

    output =  tf.matmul(fc, weights['out'])+biases['out']

    return output
    #output is going to be the one_hot thing. Remember.

#### Session ####

def train_neural_network(x):
    prediction = convolutional_neural_network(x)
    # the cost function which is going to calculate how wrong we are with the prediction and actual value
    cost = tf.reduce_mean( tf.nn.softmax_cross_entropy_with_logits(logits = prediction,labels = y) )
    # to minimize the difference in 'y' and 'prediction'
    optimizer =  tf.train.AdamOptimizer().minimize(cost)

    # how many epochs
    hm_epochs = 10

    with tf.Session() as sess:
        sess.run(tf.global_variables_initializer())

        for epoch in range(hm_epochs):
            epoch_loss = 0
            for _ in range(int(mnist.train.num_examples/batch_size)):
                epoch_x, epoch_y = mnist.train.next_batch(batch_size)
                _, c = sess.run([optimizer, cost], feed_dict= {x: epoch_x,y: epoch_y} )
                epoch_loss += c
            print('Epoch', epoch+1, 'completed out of', hm_epochs, 'loss:', epoch_loss)

        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)
