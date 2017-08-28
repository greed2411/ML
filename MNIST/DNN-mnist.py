import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'
# check https://stackoverflow.com/a/43555840
# temporary solution https://github.com/tensorflow/tensorflow/issues/7778 
import tensorflow as tf
from tensorflow.examples.tutorials.mnist import input_data

"""
What actually happens?

1) input -> weight -> hidden layer 1 (activation function) -> weights -> hidden layer 2 (activation function) -> weights -> output layer

    The above process is a feedforward network.

2) Compare output to intended output.

    Using a cost(loss) function (example : cross entropy)

3) Using the cost function result we try to minimize the cost using an optimization function.
    
    To minimize cost (example : AdamOptimizer, Stochastic Gradient descent, AdaGrad)

4) Using that value we go back and manipulate the weights.

    That is called backpropagation.

    Also feedforward + backpropagation = epoch ( 1 cycle of feedforward and backpropagation)
    After each epoch the cost function output value falls.
"""


#### Computation Graph ####

mnist = input_data.read_data_sets('/tmp/data', one_hot = True)
# one_hot returns the output as [1,0,0,0,0,0,0,0,0,0] if the image was '0'

# hidden input layers for DNN
n_nodes_hl1 = 500
n_nodes_hl2 = 500
n_nodes_hl3 = 500

# output classes 0-9
n_classes = 10

# batch size for loading '100' features (here images) to the network at a time.
batch_size = 100

# place holding variables
# x is a matrix = height x width
# but we flatten 28 x 28 into 784 string kind of thing.
x = tf.placeholder('float', [None, 784]) # input data
y = tf.placeholder('float')

def neural_network_model(data):

    # (input data * weights) + biases, used when input data = 0, mostly for making it dynamic
    hidden_1_layer = {'weights':tf.Variable(tf.random_normal([784, n_nodes_hl1])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}
    
    hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

    hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
                        'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

    output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl3, n_classes])),
                        'biases':tf.Variable(tf.random_normal([n_classes]))}

    l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
    # relu - rectified linear, activation function or threshold function.
    l1 = tf.nn.relu(l1)

    l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
    l2 = tf.nn.relu(l2)

    l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
    l3 = tf.nn.relu(l3)

    output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']

    return output
    #output is going to be the one_hot thing. Remember.

#### Session ####

def train_neural_network(x):
    prediction = neural_network_model(x)
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
            print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
        
        correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))

        accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
        print('Accuracy: ', accuracy.eval({x:mnist.test.images, y:mnist.test.labels}))

train_neural_network(x)