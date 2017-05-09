from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

############# Multi-layer ConvNet Model #####################

# placeholder for external input/output
x = tf.placeholder(tf.float32, shape=[None, 784])
labels = tf.placeholder(tf.float32, shape=[None, 10])

# filters
def weight_variable(shape):
	initial = tf.truncated_normal(shape, stddev=0.1)
	return tf.Variable(initial)

def bias_variable(shape):
	initial = tf.constant(0.1, shape=shape)
	return tf.Variable(initial)

def conv2d(x, W):
	return tf.nn.conv2d(x, W, strides=[1, 1, 1, 1], padding='SAME') # zero padded so output size is the same as input size, 'VALID' --> no padding

def max_pool_2x2(x):
	return tf.nn.max_pool(x, ksize=[1, 2, 2, 1], strides=[1, 2, 2, 1], padding='SAME')

### 1st conv layer ####

# The first two dimensions are the patch size, 
# the next is the number of input channels, 
# and the last is the number of output channels. 
# We will also have a bias vector with a component for each output channel
W_conv1 = weight_variable([5, 5, 1, 32]) # 32 5x5x1 filters
b_conv1 = bias_variable([32])

# To apply the layer, we first reshape x to a 4d tensor, 
# first dimension -1 means that the first dimension will be infered from the rest of dimensions
# with the second and third dimensions corresponding to image width and height, 
# and the final dimension corresponding to the number of color channels.
x_image = tf.reshape(x, [-1, 28, 28, 1])

h_conv1 = tf.nn.relu(conv2d(x_image, W_conv1) + b_conv1)
h_pool1 = max_pool_2x2(h_conv1) # shape 14x14x32


### 2nd conv layer ###

W_conv2 = weight_variable([5, 5, 32, 64]) # 64 5x5x32 filters
b_conv2 = bias_variable([64])

h_conv2 = tf.nn.relu(conv2d(h_pool1, W_conv2) + b_conv2)
h_pool2 = max_pool_2x2(h_conv2) # shape 7x7x64


### fully connected layer ###

W_fc_1 = weight_variable([7*7*64, 1024])
b_fc_1 = bias_variable([1024])

h_pool2_flat = tf.reshape(h_pool2, [-1, 7*7*64])
h_fc_1 = tf.nn.relu(tf.matmul(h_pool2_flat, W_fc_1) + b_fc_1)

### Drop out ###

# To reduce overfitting, we will apply dropout before the readout layer. 
# We create a placeholder for the probability that a neuron's output is kept during dropout. 
# This allows us to turn dropout on during training, and turn it off during testing. 
keep_prob = tf.placeholder(tf.float32)
h_fc_1_dropout = tf.nn.dropout(h_fc_1, keep_prob)

### Read-out layer ###

W_fc_2 = weight_variable([1024, 10])
b_fc_2 = bias_variable([10])

y_conv = tf.matmul(h_fc_1_dropout, W_fc_2) + b_fc_2


#### Train #####

cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y_conv))
train_step = tf.train.AdamOptimizer(1e-4).minimize(cross_entropy_loss)

correct_predictions = tf.equal(tf.argmax(y_conv, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

sess.run(tf.global_variables_initializer())

for i in range(1000):
	batch = mnist.train.next_batch(50)
	if i % 100 == 0:
		train_accuracy = accuracy.eval(feed_dict={x: batch[0], labels: batch[1], keep_prob: 1.0}) # pass value to placeholder, no dropout
		print("step %d, training accuracy: %g"%(i, train_accuracy))
	train_step.run(feed_dict={x: batch[0], labels: batch[1], keep_prob: 0.5}) # drop 50%


#### Test predict ####

print("test accuracy: %g"%(accuracy.eval(feed_dict={x: mnist.test.images, labels: mnist.test.labels, keep_prob: 1.0}))) # 96.38%




