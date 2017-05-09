from tensorflow.examples.tutorials.mnist import input_data
mnist = input_data.read_data_sets('MNIST_data', one_hot=True)

import tensorflow as tf
sess = tf.InteractiveSession()

############# Softmax Regression Model #####################

# placeholder for external input/output
x = tf.placeholder(tf.float32, shape=[None, 784])
labels = tf.placeholder(tf.float32, shape=[None, 10])

W = tf.Variable(tf.zeros([784, 10]))
b = tf.Variable(tf.zeros([10]))

sess.run(tf.global_variables_initializer())

y = tf.matmul(x, W) + b

# loss function
cross_entropy_loss = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(labels=labels, logits=y))

# train model

# The returned operation train_step, when run, will apply the gradient descent updates to the parameters.
# Training the model can therefore be accomplished by repeatedly running train_step.
train_step = tf.train.GradientDescentOptimizer(0.5).minimize(cross_entropy_loss)

for _ in range(1000):
	batch = mnist.train.next_batch(100)
	train_step.run(feed_dict={x: batch[0], labels: batch[1]})

# eval the model
correct_predictions = tf.equal(tf.argmax(y, 1), tf.argmax(labels, 1))
accuracy = tf.reduce_mean(tf.cast(correct_predictions, tf.float32))

print(accuracy.eval(feed_dict={x: mnist.test.images, labels: mnist.test.labels})) # 92.6%









