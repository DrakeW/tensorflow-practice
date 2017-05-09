import tensorflow as tf
import numpy as np

# Model parameters
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)

# Model input & output
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)

linear_model = W * x + b

# Model loss function
loss = tf.reduce_sum(tf.square(linear_model - y))

# choose optimizer
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)

# training data
X_train = [1,2,3,4]
y_train = [0,-1,-2,-3]

# training loop
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init) # reset variables to initial values
for i in range(1000):
	sess.run(train, feed_dict={x: X_train, y: y_train})

# eval training accuracy
curr_W, curr_b, curr_loss = sess.run([W, b, loss], feed_dict={x: X_train, y: y_train})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

