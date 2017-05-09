import tensorflow as tf
import numpy as np

# ##### Low-level Implementation #####

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


##### tf.contrib.learn High-level API #####

features = [tf.contrib.layers.real_valued_column("x", dimension=1)]

estimator = tf.contrib.learn.LinearRegressor(feature_columns=features)

X_train = np.array([1,2,3,4])
y_train = np.array([0,-1,-2,-3])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": X_train}, y_train, batch_size=4, num_epochs=1000)

estimator.fit(input_fn=input_fn, steps=1000)

print(estimator.evaluate(input_fn=input_fn))


##### custom model #####

def model(features, labels, mode):
	# parameters
	W = tf.get_variable("W", [1], dtype=tf.float64)
	b = tf.get_variable("b", [1], dtype=tf.float64)
	y = W*features["x"] + b
	# loss
	loss = tf.reduce_sum(tf.square(y - labels))
	# training sub-graph
	gloabl_step = tf.train.get_global_step()
	optimizer = tf.train.GradientDescentOptimizer(0.01)
	train = tf.group(optimizer.minimize(loss), tf.assign_add(gloabl_step, 1))
	# ModelFnOps connects subgraphs we built to the
  	# appropriate functionality.
  	return tf.contrib.learn.ModelFnOps(
  		mode=mode, predictions=y,
  		loss=loss,
  		train_op=train)

estimator = tf.contrib.learn.Estimator(model_fn=model)

X_train = np.array([1.,2.,3.,4.])
y_train = np.array([0.,-1.,-2.,-3.])
input_fn = tf.contrib.learn.io.numpy_input_fn({"x": X_train}, y_train, batch_size=4, num_epochs=1000)

# train
estimator.fit(input_fn=input_fn, steps=1000)
print(estimator.evaluate(input_fn=input_fn, steps=10))
