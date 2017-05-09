import tensorflow as tf

# ### tensors (node) ###
node1 = tf.constant(3.0, tf.float32)
node2 = tf.constant(4.0) # tf.float32 implicitly
print(node1, node2)


# ### Session ###
# To actually evaluate the nodes, we must run the computational graph within a session. 
# A session encapsulates the control and state of the TensorFlow runtime.
sess = tf.Session()
print(sess.run([node1, node2]))


# ### operations (also node) ###
node3 = tf.add(node1, node2)
print("node3: ", node3)
print("sess.run(node3): ", sess.run(node3))


# ### placeholders ### 
# A graph can be parameterized to accept external inputs, known as placeholders
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b

print(sess.run(adder_node, feed_dict={a: 3, b: 4.5}))
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3
print(sess.run(add_and_triple, {a: 3, b: 4.5}))


# ### Variable ###
# To make the model trainable, we need to be able to modify the graph to get new outputs with the same input. 
# Variables allow us to add trainable parameters to a graph
W = tf.Variable([0.3], tf.float32)
b = tf.Variable([-0.3], tf.float32)
x = tf.placeholder(tf.float32)

linear_model = W * x + b

init = tf.global_variables_initializer() # to initialize variables
sess.run(init)
print("Eval linear model: ", sess.run(linear_model, feed_dict={x: [1,2,3,4]}))


y = tf.placeholder(tf.float32) # labels
squared_loss = tf.reduce_sum(tf.square(linear_model - y))
print("Loss: ", sess.run(squared_loss, feed_dict={x: [1,2,3,4], y: [0, -1, -2, -3]}))


# Change tf.variable value using assign
fixW = tf.assign(W, [-1.0])
fixb = tf.assign(b, [1.0])
sess.run([fixW, fixb])
print("new loss: ", sess.run(squared_loss,  feed_dict={x: [1,2,3,4], y: [0, -1, -2, -3]}))

# ### tf.Train API ###
# various optimization method included
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(squared_loss)

sess.run(init)
for i in range(1000):
	sess.run(train, feed_dict={x: [1,2,3,4], y: [0, -1, -2, -3]})

print("W,b after training: ", sess.run([W, b]))





