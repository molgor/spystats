
t0 = 3 # a rank 0 tensor; this is a scalar with shape []
t1 = [1., 2., 3.] # a rank 1 tensor; this is a vector with shape [3]
t2 = [[1., 2., 3.], [4., 5., 6.]] # a rank 2 tensor; a matrix with shape [2, 3]
t3 = [[[1., 2., 3.]], [[7., 8., 9.]]] # a rank 3 tensor with shape [2, 1, 3]

# Import Tensorflow ------------------------------------------------------------
import tensorflow as tf

# Computational Graph ----------------------------------------------------------
node1 = tf.constant(3.0, dtype=tf.float32)
node2 = tf.constant(4.0) # also tf.float32 implicitly
print(node1, node2)

sess = tf.Session()
print(sess.run([node1, node2]))

node3 = tf.add(node1, node2)
print("node3:", node3)
print("sess.run(node3):", sess.run(node3))

# A placeholder is a promise to provide a value later --------------------------
a = tf.placeholder(tf.float32)
b = tf.placeholder(tf.float32)
adder_node = a + b  # + provides a shortcut for tf.add(a, b)
print(sess.run(adder_node, {a: 3, b: 4.5})) # feed_dict argument
print(sess.run(adder_node, {a: [1, 3], b: [2, 4]}))

add_and_triple = adder_node * 3.
print(sess.run(add_and_triple, {a: 3, b: 4.5}))

#  Variables allow us to add trainable parameters to a graph -------------------
W = tf.Variable([.3], dtype=tf.float32)
b = tf.Variable([-.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
linear_model = W * x + b

# Variables are not initialized when you call tf.Variable ----------------------
init = tf.global_variables_initializer() # initialize variables
sess.run(init)
print(sess.run(linear_model, {x: [1, 2, 3, 4]}))

# Loss function and response variable as placeholder ---------------------------
y = tf.placeholder(tf.float32)
squared_deltas = tf.square(linear_model - y)
loss = tf.reduce_sum(squared_deltas)
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# Manual reassigment -----------------------------------------------------------
fixW = tf.assign(W, [-1.])
fixb = tf.assign(b, [1.])
sess.run([fixW, fixb])
print(sess.run(loss, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]}))

# Optimizers -------------------------------------------------------------------
optimizer = tf.train.GradientDescentOptimizer(0.01)
train = optimizer.minimize(loss)
sess.run(init) # reset values to incorrect defaults.
for i in range(1000):
    sess.run(train, {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print(sess.run([W, b]))

# Evaluate training accuracy ---------------------------------------------------
curr_W, curr_b, curr_loss = sess.run([W, b, loss],
        {x: [1, 2, 3, 4], y: [0, -1, -2, -3]})
print("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

