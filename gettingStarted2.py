# linear regression

import tensorflow as tf

# Model parameters
W = tf.Variable([0.3], dtype=tf.float32)
b = tf.Variable([-0.3], dtype=tf.float32)
x = tf.placeholder(tf.float32)
y = tf.placeholder(tf.float32)
linear_model = W * x + b
optimizer = tf.train.GradientDescentOptimizer(0.01) # learning rate

# Training data
x_data = [1, 2, 3, 4]
y_data = [0, -1, -2, -3]

# Functions
loss = tf.reduce_sum(tf.square(linear_model-y)) # idkw reduce_sum does but it shows as 1 number
train = optimizer.minimize(loss)

init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)

for i in range(1000):
   sess.run(train, {x: x_data, y: y_data})

# Print out results
curr_W, curr_b, curr_loss = sess.run([W, b, loss], {x: x_data, y: y_data})
print ("W: %s b: %s loss: %s"%(curr_W, curr_b, curr_loss))

