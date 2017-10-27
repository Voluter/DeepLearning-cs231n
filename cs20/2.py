""" Simple linear regression example in TensorFlow
This program tries to predict the number of thefts from 
the number of fire in the city of Chicago
Author: Chip Huyen
Prepared for the class CS 20SI: "TensorFlow for Deep Learning Research"
cs20si.stanford.edu
"""
import os
os.environ['TF_CPP_MIN_LOG_LEVEL']='2'

import numpy as np
import matplotlib.pyplot as plt
import tensorflow as tf
import xlrd

import utils

DATA_FILE = 'graphs/slr05.xls'

# Step 1: read in data from the .xls file
book = xlrd.open_workbook(DATA_FILE, encoding_override="utf-8")
sheet = book.sheet_by_index(0)
data = np.asarray([sheet.row_values(i) for i in range(1, sheet.nrows)])
n_samples = sheet.nrows - 1

# Step 2: create placeholders for input X (number of fire) and label Y (number of theft)
X = tf.placeholder(tf.float32, name="X")
Y = tf.placeholder(tf.float32, name="Y")
# Step 3: create weight and bias, initialized to 0
W = tf.Variable(0.0, name="Weight_1")
b = tf.Variable(0.0, name="bias")
u = tf.Variable(0.0, name="Weight_2")
# Step 4: build model to predict Y
Y_predicted = X*X*W + X*u + b

# Step 5: use the square error as the loss function
loss = tf.square(Y-Y_predicted, name="loss")
#loss = utils.huber_loss(Y, Y_predicted)

# Step 6: using gradient descent with learning rate of 0.01 to minimize loss
optimizer = tf.train.GradientDescentOptimizer(learning_rate=0.0075).minimize(loss)
X_input = np.linspace(-1, 1, 100)
Y_input = X_input*3 + np.random.randn(X_input.shape[0])*0.5
with tf.Session() as sess:
	# Step 7: initialize the necessary variables, in this case, w and b
	sess.run(tf.global_variables_initializer())
	
	writer = tf.summary.FileWriter('./graphs/linear_reg', sess.graph)
	
	# Step 8: train the model
	for i in range(50): # train the model 100 epochs
		total_loss = 0
		for x, y in data:
			# Session runs train_op and fetch values of loss
			_,l = sess.run([optimizer, loss],feed_dict={X:X_input, Y:Y_input})
			total_loss += l
		print('Epoch {0}: {1}'.format(i, total_loss/n_samples))

	# close the writer when you're done using it
	writer.close()
	# Step 9: output the values of w and b
	w_value,b_value,u_value = sess.run([W,b,u])

# plot the results
#X, Y = data.T[0], data.T[1]
X, Y = X_input, Y_input
plt.plot(X, Y, 'bo', label='Real data')
plt.plot(X, X * X*w_value + X*u_value+b_value, 'r', label='Predicted data')
plt.legend()
plt.show()
