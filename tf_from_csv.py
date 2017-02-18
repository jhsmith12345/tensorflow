import tensorflow as tf
import pandas as pd

dataframe = pd.read_csv("train.csv") # Let's have Pandas load our dataset as a dataframe
dataframe.loc[:, ("y2")] = dataframe["y"] == 0           # y2 is the negation of y1
dataframe.loc[:, ("y2")] = dataframe["y2"].astype(int)    # Turn TRUE/FALSE values into 1/0
trainX = dataframe.loc[:, ['a', 'b']].as_matrix()
trainY = dataframe.loc[:, ["y", 'y2']].as_matrix()

dataframe = pd.read_csv("test.csv") # Let's have Pandas load our dataset as a dataframe
dataframe.loc[:, ("y2")] = dataframe["y"] == 0           # y2 is the negation of y1
dataframe.loc[:, ("y2")] = dataframe["y2"].astype(int)    # Turn TRUE/FALSE values into 1/0
testX = dataframe.loc[:, ['a', 'b']].as_matrix()
testY = dataframe.loc[:, ["y", 'y2']].as_matrix()

n_nodes_hl1 = 10
n_nodes_hl2 = 10
n_nodes_hl3 = 10

n_classes = 2
batch_size = 1

x = tf.placeholder('float',[None, 2])
y = tf.placeholder('float')

def neural_network_model(data):
	hidden_1_layer = {'weights':tf.Variable(tf.random_normal([2, n_nodes_hl1])),
				      'biases':tf.Variable(tf.random_normal([n_nodes_hl1]))}

	hidden_2_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_nodes_hl2])),
				      'biases':tf.Variable(tf.random_normal([n_nodes_hl2]))}

	hidden_3_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl2, n_nodes_hl3])),
				      'biases':tf.Variable(tf.random_normal([n_nodes_hl3]))}

	output_layer = {'weights':tf.Variable(tf.random_normal([n_nodes_hl1, n_classes])),
				      'biases':tf.Variable(tf.random_normal([n_classes]))}
	l1 = tf.add(tf.matmul(data, hidden_1_layer['weights']), hidden_1_layer['biases'])
	l1 = tf.nn.relu(l1)

	l2 = tf.add(tf.matmul(l1, hidden_2_layer['weights']), hidden_2_layer['biases'])
	l2 = tf.nn.relu(l2)

	l3 = tf.add(tf.matmul(l2, hidden_3_layer['weights']), hidden_3_layer['biases'])
	l3 = tf.nn.relu(l3)

	output = tf.matmul(l3, output_layer['weights']) + output_layer['biases']	
	return output

def train_neural_network(x):
	prediction = neural_network_model(x)
	cost = tf.reduce_mean(tf.nn.softmax_cross_entropy_with_logits(prediction,y))
	optimizer = tf.train.AdamOptimizer().minimize(cost)

	hm_epochs = 10

	with tf.Session() as sess:
		sess.run(tf.initialize_all_variables())

		for epoch in range(hm_epochs):
			epoch_loss = 1
			for _ in range(10):
				_, c = sess.run([optimizer, cost], feed_dict = {x: trainX, y: trainY})
				epoch_loss += c
			print('Epoch', epoch, 'completed out of', hm_epochs, 'loss:', epoch_loss)
		
		correct = tf.equal(tf.argmax(prediction,1), tf.argmax(y,1))
		accuracy = tf.reduce_mean(tf.cast(correct, 'float'))
		print('Accuracy:',accuracy.eval({x: testX, y: testY}))
		classification = prediction.eval(feed_dict={x: [[9,3]]})
		print (classification)
train_neural_network(x)

