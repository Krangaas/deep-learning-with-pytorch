import numpy as np


def load_mnist(path):

	data = np.load(path + 'MNIST.npz')

	x_tr = data['a'].reshape(60000, 1, 28, 28)/255
	y_tr = data['b']

	x_te = data['c'].reshape(10000, 1, 28, 28)/255
	y_te = data['d']


	return x_tr, y_tr, x_te, y_te
