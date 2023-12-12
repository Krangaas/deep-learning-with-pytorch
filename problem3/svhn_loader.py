# %%
import numpy as np
import scipy.io

def load_svhn(path):

	data = np.load(path + 'SVHN.npz')

	x_tr = data['a']
	y_tr = data['b']

	x_te = data['c']
	y_te = data['d']

	return x_tr, y_tr, x_te, y_te
