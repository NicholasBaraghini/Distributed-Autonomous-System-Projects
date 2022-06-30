import numpy as np
import matplotlib.pyplot as plt
import networkx as nx

np.random.seed(1)


#lallallala
NN = 5 #number of agents

###############################################################################
# Generate Network Binomial Graph
p_ER = 0.3
I_NN = np.eye(NN)
while 1:
	Adj = np.random.binomial(1, p_ER, (NN,NN))
	Adj = np.logical_or(Adj, Adj.T)                           # made it symmetric by doing logical or with the transpose
	Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)  # remove self loops and cast to int

	test = np.linalg.matrix_power(I_NN+Adj, NN)  # check if the graph is connected
	if np.all(test>0):  # check the non zero condition of the graph
		break
###############################################################################
# Compute mixing matrices

WW = 1.5 * I_NN + 0.5 * Adj

ONES = np.ones((NN,NN))
ZEROS = np.zeros((NN,NN))

# normalize the rows and columns
while any(abs(np.sum(WW, axis=1)-1))> 10e-10:
	WW = WW/(WW@ONES)
	WW = WW/(ONES@WW)
	WW = np.abs(WW)

'''
Normalizing the input data helps to speed up the training. Also, it reduces the chance of getting stuck in local optima, since we're using stochastic gradient descent to find the optimal weights for the network.
'''
