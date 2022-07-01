
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
np.random.seed(1)

''' GENERATION OF THE GRAPH '''
NN = 6  # number of agents

###############################################################################
# Generate Network Binomial Graph
p_ER = 0.3
I_NN = np.eye(NN)

while 1:
	G = nx.binomial_graph(NN, p_ER)
	Adj = nx.adjacency_matrix(G)
	Adj = Adj.toarray()

	test = np.linalg.matrix_power((I_NN + Adj), NN)

	if np.all(test > 0):
		print("the graph is connected")
		break
	else:
		print("the graph is NOT connected")
'''
	Adj = np.random.binomial(1, p_ER, (NN,NN))
	Adj = np.logical_or(Adj, Adj.T)                           # made it symmetric by doing logical or with the transpose
	Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)  # remove self loops and cast to int

	test = np.linalg.matrix_power(I_NN+Adj, NN)  # check if the graph is connected
	if np.all(test>0):  # check the non zero condition of the graph
		print("the graph is connected\n")
		break
	else:
		print("the graph is NOT connected\n")
		#quit()
'''
fig, ax = plt.subplots()
ax = nx.draw(G, with_labels=True)
plt.show()

###############################################################################
# Compute mixing matrices

WW = 1.5 * I_NN + 0.5 * Adj

ONES = np.ones((NN, NN))
ZEROS = np.zeros((NN, NN))

# normalize the rows and columns
while any(abs(np.sum(WW, axis=1)-1)) > 10e-10:
	WW = WW/(WW@ONES)
	WW = WW/(ONES@WW)
	WW = np.abs(WW)

with np.printoptions(precision=4, suppress=True):
    print('Check Stochasticity\n row:    {} \n column: {}'.format(
        np.sum(WW, axis=1),
        np.sum(WW, axis=0)
    ))

# print of matrix generated
# print(f"The matrix of the adjacency matrix weighted is: \r\n{WW}")

'''
Normalizing the input data helps to speed up the training. Also, it reduces the chance of getting stuck in local optima, since we're using stochastic gradient descent to find the optimal weights for the network.
'''

# close the figure
plt.close(fig)
