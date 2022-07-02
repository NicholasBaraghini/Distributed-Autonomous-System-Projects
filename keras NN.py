import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp

np.random.seed(0)

''' IMPORT AND PRE-PROCESSING OF DATASET'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# shape of the numpy arrays
# print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# let's print the shape before we reshape and normalize
# print("X_train shape", X_train.shape)
# print("y_train shape", y_train.shape)
# print("X_test shape", X_test.shape)
# print("y_test shape", y_test.shape)

''' #plot one image from the training set
plt.imshow(X_train[3])
plt.show()
'''
# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# normalizing the data to help with the training
X_train /= 255
X_test /= 255

# print the final input shape ready for training
# print("Train matrix shape", X_train.shape)
# print("Test matrix shape", X_test.shape)

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
# print("Shape before one-hot encoding: ", y_train.shape)
Y_train_class = np_utils.to_categorical(y_train, n_classes)
Y_train_class[Y_train_class == 0] = -1  # substitute 0 with -1
Y_test_class = np_utils.to_categorical(y_test, n_classes)
Y_test_class[Y_test_class == 0] = -1  # substitute 0 with -1
# print("Shape after one-hot encoding: ", Y_train_class.shape)

# the images and labels now are in the correct format


''' GENERATION OF THE GRAPH '''
N_AGENTS = 3  # number og agents

###############################################################################
# Generate Network Binomial Graph
p_ER = 0.3
I_NN = np.eye(N_AGENTS)

for gn in range(100):
    G = nx.binomial_graph(N_AGENTS, p_ER)
    Adj = nx.adjacency_matrix(G)
    Adj = Adj.toarray()

    test = np.linalg.matrix_power((I_NN + Adj), N_AGENTS)

    if np.all(test > 0):
        print("the graph is connected")
        break
    else:
        print("the graph is NOT connected")

if 0:
    fig, ax = plt.subplots()
    ax = nx.draw(G, with_labels=True)
    plt.show()

###############################################################################
# Compute mixing matrices

WW = 1.5 * I_NN + 0.5 * Adj

ONES = np.ones((N_AGENTS, N_AGENTS))
ZEROS = np.zeros((N_AGENTS, N_AGENTS))

# normalize the rows and columns
cc = 0
while any(abs(np.sum(WW, axis=1) - 1)) > 10e-10:
    WW = WW / (WW @ ONES)
    WW = WW / (ONES @ WW)
    WW = np.abs(WW)
    cc += 1
    if cc > 100:
        break

with np.printoptions(precision=4, suppress=True):
    print('Check Stochasticity\n row:    {} \n column: {}'.format(
        np.sum(WW, axis=1),
        np.sum(WW, axis=0)
    ))

# print of matrix generated
# print(f"The matrix of the adjacency matrix weighted is: \r\n{WW}")

'''Normalizing the input data helps to speed up the training. Also, it reduces the chance of getting stuck in local 
optima, since we're using stochastic gradient descent to find the optimal weights for the network. '''

# close the figure
# plt.close(fig)

'''SET UP THE NEURAL NETWORK'''
###############################################################################

T = 5  # Layers
d = [784, 392, 196, 98, 10]  # Number of neurons in each layer. bias already considered

# Gradient-Tracking Method Parameters
MAX_ITERS = 20  # epochs
N_IMAGES = 15  # number of images
stepsize = 0.025  # learning rate

###############################################################################
# SPLITTING THE DATASET FOR EACH AGENT
data_point = []
label_point = []
for Agent in range(N_AGENTS):
    data_point.append(X_train[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])  # input sample
    label_point.append(Y_train_class[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])  # supervised
    # output


###############################################################################
# Activation Function
def sigmoid_fn(xi):
    return 1 / (1 + np.exp(-xi))


# Derivative of Activation Function
def sigmoid_fn_derivative(xi):
    return sigmoid_fn(xi) * (1 - sigmoid_fn(xi))


# Inference: x_tp = f(xt,ut)
def inference_dynamics(xt, ut, t):
    """
      input:
                xt current signal
                ut current weight matrix
      output:
                xtp next signal
    """
    # save temporarily the product between signal and weights
    temp = xt @ ut[1:, :] + ut[0, :]

    return sigmoid_fn(temp)


# Forward Propagation
def forward_pass(uu, x0):
    """
      input:
                uu input trajectory: u[0],u[1],..., u[T-1]
                x0 initial condition
      output:
                xx state trajectory: x[1],x[2],..., x[T]
    """
    xx = []
    for index in range(len(d)):  # create the signal structure
        xx.append(np.zeros((1, d[index])))

    # Input layer
    xx[0] = x0

    # compute the inference dynamics
    for t in range(T - 1):
        xx[t + 1] = inference_dynamics(xx[t], uu[t], t)  # x^+ = f(x,u)

    return xx


# Adjoint dynamics:
#   state:    lambda_t = A.T lambda_tp
# output: deltau_t = B.T lambda_tp
def adjoint_dynamics(ltp, xt, ut, t):
    """
      input:
                llambda_tp current costate
                xt current state
                ut current input
      output:
                llambda_t next costate
                delta_ut loss gradient wrt u_t
    """
    # Initialization
    Delta_ut = np.ones(ut.shape)
    d_sigma = np.zeros((1, d[t + 1]))

    # linear composition of neurons activations with the weights + bias
    temp = xt @ ut[1:, :] + ut[0, :]

    # compute the gradient of the activations
    for ell in range(d_sigma.shape[0]):
        d_sigma[ell] = sigmoid_fn_derivative(temp[ell])

    # compute df_dx
    AA = (ut[1:, :] * d_sigma)

    # compute df_du
    Delta_ut[0, :] = ltp * d_sigma  # bias term
    Delta_ut[1:, :] = np.tile(xt, (d[t + 1], 1)).T * (ltp * d_sigma)

    # costate vector at layer t
    lt = AA @ ltp.T

    return lt.T, Delta_ut


# Backward Propagation
def backward_pass(xx, uu, llambdaT):
    """
      input:
                xx state trajectory: x[1],x[2],..., x[T]
                uu input trajectory: u[0],u[1],..., u[T-1]
                llambdaT terminal condition
      output:
                llambda costate trajectory
                delta_u costate output, i.e., the loss gradient
    """
    # Costate structure init
    llambda = []
    for index in range(len(d)):  # create the signal structure
        llambda.append(np.zeros((1, d[index])))
    llambda[-1] = llambdaT
    # gradient structure of connection weights init
    Delta_u = []
    for index in range(len(d) - 1):  # create the signal structure
        Delta_u.append(np.zeros((1, d[index])))

    # run the adjoint dynamics to define the costate structure and the Delta_u structure
    for t in reversed(range(T - 1)):  # T-1,T-2,...,1,0
        llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t + 1], xx[t], uu[t], t)

    return Delta_u


# Cost Function
def Cost_Func(y_pred, y_true):
    """
          input:
                    y_pred: x[1],x[2],..., x[T]
                    y_true: u[0],u[1],..., u[T-1]
          output:
                    J costate trajectory
                    dJ costate output, i.e., the loss gradient
        """

    J = (y_pred - y_true) @ (y_pred - y_true).T  # it is the cost at k+1
    dJ = 2 * (y_pred - y_true)
    return J, dJ


def Cost_Function(Agent, kk):
    """
          input:
                    Agent, kk
          output:
                    J costate trajectory
                    dJ costate output, i.e., the loss gradient
    """
    # Initialize
    J = 0
    dJ = 0
    for sample in range(len(data_point[Agent])):
        # load the supervised sample
        data_pnt = data_point[Agent][sample].reshape(1, -1)  # input sample
        y_true = label_point[Agent][sample].reshape(1, -1)  # supervised output

        # compute the prediction
        xx = forward_pass(uu[Agent][kk], data_pnt)

        # adding the cost of the new sample
        J += (xx[-1] - y_true) @ (xx[-1] - y_true).T  # it is the cost at k+1
        dJ += 2 * (xx[-1] - y_true)

    return J, dJ


###############################################################

# GO!
#J = np.zeros(MAX_ITERS)  # Cost function

# Initial Weights / Initial Input Trajectory
uu = []
dummy = []
dummy2 = []
for index in range(len(d) - 1):
    dummy.append(np.random.randn(d[index] + 1, d[index + 1]))  # bias considered
for kk in range(MAX_ITERS):
    dummy2.append(dummy)
for Agent in range(N_AGENTS):
    uu.append(dummy2)

# Initial Gradient direction
DD = []
dummy = []
dummy2 = []
for index in range(len(d) - 1):
    dummy.append(np.zeros((d[index] + 1, d[index + 1])))  # bias considered
for kk in range(MAX_ITERS):
    dummy2.append(dummy)
for Agent in range(N_AGENTS):
    DD.append(dummy2)

# Initial Gradient of U
Du = DD

JJ = np.zeros(MAX_ITERS)
dJ = np.zeros(MAX_ITERS)
accuracy = np.zeros((N_AGENTS, MAX_ITERS))

print(f'iter', end=' ')
for ii in range(N_AGENTS):
    print(f'Agent{ii}', end=' ')
print('', end='\n')

'Cycle for each Epoch'
for kk in range(MAX_ITERS-1):
    'Cycle for each Agent - Computation of local model'
    for Agent in range(N_AGENTS):
        success = 0
        'Cycle for each sample of the dataset per each agent'
        for Image in range(len(data_point[Agent])):
            # temporarely extract the weight matrix of the cuurent agent
            # U_temp = uu[Agent][kk]

            "STARTING Neural Network"
            data_pnt = data_point[Agent][Image].reshape(1, -1)  # input sample
            label_pnt = label_point[Agent][Image].reshape(1, -1)  # supervised output

            # --> FORWARD PASS
            xx = forward_pass(uu[Agent][kk], data_pnt)  # forward simulation

            # --> BACKWARD PASS
            llambdaT = 2 * (xx[-1] - label_pnt)  # nabla J in last layer
            Delta_u = backward_pass(xx, uu[Agent][kk], llambdaT)  # the gradient of the loss function

            "ENDING Neural Network"
            # Averaging the Local Gradient of the weight matrix
            for index in range(len(d) - 1):
                Du[Agent][kk][index] += Delta_u[index]


            Y_true = np.argmax(label_pnt)
            Y_pred = np.argmax(xx[-1])

            if Y_true == Y_pred:
                success += 1

        accuracy[Agent, kk] = success / len(data_point[Agent])

    "GRADIENT TRAKING"
    for ii in range(N_AGENTS):
        # Return the indices of the elements of the Adjoint Matrix that are non-zero.
        Nii = np.nonzero(Adj[ii])[0]

        # Compute the descent for the iteration k
        for index in range(len(d) - 1):
            DD[ii][kk][index] = WW[ii, ii] * DD[ii][kk-1][index] + (Du[ii][kk][index] - Du[ii][kk - 1][index])
            # compute the Average Consensus of the descent
            for jj in Nii:
                DD[ii][kk][index] += WW[ii, jj] * DD[jj][kk-1][index]

        # Update the network weigths matrix with the descent
        for index in range(len(d) - 1):
            uu[ii][kk + 1][index] = WW[ii, ii] * uu[ii][kk][index] - stepsize * DD[ii][kk][index]
            # compute the Average Consensus of the Network Weights
            for jj in Nii:
                uu[ii][kk + 1][index] += WW[ii, jj] * uu[jj][kk][index]


        # Compute the cost for plotting
        JJk, dJk = Cost_Function(ii, kk)
        _, dJk_next = Cost_Function(ii, kk + 1)
        JJ[kk] += JJk

    if kk % 1 == 0:
        print(f'{kk}', end=' ')
        for ii in range(N_AGENTS):
            print(f'{np.round(accuracy[ii, kk], 2)*100}%', end=' ')
        print('', end='\n')

# Terminal iteration
for ii in range(N_AGENTS):
    JJk, dJk = Cost_Function(ii, -1)
    JJ[-1] += JJk

###############################################################################
# Figure 3 : Cost Error Evolution
if 1:
    plt.figure()
    plt.semilogy(np.arange(MAX_ITERS), np.abs(JJ), '--', linewidth=3)
    plt.xlabel(r"iterations $t$")
    plt.ylabel(r"$JJ$")
    plt.title("Evolution of the cost error")
    plt.grid()

plt.show()
