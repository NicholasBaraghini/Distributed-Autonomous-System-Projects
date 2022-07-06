import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
import networkx as nx
import scipy as sp

np.random.seed(seed=7)
#####################################################################################
'''SIMULATION PARAMETERS'''
# False : Multi-Classifier (it classifies the digit) True: Binary-Classifier (it classifies if it is the digit selected
# or not)
BINARY = True

N_AGENTS = 10  # number og agents
p_ER = 0.8  # probability of generate a connection

N_IMAGES = 100  # Images Per Agent

# Gradient-Tracking Method Parameters
MAX_ITERS = 10  # epochs
stepsize = 0.035  # learning rate
GT_YES = True  # Enable Gradient tracking

#####################################################################################
''''USEFULL FUNCTIONS'''


# Activation Function
def sigmoid_fn(xi):
    return 1 / (1 + np.exp(-xi))


def ReLu(xi):
    return xi * (xi > 0)


# Derivative of Activation Function
def sigmoid_fn_derivative(xi):
    return sigmoid_fn(xi) * (1 - sigmoid_fn(xi))


def ReLu_derivative(xi):
    return xi > 0


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
    biases = ut[:, 0].reshape(1, -1)
    con_weight = ut[:, 1:].T
    temp = xt @ con_weight + biases
    xtp = sigmoid_fn(temp)  # ReLu(temp)

    return xtp


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


# Adjoint Dynamics
def adjoint_dynamics(ltp, xt, ut, t):
    """
      input:
                llambda_tp current costate
                xt current state
                ut current input
      output:
                llambda_t next costate
                delta_ut loss gradient wrt u_t

     Adjoint dynamics:
       state:    lambda_t = A.T lambda_tp
       output: deltau_t = B.T lambda_tp
    """
    # Initialization
    Delta_ut = np.ones(ut.shape)
    # d_sigma = np.zeros((1, d[t + 1]))

    # linear composition of neurons activations with the weights + bias
    biases = ut[:, 0].reshape(1, -1)
    con_weight = ut[:, 1:].T
    temp = xt @ con_weight + biases

    # compute the gradient of the activations
    d_sigma = sigmoid_fn_derivative(temp)  # ReLu_derivative(temp[ell])

    # compute df_dx
    AA = (con_weight * d_sigma).T

    # compute df_du
    Bias_term = (ltp * d_sigma).reshape(1, -1)
    Delta_ut[:, 0] = Bias_term  # bias term
    Tile_Matrix = np.tile(xt, (d[t + 1], 1))
    Delta_ut[:, 1:] = Tile_Matrix * Bias_term.T

    # costate vector at layer t
    lt = ltp @ AA  # NB: ltp is a row !! -> A.T @ ltp.T = ltp @ A

    return lt, Delta_ut


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
    for Image in range(N_IMAGES):
        # load the supervised sample
        data_pnt = data_test[Agent][Image].reshape(1, -1)  # input sample
        if BINARY:
            y_true = label_test[Agent][Image]  # supervised output
        else:
            y_true = label_test[Agent][Image].reshape(1, -1)  # supervised output

        # compute the prediction
        xx = forward_pass(uu[Agent][kk], data_pnt)

        # adding the cost of the new sample
        J += (xx[-1] - y_true) @ (xx[-1] - y_true).T  # it is the cost at k+1
        dJ += 2 * (xx[-1] - y_true)

    return J, dJ


#####################################################################################
'''SELECTED TYPE OF CLASSIFIER'''

''' IMPORT AND PRE-PROCESSING OF DATASET'''
# LOADING the Mnist Dataset
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# building the input vector from the 28x28 pixels
X_train = X_train.reshape(60000, 784)
X_test = X_test.reshape(10000, 784)
X_train = X_train.astype('float32')
X_test = X_test.astype('float32')

# DATA NORMALIZATION
'''Normalizing the input data helps to speed up the training. Also, it reduces the chance of getting stuck in local 
optima, since we're using stochastic gradient descent to find the optimal weights for the network. '''
X_train /= 255
X_test /= 255

# LABELS MAPPING according to the classification problem we are solving (if binary or multiclass
# classification)
if BINARY:
    # digit to be classified
    CLASS_IDENTIFIED = 4
    # build the label such that it will be 1 if the image contains the digit chones by CLASS_IDENTIFIED
    Y_train_class = [1 if y == CLASS_IDENTIFIED else -1 for y in y_train]
    Y_test_class = [1 if y == CLASS_IDENTIFIED else -1 for y in y_test]

else:
    # one-hot encoding using keras' numpy-related utilities
    # number of classes inside the dataset
    n_classes = 10
    Y_train_class = np_utils.to_categorical(y_train, n_classes)
    # Y_train_class[Y_train_class == 0] = -1  # substitute 0 with -1
    Y_test_class = np_utils.to_categorical(y_test, n_classes)
    # Y_test_class[Y_test_class == 0] = -1  # substitute 0 with -1

'''  the images and labels now are in the correct format '''

###############################################################################
''' GENERATION OF THE GRAPH '''

# Generate Network Binomial Graph
I_NN = np.eye(N_AGENTS)

for gn in range(100):
    Adj = np.random.binomial(1, p_ER, (N_AGENTS, N_AGENTS))
    Adj = np.logical_or(Adj, Adj.T)  # made it symmetric by doing logical or with the transpose
    Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)  # remove self loops and cast to int

    test = np.linalg.matrix_power(I_NN + Adj, N_AGENTS)  # check if the graph is connected
    if np.all(test > 0):
        print("the graph is connected")
        break
    else:
        print("the graph is NOT connected")

if 0:
    fig, ax = plt.subplots()
    ax = nx.draw(Adj, with_labels=True)
    plt.show()
# _____________________________________________________________________________
'''COMPUTE MIXING MATRICES'''

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
print(f"The matrix of the adjacency matrix weighted is: \r\n{WW}\n\n")

# _____________________________________________________________________________
''' SPLITTING THE DATASET FOR EACH AGENT '''

data_point = []
label_point = []
data_test = []
label_test = []
if BINARY:
    for Agent in range(N_AGENTS):
        data_point.append(X_train[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])  # input sample
        label_point.append(Y_train_class[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES])  # supervised

        data_test.append(X_test[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])
        label_test.append(Y_test_class[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES])
        # output
else:
    for Agent in range(N_AGENTS):
        data_point.append(X_train[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])  # input sample
        label_point.append(Y_train_class[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])  # supervised

        data_test.append(X_test[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])
        label_test.append(Y_test_class[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])
        # output

###############################################################################
'''SET UP THE NEURAL NETWORK'''

if BINARY:  # Binary classifier
    d = [784, 512, 512, 2]  # Number of neurons in each layer. bias already considered
else:  # Multiclass Classifier
    d = [784, 512, 512, 10]  # Number of neurons in each layer. bias already considered

# Number of Layers
T = len(d)
###############################################################################
''' START THE ALGORITHM'''

# Initial Weights / Initial Input Trajectory
uu = []
for Agent in range(N_AGENTS):
    uu.append([])
    for kk in range(MAX_ITERS):
        uu[Agent].append([])
        for index in range(len(d) - 1):
            if kk == 0:
                uu[Agent][kk].append(np.random.randn(d[index + 1], d[index] + 1))  # bias considered
            else:
                uu[Agent][kk].append(np.zeros((d[index + 1], d[index] + 1)))  # bias considered

# Initial Gradient direction
YY = []
for Agent in range(N_AGENTS):
    YY.append([])
    for kk in range(MAX_ITERS):
        YY[Agent].append([])
        for index in range(len(d) - 1):
            YY[Agent][kk].append(np.zeros((d[index + 1], d[index] + 1)))  # bias considered

# Initial Gradient of U
Du = []
for Agent in range(N_AGENTS):
    Du.append([])
    for kk in range(MAX_ITERS):
        Du[Agent].append([])
        for index in range(len(d) - 1):
            Du[Agent][kk].append(np.zeros((d[index + 1], d[index] + 1)))  # bias considered

# Initialization of the Cost Function, its Gradient and the accuracy
JJ = np.zeros((N_AGENTS, MAX_ITERS))
dJ_norm = np.zeros((N_AGENTS, MAX_ITERS))
accuracy = np.zeros((N_AGENTS, MAX_ITERS))

# Prep. of the printing
print('__TRAINING SET ACCURACY__')
print(f'iter', end=' ')
for ii in range(N_AGENTS):
    print(f'Agent{ii}', end=' ')
print('', end='\n')

# Initialize the Discent Du
for ii in range(N_AGENTS):
    # Return the indices of the elements of the Adjoint Matrix that are non-zero.
    Nii = np.nonzero(Adj[ii])[0]
    for Image in range(len(data_point[Agent])):
        # Extract the sample Imgage from the training set
        data_pnt = data_point[ii][Image].reshape(1, -1)  # input sample
        if BINARY:
            label_pnt = label_point[ii][Image]  # supervised output
        else:
            label_pnt = label_point[ii][Image].reshape(1, -1)  # supervised output
        "STARTING Neural Network"
        # --> FORWARD PASS
        xx = forward_pass(uu[ii][0], data_pnt)  # forward simulation

        # --> BACKWARD PASS
        prediction = xx[-1]
        llambdaT = 2 * (prediction - label_pnt)  # nabla J in last layer
        Delta_u = backward_pass(xx, uu[ii][0], llambdaT)  # the gradient of the loss function
        "ENDING Neural Network"

        # Averaging the Local Gradient of the weight matrix
        for index in range(len(d) - 1):
            Du[ii][0][index] += Delta_u[index] / len(data_point[Agent])
            YY[ii][0][index] += Delta_u[index] / len(data_point[Agent])

    for index in range(len(d) - 1):
        uu[ii][1][index] = WW[ii, ii] * uu[ii][0][index] - stepsize * Du[ii][0][index]
        # compute the Average Consensus of the Network Weights
        for jj in Nii:
            uu[ii][1][index] += WW[ii, jj] * uu[jj][0][index]

    # --> FORWARD PASS
    # xx = forward_pass(uu[ii][1], data_pnt)  # forward simulation

'Cycle for each Epoch'
for kk in range(1, MAX_ITERS - 1):
    # Diminiscing Step-Size
    # stepsize = 10 / kk
    'Cycle for each Agent - Computation of local model'
    for Agent in range(N_AGENTS):
        # Counter of correctly classified samples
        success = 0
        'Cycle for each sample of the dataset per each agent'
        for Image in range(len(data_point[Agent])):
            # Extract the sample Imgage from the training set
            data_pnt = data_point[Agent][Image].reshape(1, -1)  # input sample
            if BINARY:
                label_pnt = label_point[Agent][Image]  # supervised output
            else:
                label_pnt = label_point[Agent][Image].reshape(1, -1)  # supervised output

            "STARTING Neural Network"
            # --> FORWARD PASS
            xx = forward_pass(uu[Agent][kk], data_pnt)  # forward simulation

            # --> BACKWARD PASS
            prediction = xx[-1]
            llambdaT = 2 * (prediction - label_pnt)  # nabla J in last layer
            Delta_u = backward_pass(xx, uu[Agent][kk], llambdaT)  # the gradient of the loss function
            "ENDING Neural Network"

            # Averaging the Local Gradient of the weight matrix
            for index in range(len(d) - 1):
                Du[Agent][kk][index] += Delta_u[index] / len(data_point[Agent])

            # Compute the prediction
            # xx = forward_pass(uu[Agent][kk + 1], data_pnt)  # forward simulation

            Y_true = np.argmax(label_pnt)
            Y_pred = np.argmax(xx[-1])
            if Y_true == Y_pred:
                success += 1

        # Accuracy of The training Set
        accuracy[Agent, kk] = success / len(data_point[Agent])

    "GRADIENT TRAKING"
    if GT_YES:
        for ii in range(N_AGENTS):
            # Return the indices of the elements of the Adjoint Matrix that are non-zero.
            Nii = np.nonzero(Adj[ii])[0]

            # Compute the descent for the iteration k
            for index in range(len(d) - 1):
                YY[ii][kk][index] = WW[ii, ii] * YY[ii][kk - 1][index] + (Du[ii][kk][index] - Du[ii][kk - 1][index])
                # compute the Average Consensus of the descent
                for jj in Nii:
                    YY[ii][kk][index] += WW[ii, jj] * YY[jj][kk - 1][index]

            # Update the network weigths matrix with the descent
            for index in range(len(d) - 1):
                uu[ii][kk + 1][index] = WW[ii, ii] * uu[ii][kk][index] - stepsize * YY[ii][kk][index]
                # compute the Average Consensus of the Network Weights
                for jj in Nii:
                    uu[ii][kk + 1][index] += WW[ii, jj] * uu[jj][kk][index]
    if kk % 1 == 0:
        print(f'{kk}', end=' ')
        for ii in range(N_AGENTS):
            print(f'{np.round(accuracy[ii, kk] * 100, 2)}%', end=' ')
        print('', end='\n')

# Terminal iteration
for kk in range(0, MAX_ITERS):
    for ii in range(N_AGENTS):
        # Compute the cost for plotting
        JJk_i, dJk_i = Cost_Function(ii, kk)
        dJ_norm[ii, kk] = np.sqrt(dJk_i @ dJk_i.T)
        JJ[ii, kk] += np.abs(JJk_i)

###############################################################################
lpf = "C:/Users/barag/Documents/GitHub/Distributed-Autonomous-System-Projects"
# Figure 1 : Cost Error Evolution
CostErrorEvolution_YES = True
if CostErrorEvolution_YES:
    plt.figure()
    for ii in range(N_AGENTS):
        plt.semilogy(np.arange(MAX_ITERS), JJ[ii, :], '--', linewidth=3)
    plt.xlabel(r"iterations $t$")
    plt.ylabel(r"$JJ$")
    plt.title("Evolution of the cost error")
    plt.grid()
    plt.savefig(lpf + f"/plot/task1/Cost_Error_{N_AGENTS}_{MAX_ITERS}_{GT_YES}.jpg", transparent=True)

plt.show()

###############################################################################
# Figure 2 : Norm Gradient Error Evolution
NormGradientErrorEvolution_YES = True
if NormGradientErrorEvolution_YES:
    plt.figure()
    for ii in range(N_AGENTS):
        plt.semilogy(np.arange(MAX_ITERS), dJ_norm[ii, :], '--', linewidth=3)
    plt.xlabel(r"iterations $t$")
    plt.ylabel(r"$JJ$")
    plt.title("Norm Gradient Error Evolution")
    plt.grid()
    plt.savefig(lpf + f"/plot/task1/Norm_Grad_Error_{N_AGENTS}_{MAX_ITERS}_{GT_YES}.jpg", transparent=True)
plt.show()
