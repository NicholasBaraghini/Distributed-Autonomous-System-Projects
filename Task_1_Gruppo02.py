import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from keras.datasets import mnist
from keras.utils import np_utils


np.random.seed(seed=7)
#####################################################################################
'''SIMULATION PARAMETERS'''
# False : Multi-Classifier (it classifies the digit) True: Binary-Classifier (it classifies if it is the digit selected
# or not)
BINARY = False

N_AGENTS = 5  # number og agents
p_ER = 0.85  # probability of generate a connection

N_IMAGES = 80  # Images Per Agent

# Gradient-Tracking Method Parameters
MAX_ITERS = 10  # epochs
stepsize = 1  # learning rate
alpha = 0.25
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
def adjoint_dynamics(ltp, xt, ut, t, BINARY):
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
    ltp = np.asarray(ltp).reshape(1, -1)
    lt = ltp @ AA  # NB: ltp is a row !! -> A.T @ ltp.T = ltp @ A

    return lt, Delta_ut


# Backward Propagation
def backward_pass(xx, uu, llambdaT, BINARY):
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
        llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t + 1], xx[t], uu[t], t, BINARY)

    return Delta_u


# Cost Function
def Cost_Function(Prediction, Label, BINAY):
    """
          input:
                    Prediction : Output prediction of the class from the Neural Network
                    Label : Ground-truth label of the class
          output:
                    J cost functiion of an item
                    dJ gradient of J
    """
    if BINARY:
        J = (Prediction - Label) * (Prediction - Label)
        dJ = 2 * (Prediction - Label)
    else:
        J = (Prediction - Label) @ (Prediction - Label).T
        dJ = 2 * (Prediction - Label)

    return J, dJ


def Prediction_Performance(uu, Test_data, Test_label, BINARY):
    accuracy = np.zeros(N_AGENTS)

    for Agent in range(N_AGENTS):
        # Counter of correctly classified samples
        success = 0
        'Cycle for each sample of the dataset per each agent'
        for Image in range(len(Test_data[Agent])):
            # Extract the sample Imgage from the test set
            data_pnt = Test_data[Agent][Image].reshape(1, -1)  # input sample
            if BINARY:
                label_pnt = Test_label[Agent][Image]  # supervised output
            else:
                label_pnt = Test_label[Agent][Image].reshape(1, -1)  # supervised output

            # --> FORWARD PASS
            xx = forward_pass(uu[Agent][kk], data_pnt)  # forward simulation
            prediction = xx[-1]

            Y_true = np.argmax(label_pnt)
            Y_pred = np.argmax(prediction)
            if Y_true == Y_pred:
                success += 1

        # Accuracy on the test Set
        accuracy[Agent] = success / len(Test_data[Agent])

    return accuracy


# Dynamica step size to increase the performance of the model
def ADAM(g_t, s_t, Grad_cost, stepsize, Beta_1=0.9, Beta_2=0.999):
    g_tp = Beta_1 * g_t + (1 - Beta_1) * Grad_cost  # Momentum gained thanks to previus descent directions
    s_tp = Beta_2 * s_t + (1 - Beta_2) * Grad_cost * Grad_cost  # RMSprop contribution

    # Debiasing all terms
    g_debiased = g_tp / (1 - Beta_1)
    s_debiased = s_tp / (1 - Beta_2)

    # computing the new learning rate
    learning_rate = (stepsize / (np.sqrt(s_debiased) + 0.001)) * g_debiased

    return learning_rate, g_debiased, s_debiased


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
print('########################    GENERATION OF THE GRAPH    ##########################\n')
# Generate Network Binomial Graph
I_NN = np.eye(N_AGENTS)

for gn in range(100):
    Adj = np.random.binomial(1, p_ER, (N_AGENTS, N_AGENTS))
    Adj = np.logical_or(Adj, Adj.T)  # made it symmetric by doing logical or with the transpose
    Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)  # remove self loops and cast to int

    test = np.linalg.matrix_power(I_NN + Adj, N_AGENTS)  # check if the graph is connected
    if np.all(test > 0):
        print("the graph is connected\n")
        break
    else:
        print("the graph is NOT connected\n")

if False:
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

# print of matrix generated
print('#######################    WEIGHTED ADJIACENCY MATRIX    ########################\n')
print(f"\t\t\r{WW}\n")

with np.printoptions(precision=4, suppress=True):
    print('Check Stochasticity\n row:    {} \n column: {}\n'.format(
        np.sum(WW, axis=1),
        np.sum(WW, axis=0)
    ))

# _____________________________________________________________________________
''' SPLITTING THE DATASET FOR EACH AGENT '''

Train_data = []
Train_label = []
Test_data = []
Test_label = []
if BINARY:
    for Agent in range(N_AGENTS):
        Train_data.append(X_train[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])  # input sample
        Train_label.append(Y_train_class[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES])  # supervised

        Test_data.append(X_test[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])
        Test_label.append(Y_test_class[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES])
        # output
else:
    for Agent in range(N_AGENTS):
        Train_data.append(X_train[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])  # input sample
        Train_label.append(Y_train_class[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])  # supervised

        Test_data.append(X_test[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])
        Test_label.append(Y_test_class[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])
        # output

###############################################################################
'''SET UP THE NEURAL NETWORK'''

if BINARY:  # Binary classifier
    d = [784, 512, 512, 1]  # Number of neurons in each layer. bias already considered
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

# ADAM : Initial Momentum and RMSprop parameters
ss = np.zeros((N_AGENTS, MAX_ITERS))
gg = np.zeros((N_AGENTS, MAX_ITERS))
learning_rate = np.zeros((N_AGENTS, MAX_ITERS, T))

# Prep. of the printing
print('#########################    TRAINING SET COST ERROR    #########################\n')
print(f'iter', end=' ')
for Agent in range(N_AGENTS):
    print(f'Agent{Agent}', end=' ')
print('', end='\n')

# __________________________________INIT CYCLE____________________________________
'Cycle for each Agent - Computation of local model'
for Agent in range(N_AGENTS):
    # Return the indices of the elements of the Adjoint Matrix that are non-zero.
    Neighbours = np.nonzero(Adj[Agent])[0]
    'Cycle for each sample of the dataset per each agent'
    for Image in range(len(Train_data[Agent])):
        # Extract the sample Imgage from the training set
        data_pnt = Train_data[Agent][Image].reshape(1, -1)  # input sample
        if BINARY:
            label_pnt = Train_label[Agent][Image]  # supervised output
        else:
            label_pnt = Train_label[Agent][Image].reshape(1, -1)  # supervised output

        # --> FORWARD PASS
        xx = forward_pass(uu[Agent][0], data_pnt)  # forward simulation

        # --> BACKWARD PASS
        prediction = xx[-1]
        JJ_i_k, llambdaT = Cost_Function(prediction, label_pnt, BINARY)
        Delta_u = backward_pass(xx, uu[Agent][0], llambdaT, BINARY)  # the gradient of the loss function

        # update the cost vector and the gradient vector
        JJ[Agent, 0] += JJ_i_k
        if BINARY:
            dJ_norm[Agent, 0] += np.sqrt(llambdaT * llambdaT)
        else:
            dJ_norm[Agent, 0] += np.sqrt(llambdaT @ llambdaT.T)

        # Averaging the Local Gradient of the weight matrix
        for index in range(len(d) - 1):
            Du[Agent][0][index] += Delta_u[index] / len(Train_data[Agent])
            YY[Agent][0][index] += Delta_u[index] / len(Train_data[Agent])

    # Learning rate update
    lr, gg_tp, ss_tp = ADAM(gg[Agent, 0], ss[Agent, 0], Du[Agent][0][index], stepsize)
    learning_rate[Agent, 0, index] = np.sum(lr) / (len(lr.flatten().tolist()))
    gg[Agent, 1] = np.sum(gg_tp) / (len(gg_tp.flatten().tolist()))
    ss[Agent, 1] = np.sum(ss_tp) / (len(ss_tp.flatten().tolist()))

    for index in range(len(d) - 1):
        uu[Agent][1][index] = WW[Agent, Agent] * uu[Agent][0][index] - learning_rate[Agent, 0, index] * Du[Agent][0][
            index]
        # compute the Average Consensus of the Network Weights
        for neigh in Neighbours:
            uu[Agent][1][index] += WW[Agent, neigh] * uu[neigh][0][index]

# __________________________________CYCLE FOR EACH EPOCH____________________________________
'Cycle for each Epoch'
for kk in range(1, MAX_ITERS - 1):
    # Diminiscing Step-Size
    # stepsize = alpha ** kk
    'Cycle for each Agent - Computation of local model'
    for Agent in range(N_AGENTS):
        # Counter of correctly classified samples
        'Cycle for each sample of the dataset per each agent'
        for Image in range(len(Train_data[Agent])):
            # Extract the sample Imgage from the training set
            data_pnt = Train_data[Agent][Image].reshape(1, -1)  # input sample
            if BINARY:
                label_pnt = Train_label[Agent][Image]  # supervised output
            else:
                label_pnt = Train_label[Agent][Image].reshape(1, -1)  # supervised output

            # --> FORWARD PASS
            xx = forward_pass(uu[Agent][kk], data_pnt)  # forward simulation

            # --> BACKWARD PASS
            prediction = xx[-1]
            JJ_i_k, llambdaT = Cost_Function(prediction, label_pnt, BINARY)
            Delta_u = backward_pass(xx, uu[Agent][kk], llambdaT, BINARY)  # the gradient of the loss function

            # update the cost vector
            JJ[Agent, kk] += JJ_i_k
            # Averaging the Local Gradient of the weight matrix
            for index in range(len(d) - 1):
                Du[Agent][kk][index] += Delta_u[index] / len(Train_data[Agent])

        # Learning rate update
        lr, gg_tp, ss_tp = ADAM(gg[Agent, kk], ss[Agent, kk], Du[Agent][kk][index], stepsize)
        learning_rate[Agent, kk, index] = np.sum(lr) / (len(lr.flatten().tolist()))
        gg[Agent, kk + 1] = np.sum(gg_tp) / (len(gg_tp.flatten().tolist()))
        ss[Agent, kk + 1] = np.sum(ss_tp) / (len(ss_tp.flatten().tolist()))

    # Print The Actual Cost
    if kk % 1 == 0:
        print(f'{kk}', end=' ')
        for Agent in range(N_AGENTS):
            print(f'{np.round(JJ[Agent, kk], 2)}', end=' ')
        print('', end='\n')

    "GRADIENT TRAKING"
    if GT_YES:
        for Agent in range(N_AGENTS):
            # Return the indices of the elements of the Adjoint Matrix that are non-zero.
            Neighbours = np.nonzero(Adj[Agent])[0]

            # Compute the descent for the iteration k
            for index in range(len(d) - 1):
                YY[Agent][kk][index] = WW[Agent, Agent] * YY[Agent][kk - 1][index] + (
                        Du[Agent][kk][index] - Du[Agent][kk - 1][index])
                # compute the Average Consensus of the descent
                for neigh in Neighbours:
                    YY[Agent][kk][index] += WW[Agent, neigh] * YY[neigh][kk - 1][index]

            # Update the network weigths matrix with the descent
            for index in range(len(d) - 1):
                uu[Agent][kk + 1][index] = WW[Agent, Agent] * uu[Agent][kk][index] - learning_rate[Agent, kk, index] * \
                                           YY[Agent][kk][index]
                # compute the Average Consensus of the Network Weights
                for neigh in Neighbours:
                    uu[Agent][kk + 1][index] += WW[Agent, neigh] * uu[neigh][kk][index]

# Terminal iteration
'Cycle for each Agent - Computation of local model'
for Agent in range(N_AGENTS):
    # Counter of correctly classified samples
    'Cycle for each sample of the dataset per each agent'
    for Image in range(len(Train_data[Agent])):
        # Extract the sample Imgage from the training set
        data_pnt = Train_data[Agent][Image].reshape(1, -1)  # input sample
        if BINARY:
            label_pnt = Train_label[Agent][Image]  # supervised output
        else:
            label_pnt = Train_label[Agent][Image].reshape(1, -1)  # supervised output

        # --> FORWARD PASS
        xx = forward_pass(uu[Agent][MAX_ITERS - 1], data_pnt)  # forward simulation

        # --> BACKWARD PASS
        prediction = xx[-1]
        JJ_i_final, llambdaT = Cost_Function(prediction, label_pnt, BINARY)

        # update the cost vector and the gradient vector
        JJ[Agent, MAX_ITERS - 1] += JJ_i_final

    dJ_norm[Agent, :] = np.abs(np.gradient(JJ[Agent]))

# Print the test accuracy
print('############################\tTEST SET ACCURACY\t############################\n')
print(f'iter', end=' ')
for Agent in range(N_AGENTS):
    print(f'Agent{Agent}', end=' ')
print('', end='\n')
# Compute the Test Accuracy
Accuracy = Prediction_Performance(uu, Test_data, Test_label, BINARY)
for Agent in range(N_AGENTS):
    print(f'{np.round(Accuracy[Agent], 2)}%', end=' ')
print('', end='\n')

###############################################################################
lpf = "C:/Users/barag/Documents/GitHub/Distributed-Autonomous-System-Projects"
# Figure 1 : Cost Error Evolution
CostErrorEvolution_YES = True
if CostErrorEvolution_YES:
    plt.figure()
    for Agent in range(N_AGENTS):
        plt.semilogy(np.arange(MAX_ITERS), JJ[Agent, :].flatten(), '--', linewidth=3)
    plt.xlabel(r"Epochs")
    # plt.ylabel(r"$J$")
    plt.title(r"Prediction Error $= \sum((y_{pred} - y_{true})^{T}(y_{pred} - y_{true}))$")
    plt.grid()
    plt.savefig(lpf + f"/plot/task1/Cost_Error_{N_AGENTS}_{MAX_ITERS}_{GT_YES}.jpg", transparent=True)

plt.show()

###############################################################################
# Figure 2 : Norm Gradient Error Evolution
NormGradientErrorEvolution_YES = True
if NormGradientErrorEvolution_YES:
    plt.figure()
    for Agent in range(N_AGENTS):
        plt.semilogy(np.arange(MAX_ITERS), dJ_norm[Agent, :].flatten(), '--', linewidth=3)
    plt.xlabel(r"iterations $t$")
    # plt.ylabel(r"$JJ$")
    plt.title(r"Prediction Error Gradient Norm $= \sqrt{\nabla J^{T} \nabla J}$ ")
    plt.grid()
    plt.savefig(lpf + f"/plot/task1/Norm_Grad_Error_{N_AGENTS}_{MAX_ITERS}_{GT_YES}.jpg", transparent=True)
plt.show()

###############################################################################
# Figure 4 : ADAM learning rate evolution
ADAM_lr_plot = True
if ADAM_lr_plot:
    plt.figure()
    for Agent in range(N_AGENTS):
        plt.semilogy(np.arange(MAX_ITERS), learning_rate[Agent, :, T - 1], '--', linewidth=3)
    plt.xlabel(r"iterations $t$")
    # plt.ylabel(r"$JJ$")
    plt.title(r"Learning rate Evolution per agent$ ")
    plt.grid()
    plt.savefig(lpf + f"/plot/task1/LearningRate_{N_AGENTS}_{MAX_ITERS}_{GT_YES}.jpg", transparent=True)
plt.show()

###############################################################################
# Figure 3 : Consensus in Matrix of Weights
ConsensusWeights_YES = False
if ConsensusWeights_YES:
    plt.figure()
    uit = np.zeros((N_AGENTS, MAX_ITERS))
    for Agent in range(N_AGENTS):
        for t in range(MAX_ITERS):
            uuu = uu[Agent][t][1][1][1]
            uit[Agent, t] = uuu

        plt.semilogy(np.arange(MAX_ITERS), uit[Agent, :], '--', linewidth=3)

    plt.xlabel(r"iterations $t$")
    # plt.ylabel(r"$JJ$")
    plt.title(r"Consensus in Matrix of Weights : element $U[1,1]$ ")
    plt.grid()
    plt.savefig(lpf + f"/plot/task1/WeightConsensus_{N_AGENTS}_{MAX_ITERS}_{GT_YES}.jpg", transparent=True)
plt.show()
