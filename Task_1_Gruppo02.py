import matplotlib.pyplot as plt
import numpy as np
import networkx as nx
from keras.datasets import mnist
from keras.utils import np_utils, losses_utils
from imblearn.under_sampling import RandomUnderSampler

np.random.seed(seed=7)
#####################################################################################
'''SIMULATION PARAMETERS'''
# Digit to be Identified
TARGET_CLASS = 7

N_AGENTS = 8  # number og agents
p_ER = 0.6  # probability of generate a connection

N_IMAGES = 60  # Images Per Agent

# Gradient-Tracking Method Parameters
MAX_ITERS = 100  # epochs
stepsize = 0.95  # nominal learning rate
ADAM_Yes = False  # Activate the ADAM algorithm to computre the optimal learning rate or use a standard diminishing learning rate

GT_YES = True  # Enable Gradient tracking

#####################################################################################
''''USEFULL FUNCTIONS'''


def Activation_Function(xi):
    return 1 / (1 + np.exp(-xi))  # sigmoid
    # return xi * (xi > 0) + 0.1  # ReLu


# Derivative of Activation Function
def Activation_Function_Derivative(xi):
    return Activation_Function(xi) * (1 - Activation_Function(xi))  # sigmoid
    # return xi > 0


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
    xtp = Activation_Function(temp)

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
    for layer in range(len(d)):  # create the signal structure
        xx.append(np.zeros((1, d[layer])))

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
    d_sigma = Activation_Function_Derivative(temp)

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
    for layer in range(len(d)):  # create the signal structure
        llambda.append(np.zeros((1, d[layer])))
    llambda[-1] = llambdaT
    # gradient structure of connection weights init
    Delta_u = []
    for layer in range(len(d) - 1):  # create the signal structure
        Delta_u.append(np.zeros((1, d[layer])))

    # run the adjoint dynamics to define the costate structure and the Delta_u structure
    for t in reversed(range(T - 1)):  # T-1,T-2,...,1,0
        llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t + 1], xx[t], uu[t], t)

    return Delta_u


# Cost Function
def Cost_Function(Prediction, Label):
    """
          input:
                    Prediction : Output prediction of the class from the Neural Network
                    Label : Ground-truth label of the class
          output:
                    J cost functiion of an item
                    dJ gradient of J
    """

    # Cross Entropy For the Binary Classification adapted for class label -1 and 1
    Y = (Label + 1) / 2
    J = -(Y * np.log(prediction) + (1 - Y) * np.log(1 - prediction))  # (Prediction - Label) * (Prediction - Label)
    # Gradient of the Cross Entropy
    dJ = (Prediction - Y) / (Prediction * (1 - prediction) + 0.00001)

    return J, dJ


def Prediction_Performance(uu, Test_data, Test_label):
    N_TEST_SAMPLES = len(Test_data)
    accuracy = np.zeros(N_AGENTS)
    predictions = np.zeros((N_AGENTS, N_TEST_SAMPLES))
    for Agent in range(N_AGENTS):
        # Counter of correctly classified samples
        'Cycle for each sample of the dataset per each agent'
        for Image in range(N_TEST_SAMPLES):
            # Extract the sample Imgage from the test set
            data_pnt = Test_data[Image].reshape(1, -1)  # input sample

            label_pnt = Test_label[Image]  # supervised output

            # --> FORWARD PASS
            xx = forward_pass(uu[Agent][MAX_ITERS - 1], data_pnt)  # forward simulation

            if xx[-1] <= 0:
                prediction = -1
            else:
                prediction = 1

            if prediction == label_pnt:
                # Accuracy on the test Set
                accuracy[Agent] += 1

            predictions[Agent, Image] = prediction

    return accuracy / N_TEST_SAMPLES, predictions


# Dynamica step size to increase the performance of the model
def ADAM(g_t, s_t, Grad_cost, stepsize, Beta_1=0.9, Beta_2=0.999):
    g_tp = Beta_1 * g_t + (1 - Beta_1) * Grad_cost  # Momentum gained thanks to previus descent directions
    s_tp = Beta_2 * s_t + (1 - Beta_2) * Grad_cost * Grad_cost  # RMSprop contribution

    # Debiasing all terms
    g_debiased = g_tp / (1 - Beta_1)
    s_debiased = s_tp / (1 - Beta_2)

    # computing the new learning rate
    learning_rate = (stepsize / (np.sqrt(s_debiased) + 0.001)) * g_debiased

    # print(f'{learning_rate} learning_rate')

    return learning_rate, g_debiased, s_debiased


#####################################################################################
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

print('#########################    DATASET BALANCE CHECK    ###########################\n')
# build the label such that it will be 1 if the image contains the TARGET_CLASS digit, -1 otherwise
Y_train_class = [-1 if y == TARGET_CLASS else 1 for y in y_train]
Y_test_class = [-1 if y == TARGET_CLASS else 1 for y in y_test]
# Balancing the training set

RandomUS = RandomUnderSampler()
X_train_bal, y_train_bal = RandomUS.fit_resample(X_train, Y_train_class)
X_test_bal, y_test_bal = RandomUS.fit_resample(X_test, Y_test_class)

print(f' TRAINING SET BALANCED ENTROPHY = {np.sum(y_train_bal) / len(y_train_bal)}')
print(f' TEST SET BALANCED ENTROPHY = {np.sum(y_test_bal) / len(y_test_bal)}')

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
# Adding the self-loops in the Adj. Matric to obtain Periodicity of the Communication Net
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
    print('Check Doubly Stochasticity\n row:    {} \n column: {}\n'.format(
        np.sum(WW, axis=1),
        np.sum(WW, axis=0)
    ))
'''' NB: If the Grap is Periodic, Strongly Connected and Doubly Stochastic then Average Consensus is guaranteed!!'''

###############################################################################
''' SPLITTING THE DATASET FOR EACH AGENT '''
Train_data = []
Train_label = []
# Test data do not need to be separated among thw agents as the final model reach consensus and the test data are used
# only to evaluate the performance of the network
Test_data = X_test
Test_label = Y_test_class

for Agent in range(N_AGENTS):
    Train_data.append(X_train_bal[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES, :])  # input sample
    Train_label.append(y_train_bal[(Agent * N_IMAGES):(Agent + 1) * N_IMAGES])  # supervised

###############################################################################
'''SET UP THE NEURAL NETWORK'''
# Number of neurons in each layer. bias already considered
d = [784, 512, 512, 250, 10, 1]

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
        for layer in range(len(d) - 1):
            if kk == 0:
                uu[Agent][kk].append(np.random.randn(d[layer + 1], d[layer] + 1) * 10)  # bias considered
            else:
                uu[Agent][kk].append(np.zeros((d[layer + 1], d[layer] + 1)))  # bias considered

# Initial Gradient direction
YY = []
for Agent in range(N_AGENTS):
    YY.append([])
    for kk in range(MAX_ITERS):
        YY[Agent].append([])
        for layer in range(len(d) - 1):
            YY[Agent][kk].append(np.zeros((d[layer + 1], d[layer] + 1)))  # bias considered

# Initial Gradient of U
Du = []
for Agent in range(N_AGENTS):
    Du.append([])
    for kk in range(MAX_ITERS):
        Du[Agent].append([])
        for layer in range(len(d) - 1):
            Du[Agent][kk].append(np.zeros((d[layer + 1], d[layer] + 1)))  # bias considered

# Initialization of the Cost Function, its Gradient and the accuracy
JJ = np.zeros((N_AGENTS, MAX_ITERS))
dJ_norm = np.zeros((N_AGENTS, MAX_ITERS))
JJ_Tot = np.zeros(MAX_ITERS)

# ADAM : Initial Momentum and RMSprop parameters
ss = np.zeros((N_AGENTS, MAX_ITERS))
gg = np.zeros((N_AGENTS, MAX_ITERS))
learning_rate = np.zeros((N_AGENTS, MAX_ITERS, T))
InnovationNorm = np.zeros((N_AGENTS, MAX_ITERS))

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

        label_pnt = Train_label[Agent][Image]  # supervised output

        # --> FORWARD PASS
        xx = forward_pass(uu[Agent][0], data_pnt)  # forward simulation
        prediction = xx[-1]

        # --> BACKWARD PASS
        JJ_i_k, llambdaT = Cost_Function(prediction, label_pnt)
        Delta_u = backward_pass(xx, uu[Agent][0], llambdaT)  # the gradient of the loss function

        # update the cost vector and the gradient vector
        JJ[Agent, 0] += JJ_i_k[0]
        JJ_Tot[0] += JJ_i_k[0]
        dJ_norm[Agent, 0] += np.abs(llambdaT)

        # Averaging the Local Gradient of the weight matrix
        for layer in range(len(d) - 1):
            Du[Agent][0][layer] += Delta_u[layer] / len(Train_data[Agent])
            YY[Agent][0][layer] += Delta_u[layer] / len(Train_data[Agent])
            if layer == 4:
                InnovationNorm[Agent, 0] = np.abs((np.sum(Du[Agent][0][layer])))

    # Learning rate update
    if ADAM_Yes:
        for layer in range(len(d) - 1):
            lr, gg_tp, ss_tp = ADAM(gg[Agent, 0], ss[Agent, 0], Du[Agent][0][layer], stepsize)
            learning_rate[Agent, 0, layer] = np.sum(lr) / (len(lr.flatten().tolist()))
            gg[Agent, 1] = np.sum(gg_tp) / (len(gg_tp.flatten().tolist()))
            ss[Agent, 1] = np.sum(ss_tp) / (len(ss_tp.flatten().tolist()))
    else:
        for layer in range(len(d) - 1):
            learning_rate[Agent, 0, layer] = stepsize ** 0

    for layer in range(len(d) - 1):
        uu[Agent][1][layer] = WW[Agent, Agent] * uu[Agent][0][layer] - learning_rate[Agent, 0, layer] * Du[Agent][0][
            layer]
        # compute the Average Consensus of the Network Weights
        for neigh in Neighbours:
            uu[Agent][1][layer] += WW[Agent, neigh] * uu[neigh][0][layer]

# __________________________________CYCLE FOR EACH EPOCH____________________________________
'Cycle for each Epoch'
for kk in range(1, MAX_ITERS - 1):
    'Cycle for each Agent - Computation of local model'
    for Agent in range(N_AGENTS):
        # Counter of correctly classified samples
        'Cycle for each sample of the dataset per each agent'
        for Image in range(len(Train_data[Agent])):
            # Extract the sample Imgage from the training set
            data_pnt = Train_data[Agent][Image].reshape(1, -1)  # input sample

            label_pnt = Train_label[Agent][Image]  # supervised output

            # --> FORWARD PASS
            xx = forward_pass(uu[Agent][kk], data_pnt)  # forward simulation
            prediction = xx[-1]

            # --> BACKWARD PASS
            JJ_i_k, llambdaT = Cost_Function(prediction, label_pnt)
            Delta_u = backward_pass(xx, uu[Agent][kk], llambdaT)  # the gradient of the loss function

            # update the cost vector and the gradient vector
            JJ[Agent, kk] += JJ_i_k[0]
            JJ_Tot[kk] += JJ_i_k[0]
            dJ_norm[Agent, kk] += np.abs(llambdaT)

            # Averaging the Local Gradient of the weight matrix
            for layer in range(len(d) - 1):
                Du[Agent][kk][layer] += Delta_u[layer] / len(Train_data[Agent])

        # Learning rate update
        if ADAM_Yes:
            for layer in range(len(d) - 1):
                lr, gg_tp, ss_tp = ADAM(gg[Agent, kk], ss[Agent, kk], Du[Agent][kk][layer], stepsize)
                learning_rate[Agent, kk, layer] = np.sum(lr) / (len(lr.flatten().tolist()))
                gg[Agent, kk + 1] = np.sum(gg_tp) / (len(gg_tp.flatten().tolist()))
                ss[Agent, kk + 1] = np.sum(ss_tp) / (len(ss_tp.flatten().tolist()))
        else:
            for layer in range(len(d) - 1):
                if kk % 3:
                    learning_rate[Agent, kk, layer] = stepsize ** kk
                else:
                    learning_rate[Agent, kk, layer] = learning_rate[Agent, kk - 1, layer]

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
            for layer in range(len(d) - 1):

                if layer == 4:
                    InnovationNorm[Agent, kk] = np.abs(
                        (np.sum(Du[Agent][kk][layer]) - np.sum(Du[Agent][kk - 1][layer])) / Du[Agent][kk][layer].shape[
                            1])

                YY[Agent][kk][layer] = WW[Agent, Agent] * YY[Agent][kk - 1][layer] + (
                        Du[Agent][kk][layer] - Du[Agent][kk - 1][layer])
                # compute the Average Consensus of the descent
                for neigh in Neighbours:
                    YY[Agent][kk][layer] += WW[Agent, neigh] * YY[neigh][kk - 1][layer]

            # Update the network weigths matrix with the descent
            for layer in range(len(d) - 1):
                uu[Agent][kk + 1][layer] = WW[Agent, Agent] * uu[Agent][kk][layer] - learning_rate[Agent, kk, layer] * \
                                           YY[Agent][kk][layer]
                # compute the Average Consensus of the Network Weights
                for neigh in Neighbours:
                    uu[Agent][kk + 1][layer] += WW[Agent, neigh] * uu[neigh][kk][layer]

# Terminal iteration
'Cycle for each Agent - Computation of local model'
for Agent in range(N_AGENTS):
    # Counter of correctly classified samples
    'Cycle for each sample of the dataset per each agent'
    for Image in range(len(Train_data[Agent])):
        # Extract the sample Imgage from the training set
        data_pnt = Train_data[Agent][Image].reshape(1, -1)  # input sample

        label_pnt = Train_label[Agent][Image]  # supervised output

        # --> FORWARD PASS
        xx = forward_pass(uu[Agent][MAX_ITERS - 1], data_pnt)  # forward simulation
        prediction = xx[-1]

        # --> BACKWARD PASS
        JJ_i_final, llambdaT = Cost_Function(prediction, label_pnt)

        # update the cost vector and the gradient vector
        JJ[Agent, MAX_ITERS - 1] += JJ_i_final
        JJ_Tot[MAX_ITERS-1] += JJ_i_final[0]
        dJ_norm[Agent, MAX_ITERS - 1] += np.abs(llambdaT)

        # Averaging the Local Gradient of the weight matrix
        for layer in range(len(d) - 1):
            Du[Agent][MAX_ITERS - 1][layer] += Delta_u[layer] / len(Train_data[Agent])
            if layer == 4:
                InnovationNorm[Agent, MAX_ITERS - 1] = np.abs(
                    (np.sum(Du[Agent][MAX_ITERS - 1][layer]) - np.sum(Du[Agent][kk - 1][layer])) /
                    Du[Agent][kk][layer].shape[1])

# Print the test accuracy
print('\n############################\tTEST SET ACCURACY\t############################\n')
print(end='      ')
for Agent in range(N_AGENTS):
    print(f'Agent{Agent}', end='  ')
print('', end='\n')
# Compute the Test Accuracy
Accuracy, predictions = Prediction_Performance(uu, Test_data, Test_label)
print(end='      ')
for Agent in range(N_AGENTS):
    print(f'{np.round(Accuracy[Agent] * 100, 2)}%', end='  ')
print('', end='\n')

###############################################################################
lpf = "C:/Users/barag/Documents/GitHub/Distributed-Autonomous-System-Projects"
# Figure 1 : Cost Error Evolution
CostErrorEvolution_YES = True
if CostErrorEvolution_YES:
    plt.figure()
    for Agent in range(N_AGENTS):
        plt.semilogy(np.arange(MAX_ITERS), JJ[Agent, :].flatten(), '--', linewidth=1)
    plt.semilogy(range(MAX_ITERS), np.sum(JJ, axis=0) / N_AGENTS, 'k-', label=f'Global', linewidth=3)
    plt.xlabel(r"Epochs")
    plt.title(r"Prediction Error $= \sum((y_{pred} - y_{true})^{T}(y_{pred} - y_{true}))$")
    plt.grid()
    plt.savefig(lpf + f"/plot/task1/Cost_Error_{N_AGENTS}_{MAX_ITERS}_{GT_YES}.jpg", transparent=True)

plt.show()

###############################################################################
# Figure 2 : Gradient Innovation Norm
GInnovationNorm_YES = True
if GInnovationNorm_YES:
    plt.figure()
    for Agent in range(N_AGENTS):
        plt.semilogy(range(MAX_ITERS - 2), InnovationNorm[Agent, :MAX_ITERS - 2], '--', label=f'Agent_{Agent}',
                     linewidth=1)
    plt.semilogy(range(MAX_ITERS - 2), np.sum(InnovationNorm[:, :MAX_ITERS - 2], axis=0) / N_AGENTS, 'k-',
                 label=f'Global',
                 linewidth=3, )
    plt.xlabel(r"# Iterations $t$")
    plt.title(r"Gradient Norm ")
    plt.grid()
    plt.savefig(lpf + f"/plot/task1/GInnovationNormPlot_{N_AGENTS}_{MAX_ITERS}_{GT_YES}.jpg", transparent=True)
plt.show()
###############################################################################
# Figure 3 : Gradient Innovation Norm
GInnovationNorm_YES = True
if GInnovationNorm_YES:
    plt.figure()
    for Agent in range(N_AGENTS):
        plt.plot(range(MAX_ITERS - 2), InnovationNorm[Agent, :MAX_ITERS - 2], '--', label=f'Agent_{Agent}', linewidth=1)
    plt.plot(range(MAX_ITERS - 2), np.sum(InnovationNorm[:, :MAX_ITERS - 2], axis=0) / N_AGENTS, 'k-', label=f'Global',
             linewidth=3, )
    plt.xlabel(r"# Iterations $t$")
    plt.title(r"Gradient Norm ")
    plt.grid()
    plt.savefig(lpf + f"/plot/task1/GInnovationNormPlot_{N_AGENTS}_{MAX_ITERS}_{GT_YES}.jpg", transparent=True)
plt.show()
###############################################################################
# Figure 4 : Plot of the global neural network
CostErrorEvolution_YES = True
if CostErrorEvolution_YES:
    plt.figure()
    plt.semilogy(range(MAX_ITERS), JJ_Tot, 'b-', label=f'Global', linewidth=3)
    plt.xlabel(r"Epochs")
    plt.title(r"Global Prediction Error $= \sum(\sum((y_{pred} - y_{true})^{T}(y_{pred} - y_{true})))$")
    plt.grid()
    plt.savefig(lpf + f"/plot/task1/Global_Cost_Error_{N_AGENTS}_{MAX_ITERS}_{GT_YES}.jpg", transparent=True)

plt.show()
