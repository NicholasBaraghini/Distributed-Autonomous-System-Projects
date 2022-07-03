from keras.datasets import mnist
import numpy as np
import matplotlib.pyplot as plt
from keras.models import Sequential
from keras.layers import Dense
from keras.layers import Input
import tensorflow as tf

# Useful constants
N = 10
data = mnist.load_data()

# Split the dataset
(X_train, y_train), (X_test, y_test) = data

# Reshape the dataset to acces every pixel
X_train = X_train.reshape((X_train.shape[0], 28 * 28)).astype('float32')
X_test = X_test.reshape((X_test.shape[0], 28 * 28)).astype('float32')

# Get value of every pixel as a number between 0 and 1
X_train = X_train / 255
X_test = X_test / 255

# 1
# Digit = 4
# Digit 4 -> Label 1
# All other digits -> Label -1

y_train = [1 if y == 4 else 0 for y in y_train]
y_test = np.array([1 if y == 4 else 0 for y in y_test], dtype='float32')

# 2
# Shuffle training set randomly
p = np.random.permutation(len(X_train))
y_train = np.array(y_train, dtype='float32')[p.astype(int)]
X_train = np.array(X_train, dtype='float32')[p.astype(int)]


# Split training set in N subsets
def splitSet(n):
    subsets_X = np.array_split(X_train, n)
    subsets_y = np.array_split(y_train, n)
    return subsets_X, subsets_y


# 3
# Distributed Gradient tracking
T = 3  # Layers
d = 784  # Number of neurons in each layer. Same numbers for all the layers

# Training Set
(images, labels) = splitSet(N)
data_arrays = images
label_arrays = labels
# Gradient Method Parameters
max_iters = 10  # epochs
stepsize = 0.1  # learning rate


###############################################################################
# Activation Function
def sigmoid_fn(xi):
    return 1 / (1 + np.exp(-xi))


# Derivative of Activation Function
def sigmoid_fn_derivative(xi):
    return sigmoid_fn(xi) * (1 - sigmoid_fn(xi))


# Inference: x_tp = f(xt,ut)
def inference_dynamics(xt, ut):
    """
      input:
                xt current state
                ut current input
      output:
                xtp next state
    """
    xtp = np.zeros(d)
    for ell in range(d):
        temp = xt @ ut[ell, 1:] + ut[ell, 0]  # including the bias

        xtp[ell] = sigmoid_fn(temp)  # x' * u_ell

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
    xx = np.zeros((T, d))
    xx[0] = x0

    for t in range(T - 1):
        xx[t + 1] = inference_dynamics(xx[t], uu[t])  # x^+ = f(x,u)
    return xx


# Adjoint dynamics:
#   state:    lambda_t = A.T lambda_tp
#   output: deltau_t = B.T lambda_tp
def adjoint_dynamics(ltp, xt, ut):
    """
      input:
                llambda_tp current costate
                xt current state
                ut current input
      output:
                llambda_t next costate
                delta_ut loss gradient wrt u_t
    """
    df_dx = np.zeros((d, d))

    # df_du = np.zeros((d,(d+1)*d))
    Delta_ut = np.zeros((d, d + 1))

    for j in range(d):
        dsigma_j = sigmoid_fn_derivative(xt @ ut[j, 1:] + ut[j, 0])

        df_dx[:, j] = ut[j, 1:] * dsigma_j
        # df_du[j, XX] = dsigma_j*np.hstack([1,xt])

        # B'@ltp
        Delta_ut[j, 0] = ltp[j] * dsigma_j
        Delta_ut[j, 1:] = xt * ltp[j] * dsigma_j

    lt = df_dx @ ltp  # A'@ltp
    # Delta_ut = df_du@ltp

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
    llambda = np.zeros((T, d))
    llambda[-1] = llambdaT

    Delta_u = np.zeros((T - 1, d, d + 1))

    for t in reversed(range(T - 1)):  # T-2,T-1,...,1,0
        llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t + 1], xx[t], uu[t])

    return Delta_u


I_N = np.identity(N, dtype=int)
p_ER = 0.3
while 1:
    neigh = np.random.binomial(1, p_ER, (N, N))
    neigh = np.logical_or(neigh, neigh.T)
    neigh = np.multiply(neigh, np.logical_not(I_N)).astype(int)

    test = np.linalg.matrix_power((I_N + neigh), N)

    if np.all(test > 0):
        print("the graph is connected\n")
        break
    else:
        print("the graph is NOT connected\n")
        quit()

W = np.zeros((N, N))

for i in range(N):
    N_i = np.nonzero(neigh[i])[0]  # In-Neighbors of node i
    deg_i = len(N_i)

    for j in N_i:
        N_j = np.nonzero(neigh[j])[0]  # In-Neighbors of node j
        # deg_jj = len(N_jj)
        deg_j = N_j.shape[0]

        W[i, j] = 1 / (1 + max([deg_i, deg_j]))
    # WW[i,jj] = 1/(1+np.max(np.stack((deg_i,deg_jj)) ))

W += I_N - np.diag(np.sum(W, axis=0))


def gradient_tracking(neigh, delta_f):
    for j in neigh:
        part0 = a[i, j] * zz[j]
        part1 = a[i, j] * delta_f[j] - delta_f[i]
    zz = part0 - stepsize * part1

    for j in neigh:
        xx = a[i, j] * xx[j] - stepsize * delta_f[i]


def binaryCrossEntropy(data, label):
    # part0 = (1 - y_true) * np.log(1 - y_pred + 1e-7)
    # part1 = y_true * np.log(y_pred + 1e-7)
    part0 = label * np.log(data)
    part1 = (1 - label) * np.log(1 - data)

    return part0 + part1


###############################################################################
# MAIN
###############################################################################

J = np.zeros(max_iters)  # Cost

# Initial Weights / Initial Input Trajectory
uu = np.random.randn(T - 1, d, d + 1)
zz = np.zeros(d)
Delta_u = np.zeros((d, d + 1))
bce = np.zeros((d, d + 1))

# Want 1 neuron with weights in the output layer
for j in range(1, d):
    uu[-1, j] = 0

# GO!
for k in range(max_iters):
    if k % 10 == 0:
        print("k: ", k)
        print('Cost at k={:d} is {:.4f}'.format(k, J[k - 1]))
        for i in range(N):
            # Initial State Trajectory
            print("i: ", i)
            xx = forward_pass(uu, images[i][0])  # T x d
            for img in range(100):
                print("image ", img)
                data_point = data_arrays[i][img]
                label_point = label_arrays[i][img]

                # Forward propagation
                xx_i = forward_pass(uu, data_point)

                # Binary cross entropy
                # bce[i, img] = binaryCrossEntropy(data_point, label_point)
                # Backward propagation
                llambdaT = 2 * (xx[-1, :] - label_point)  # xT
                Delta_u[i, img] = backward_pass(xx[i], uu[i], llambdaT)  # the gradient of the loss function

            # Update the weights
            # Want to find a common u for all agents, where u is calculated over the sum of the
            # optimized u (with gradient) for all agents i
            # uu = uu - stepsize*Delta_u # overwriting the old value
            uu[i] = W[i, i] * uu[i] + zz[i] - stepsize * np.sum(Delta_u[i], axis=0)
            zz[i] = W[i, i] * zz[i] - stepsize * np.sum(Delta_u[i], axis=0) * (W[i, i] - 1)

            for j in neigh:
                uu[i] += W[i, j] * uu[j]
                zz[i] += W[i, j] * (zz[j] - stepsize * np.sum(Delta_u[j], axis=0))
            print("weights updated")
    # The loss function for classification problems with (0,1) classes - Binary Cross Entropy
    # Store the Loss Value across Iterations
    J[k] = (xx[-1, :] - label_point) @ (xx[-1, :] - label_point)
    # J[k] = np.sum(bce[i]) # it is the cost at k+1
    # np.linalg.norm( xx[-1,:] - label_point )**2

_, ax = plt.subplots()
ax.plot(range(max_iters), J)
plt.show()