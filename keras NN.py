import matplotlib.pyplot as plt
from keras.datasets import mnist
from keras.models import Sequential, load_model
from keras.layers.core import Dense, Dropout, Activation
from keras.utils import np_utils
import numpy as np
import matplotlib.pyplot as plt
np.random.seed(0)

''' IMPORT AND PR-PROCESSING OF DATASET'''
(X_train, y_train), (X_test, y_test) = mnist.load_data()

# shape of the numpy arrays
print(X_train.shape, y_train.shape, X_test.shape, y_test.shape)

# let's print the shape before we reshape and normalize
print("X_train shape", X_train.shape)
print("y_train shape", y_train.shape)
print("X_test shape", X_test.shape)
print("y_test shape", y_test.shape)

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
print("Train matrix shape", X_train.shape)
print("Test matrix shape", X_test.shape)

# one-hot encoding using keras' numpy-related utilities
n_classes = 10
print("Shape before one-hot encoding: ", y_train.shape)
Y_train_class = np_utils.to_categorical(y_train, n_classes)
Y_train_class[Y_train_class == 0] = -1      # substitute 0 with -1
Y_test_class = np_utils.to_categorical(y_test, n_classes)
Y_test_class[Y_test_class == 0] = -1         # substitute 0 with -1
print("Shape after one-hot encoding: ", Y_train_class.shape)

# the images and labels now are in the correct format

'''SET UP THE NEURAL NETWORK'''
###############################################################################

T = 5  # Layers
d = [784, 392, 196, 98, 10]  # Number of neurons in each layer. bias already considered

# Gradient-Tracking Method Parameters
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
def inference_dynamics(xt, ut, t):
    """
      input:
                xt current signal
                ut current weight matrix
      output:
                xtp next signal
    """
    print(f"u : {ut[1:, :].shape}, transp {ut[1:, :].T.shape}, {xt.reshape(-1, 1).shape}")
    xtp = np.zeros((1,d[t+1]))
    temp = xt.reshape(1, -1) @ ut[1:, :] + ut[0, :]  # save temporarily the product between signal and weights
    print(f"temp {temp.shape}")

    xtp = sigmoid_fn(temp)  # x' * u_ell

    # for ell in range(len(xtp)):
    #   temp = xt @ ut[ell, 1:] + ut[ell, 0]  # including the bias

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
    for index in range(len(d)):     # create the signal structure
        xx.append(np.zeros((d[index], 1)))
    xx[0] = x0

    for t in range(T-1):
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
    rows = ut.shape[0]-1    # save the dimension of input layer (without bias)
    cols = ut.shape[1]      # save the dimension of output layer
    AA = np.zeros(rows, cols)
    BB = np.zeros(rows*cols, cols)
    # df_du = np.zeros((d,(d+1)*d))
    Delta_ut = np.zeros((d, d + 1))
    d_sigma = np.zeros(d[t+1])

    temp = np.matmul(ut[1:, :].T, xt) + ut[0, :].T
    for ell in range(d_sigma.shape[0]):
        d_sigma[ell] = sigmoid_fn_derivative(temp[ell])
    # dsigma_j = sigmoid_fn_derivative(xt @ ut[j, 1:] + ut[j, 0])
    AA = (ut.T * d_sigma).T      # sarà giusto???

    for col in range(cols):
        BB[(col*rows):((col+1)*rows),col] = xt * d_sigma[col]
    #df_dx[:, j] = ut[j, 1:] * d_sigma_j
    # df_du[j, XX] = dsigma_j*np.hstack([1,xt])
    '''
    # B'@ltp
    Delta_ut[j, 0] = ltp[j] * dsigma_j
    Delta_ut[j, 1:] = xt * ltp[j] * dsigma_j

    lt = df_dx @ ltp  # '@ltp
    # Delta_ut = df_du@ltp
    '''
    lt = AA * ltp

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
        llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t + 1], xx[t], uu[t], t)

    return Delta_u

###############################################################
# GO!
J = np.zeros(max_iters)  # Cost function

# Initial Weights / Initial Input Trajectory
uu = []

for index in range(len(d)-1):
    uu.append(np.random.randn(d[index]+1, d[index+1]))  # bias considered

for k in range(max_iters):
    if k % 2 == 0:
        # print('Cost at k={:d} is {:.4f}'.format(k, J[k - 1])) # da rimuovere
        print(f"Cost at {k} is {np.round(J[k-1], decimals=4)}")

    for sample in range(0, 12):
        data_point = X_train[sample].reshape(1, -1)  # x0
        label_point = Y_train_class[sample].reshape(1,-1)

        # Initial State Trajectory
        xx = forward_pass(uu, data_point)  # forward simulation
        a = xx[4].reshape(-1, 1)
        print(a.shape)
        # GO!
        # Backward propagation
        llambdaT = 2 * (xx[-1, :] - label_point)  # nabla J in last layer
        Delta_u = backward_pass(xx, uu, llambdaT)  # the gradient of the loss function

        # Update the weights
        uu = uu - stepsize * Delta_u  # overwriting the old value

        # Forward propagation
        xx = forward_pass(uu, data_point)

        # Store the Loss Value across Iterations
        J[k] = (xx[-1, :] - label_point) @ (xx[-1, :] - label_point)  # it is the cost at k+1
        # np.linalg.norm( xx[-1,:] - label_point )**2
