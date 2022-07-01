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
Y_train = np_utils.to_categorical(y_train, n_classes)
Y_train[Y_train == 0] = -1      # substitute 0 with -1
Y_test = np_utils.to_categorical(y_test, n_classes)
Y_test[Y_test == 0] = -1         # substitute 0 with -1
print("Shape after one-hot encoding: ", Y_train.shape)

# the images and labels now are in the correct format

'''SET UP THE NEURAL NETWORK'''
###############################################################################

T = 3  # Layers
d = 300  # Number of neurons in each layer. Same numbers for all the layers
d_out = 10  # Number of neurons in the last layer, that generate the prediction label

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
# output: deltau_t = B.T lambda_tp
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

    lt = df_dx @ ltp  # '@ltp
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


J = np.zeros(max_iters)  # Cost

# Initial Weights / Initial Input Trajectory
uu = np.random.randn(T - 1, d, d + 1)

for i in range(0, 10):
    data_point = X_train[i]
    label_point = y_train[i]

    # Initial State Trajectory
    xx = forward_pass(uu, data_point)  # T x d

    # GO!
    for k in range(max_iters):
        if k % 10 == 0:
            print('Cost at k={:d} is {:.4f}'.format(k, J[k - 1]))

        # Backward propagation
        llambdaT = 2 * (xx[-1, :] - label_point)  # xT
        Delta_u = backward_pass(xx, uu, llambdaT)  # the gradient of the loss function

        # Update the weights
        uu = uu - stepsize * Delta_u  # overwriting the old value

        # Forward propagation
        xx = forward_pass(uu, data_point)

        # Store the Loss Value across Iterations
        J[k] = (xx[-1, :] - label_point) @ (xx[-1, :] - label_point)  # it is the cost at k+1
        # np.linalg.norm( xx[-1,:] - label_point )**2
