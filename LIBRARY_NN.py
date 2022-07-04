import numpy as np


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
def inference_dynamics(xt, ut):
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
    xtp = ReLu(temp) # sigmoid_fn(temp)

    return xtp


# Forward Propagation
def forward_pass(uu, x0, d):
    """
      input:
                uu input trajectory: u[0],u[1],..., u[T-1]
                x0 initial condition
                d list of layer dimensions
      output:
                xx state trajectory: x[1],x[2],..., x[T]
    """
    # number of layers
    T = len(d)

    xx = []
    for index in range(len(d)):  # create the signal structure
        xx.append(np.zeros((1, d[index])))

    # Input layer
    xx[0] = x0

    # compute the inference dynamics
    for t in range(T - 1):
        xx[t + 1] = inference_dynamics(xx[t], uu[t])  # x^+ = f(x,u)

    return xx


# Adjoint dynamics:
#   state:    lambda_t = A.T lambda_tp
# output: deltau_t = B.T lambda_tp
def adjoint_dynamics(ltp, xt, ut, t, d):
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
    # d_sigma = np.zeros((1, d[t + 1]))

    # linear composition of neurons activations with the weights + bias
    biases = ut[:, 0].reshape(1, -1)
    con_weight = ut[:, 1:].T
    temp = xt @ con_weight + biases

    # compute the gradient of the activations
    d_sigma =  ReLu_derivative(temp) #  sigmoid_fn_derivative(temp)

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
def backward_pass(xx, uu, llambdaT, d):
    """
      input:
                xx state trajectory
                uu input trajectory
                llambdaT terminal condition
                d list of layer dimensions
      output:
                llambda costate trajectory
                delta_u costate output, i.e., the loss gradient
    """
    # number of layers
    T = len(d)
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
        llambda[t], Delta_u[t] = adjoint_dynamics(llambda[t + 1], xx[t], uu[t], t, d)

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


def Cost_Function(Agent, kk, uu, d, data_test, label_test, N_IMAGES, BINARY):
    """
          input:
                    Agent : Agent ID
                    kk : iteration ID
                    uu : Neural Net Weight Matrix
                    d : list of layer dimensions
                    data_test : Data Samples
                    label_test : Label Samples
                    N_IMAGES : Number of images to be processed
                    BINARY: Type of classification
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
        xx = forward_pass(uu[Agent][kk], data_pnt, d)

        # adding the cost of the new sample
        J += (xx[-1] - y_true) @ (xx[-1] - y_true).T  # it is the cost at k+1
        dJ += 2 * (xx[-1] - y_true)

    return J, dJ
