import numpy as np
import matplotlib.pyplot as plt

np.random.seed(0)


# Quadratic Function
def quadratic_fn(x, Q, r):
    if not np.isscalar(Q) or not np.isscalar(r):
        print('Error')
        exit(0)

    fval = 0.5 * x * Q * x + r * x
    fgrad = Q * x + r
    return fval, fgrad


###############################################################################
# Useful constants
MAXITERS = np.int(1e3)  # Explicit Casting
NN = 10

###############################################################################
# Generate Network Bynomial Graph
I_NN = np.identity(NN, dtype=int)
p_ER = 0.3
while 1:
    Adj = np.random.binomial(1, p_ER, (NN, NN))
    Adj = np.logical_or(Adj, Adj.T)
    Adj = np.multiply(Adj, np.logical_not(I_NN)).astype(int)

    test = np.linalg.matrix_power((I_NN + Adj), NN)

    if np.all(test > 0):
        print("the graph is connected\n")
        break
    else:
        print("the graph is NOT connected\n")
        # quit()

###############################################################################
# Compute mixing matrices
# Metropolis-Hastings method to obtain a doubly-stochastic matrix
WW = np.zeros((NN, NN))

for ii in range(NN):
    N_ii = np.nonzero(Adj[ii])[0]  # In-Neighbors of node i
    deg_ii = len(N_ii)

    for jj in N_ii:
        N_jj = np.nonzero(Adj[jj])[0]  # In-Neighbors of node j
        # deg_jj = len(N_jj)
        deg_jj = N_jj.shape[0]

        WW[ii, jj] = 1 / (1 + max([deg_ii, deg_jj]))
    # WW[ii,jj] = 1/(1+np.max(np.stack((deg_ii,deg_jj)) ))

WW += I_NN - np.diag(np.sum(WW, axis=0))

with np.printoptions(precision=4, suppress=True):
    print('Check Stochasticity\n row:    {} \n column: {}'.format(
        np.sum(WW, axis=1),
        np.sum(WW, axis=0)
    ))

# quit(0)
###############################################################################
# Declare Cost Variables
Q = 10 * np.random.rand(NN)
# Q = Q + Q.T
R = 10 * (np.random.rand(NN) - 1)

xopt = -np.sum(R) / np.sum(Q)
fopt = 0.5 * xopt * np.sum(Q) * xopt + np.sum(R) * xopt
print('The optimal cost is: {:.4f}'.format(fopt))

# Declare Algorithmic Variables
XX = np.zeros((NN, MAXITERS))
XX_dg = np.zeros((NN, MAXITERS))  # distributed gradient
SS = np.zeros((NN, MAXITERS))

XX_init = 10 * np.random.rand(NN)
XX[:, 0] = XX_init
XX_dg[:, 0] = XX_init

for ii in range(NN):
    _, SS[ii, 0] = quadratic_fn(XX[ii, 0], Q[ii], R[ii])

FF = np.zeros(MAXITERS)
FF_dg = np.zeros(MAXITERS)

###############################################################################
# GO!
ss = 1e-2  # stepsize

for tt in range(MAXITERS - 1):
    if (tt % 50) == 0:
        print("Iteration {:3d}".format(tt), end="\n")

    for ii in range(NN):
        Nii = np.nonzero(Adj[ii])[0]

        XX[ii, tt + 1] = WW[ii, ii] * XX[ii, tt] - ss * SS[ii, tt]
        for jj in Nii:
            XX[ii, tt + 1] += WW[ii, jj] * XX[jj, tt]

        f_ii, grad_fii = quadratic_fn(XX[ii, tt], Q[ii], R[ii])
        _, grad_fii_p = quadratic_fn(XX[ii, tt + 1], Q[ii], R[ii])

        SS[ii, tt + 1] = WW[ii, ii] * SS[ii, tt] + (grad_fii_p - grad_fii)
        for jj in Nii:
            SS[ii, tt + 1] += WW[ii, jj] * SS[jj, tt]

        FF[tt] += f_ii

        # Distributed Gradient for Comparison
        f_ii_dg, grad_fii_dg = quadratic_fn(XX_dg[ii, tt], Q[ii], R[ii])
        FF_dg[tt] += f_ii_dg
        XX_dg[ii, tt + 1] = WW[ii, ii] * XX_dg[ii, tt] - ss * grad_fii_dg
        # XX_dg[ii,tt+1] = WW[ii,ii]*XX_dg[ii,tt] - ss*grad_fii_dg/(tt+1)*10 # diminishing
        for jj in Nii:
            XX_dg[ii, tt + 1] += WW[ii, jj] * XX_dg[jj, tt]

# Terminal iteration
for ii in range(NN):
    f_ii, _ = quadratic_fn(XX[ii, -1], Q[ii], R[ii])
    FF[-1] += f_ii
    f_ii_dg, _ = quadratic_fn(XX_dg[ii, -1], Q[ii], R[ii])
    FF_dg[-1] += f_ii_dg

###############################################################################
# generate N random colors
colors = {}
for ii in range(NN):
    colors[ii] = np.random.rand(3)

###############################################################################
# Figure 1 : Evolution of the local estimates
if 0:
    plt.figure()
    plt.plot(np.arange(MAXITERS), np.repeat(xopt, MAXITERS), '--', linewidth=3)
    for ii in range(NN):
        plt.plot(np.arange(MAXITERS), XX[ii], color=colors[ii])

    plt.xlabel(r"iterations $t$")
    plt.ylabel(r"$x_i^t$")
    plt.title("Evolution of the local estimates")
    plt.grid()

###############################################################################
# Figure 2 : Cost Evolution
if 0:
    plt.figure()
    plt.plot(np.arange(MAXITERS), np.repeat(fopt, MAXITERS), '--', linewidth=3)
    plt.plot(np.arange(MAXITERS), FF)
    plt.xlabel(r"iterations $t$")
    plt.ylabel(r"$x_i^t$")
    plt.title("Evolution of the cost")
    plt.grid()

###############################################################################
# Figure 3 : Cost Error Evolution
if 1:
    plt.figure()
    plt.semilogy(np.arange(MAXITERS), np.abs(FF - np.repeat(fopt, MAXITERS)), '--', linewidth=3)
    plt.semilogy(np.arange(MAXITERS), np.abs(FF_dg - np.repeat(fopt, MAXITERS)), '--', linewidth=3)
    plt.xlabel(r"iterations $t$")
    plt.ylabel(r"$|\sum_{i=1}^N f_i(x_i^t) - f^\star|$")
    plt.title("Evolution of the cost error")
    plt.grid()

plt.show()
