import numpy as np
import torch
from util.zonotope import Zonotope
from util.constrained_zonotope import ConstrainedZonotope, TorchConstrainedZonotope
from scipy.optimize import linprog

"""
This code is adapted from zono-reach-net by Adam Dai, Long Kiu Chung, and Derek Knowles, which was developed for ReLU 
networks. This code is for LReL networks instead.
"""

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

def n_tuple(n, a):
    """
    Returns a list of n-tuple over the set {1, a}.
    >>> n_tuple(2, 0.1)
    [array([[1.],
           [1.]]), array([[0.1],
           [1. ]]), array([[1. ],
           [0.1]]), array([[0.1],
           [0.1]])]
    >>> n_tuple(3, 0.)
    [array([[1.],
           [1.],
           [1.]]), array([[0.],
           [1.],
           [1.]]), array([[1.],
           [0.],
           [1.]]), array([[0.],
           [0.],
           [1.]]), array([[1.],
           [1.],
           [0.]]), array([[0.],
           [1.],
           [0.]]), array([[1.],
           [0.],
           [0.]]), array([[0.],
           [0.],
           [0.]])]
    """
    results = []
    for i in range(2 ** n):
        result = np.ones((n, 1))
        dividend = i
        j = 0

        # decimal to binary converter for all permutations of the tuple
        while dividend != 0:
            remainder = dividend % 2
            dividend = dividend // 2
            if remainder == 1:
                result[j][0] = a
            j = j + 1

        results.append(result)
    return results


def LReL_con_zono_single(Z_in, negative_slope=0):
    """
    INPUT:
    Z_in: A single constrained zonotope of class ConstrainedZonotope from constrained_zonotope.py.
    negative_slope: A float that is the activation coefficient for the LReL network. Defaults as 0 (regular ReLU).
    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    ReLU-activating Z_in.
    """
    # SETUP
    # Get the zonotope parameters
    c = Z_in.c
    G = Z_in.G
    A = Z_in.A
    b = Z_in.b

    # Get dimension of things
    n = c.shape[0]
    n_gen = G.shape[1]
    if type(A) == np.ndarray:
        n_con = A.shape[0]
    else:
        n_con = 0
    n_out_max = 2 ** n

    # Create list of output zonotopes
    Z_out = []

    # CREATE THE ZONOTOPES
    cG_tuples = n_tuple(n, negative_slope)
    Ab_tuples = n_tuple(n, 0)

    for i in range(n_out_max):
        D_i = np.diag(np.transpose(cG_tuples[i])[0])
        H_i = np.diag(np.transpose(Ab_tuples[i] * (-2) + 1)[0])

        # Get new center and generator matrices
        c_i = D_i @ c
        G_i = D_i @ G
        G_i = np.concatenate((G_i, np.zeros((n, n))), axis=1)

        # Get new constraint arrays
        HG = H_i @ G
        d_i = np.absolute(HG) @ np.ones((n_gen, 1))
        Hc = H_i @ c
        d_i = 0.5 * (d_i - Hc)

        b_i = -Hc - d_i
        if type(b) == np.ndarray:
            b_i = np.concatenate((b, b_i), axis=0)

        A_i = np.concatenate((HG, np.diag(np.transpose(d_i)[0])), axis=1)
        if n_con > 0:
            A_i = np.concatenate((np.concatenate((A, np.zeros((n_con, n))), axis=1), A_i), axis=0)

        Z_i = ConstrainedZonotope(c_i, G_i, A_i, b_i)
        if not Z_i.isEmpty():
            Z_out.append(Z_i)

    return Z_out


def LReL_con_zono(Z_in, negative_slope=0):
    """
    INPUT:
    Z_in: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py.
    negative_slope: A float that is the activation coefficient for the LReL network. Defaults as 0 (regular ReLU).
    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    ReLU-activating each element in Z_in.
    """
    # Get the number of input zonotopes
    n_in = len(Z_in)

    # Iterate through input zonotopes and generate output
    Z_out = []

    for i in range(n_in):
        Z_i = Z_in[i]
        Z_out_i = LReL_con_zono_single(Z_i, negative_slope)
        for j in range(len(Z_out_i)):
            Z_out.append(Z_out_i[j])

    return Z_out


def linear_layer_con_zono(Z_in, W, b):
    """
    INPUTS:
    Z_in: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py.
    W: A single weight matrix as numpy array.
    b: A single bias vector as numpy array.
    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    multiplying and adding W and b to each element in Z_in.
    """
    # Preallocate the output
    Z_out = []

    # Get the number of input zonotopes
    n_in = len(Z_in)

    for i in range(n_in):
        # Get the current zonotope
        Z = Z_in[i]

        # Get the zonotope's parameters
        c = Z.c
        G = Z.G

        # Do the linear transformation
        c = W @ c + b
        G = W @ G

        # Update the output
        Z_out.append(ConstrainedZonotope(c, G, Z.A, Z.b))

    return Z_out


def forward_pass_LReL_conzono(Z_in, NN_weights, NN_biases, negative_slope=0):
    """
    INPUTS:
    Z_in: A single zonotope of class Zonotope from zonotope.py.
    NN_weights: A list of numpy arrays where each element is the neural network layer's weight matrix. Its length is
    the depth of the neural network.
    NN_biases: A list of numpy arrays where each element is the neural network layer's bias. Same length as NN_weights.
    negative_slope: A float that is the activation coefficient for the LReL network. Defaults as 0 (regular ReLU).
    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    passing Z_in through a LReL neural network defined by NN_weights and NN_biases
    """
    # Get depth of neural network
    n_depth = len(NN_weights)

    # Convert input zonotope into a constrained zonotope
    Z_in = ConstrainedZonotope(Z_in.c, Z_in.G)

    # Run through layers and perform ReLU activations
    Z_out = [Z_in]
    for i in range(n_depth - 1):
        W = NN_weights[i]
        b = NN_biases[i]
        Z_out = linear_layer_con_zono(Z_out, W, b)
        Z_out = LReL_con_zono(Z_out, negative_slope)

    # Evaluate final layer
    Z_out = linear_layer_con_zono(Z_out, NN_weights[-1], NN_biases[-1])

    return Z_out

#### ------------------ PYTORCH VERSION ------------------ ####


def n_tuple_torch(n, a):
    """
    Returns a list of n-tuple over the set {1, a}.
    """
    results = []
    for i in range(2 ** n):
        result = torch.ones(n, 1).to(device)
        dividend = i
        j = 0

        # decimal to binary converter for all permutations of the tuple
        while dividend != 0:
            remainder = dividend % 2
            dividend = dividend // 2
            if remainder == 1:
                result[j][0] = a
            j = j + 1

        results.append(result)
    return results


def LReL_con_zono_single_torch(Z_in, negative_slope=0):
    """
    INPUT:
    Z_in: A single constrained zonotope of class TorchConstrainedZonotope from constrained_zonotope.py.
    negative_slope: A float that is the activation coefficient for the LReL network. Defaults as 0 (regular ReLU).
    OUTPUT:
    Z_out: A list of constrained zonotopes of class TorchConstrainedZonotope from constrained_zonotope.py as a result of
    ReLU-activating Z_in.
    """
    # SETUP
    # Get the zonotope parameters
    c = Z_in.c
    G = Z_in.G
    A = Z_in.A
    b = Z_in.b

    # Get dimension of things
    n = c.shape[0]
    n_gen = G.shape[1]
    if type(A) == torch.Tensor:
        n_con = A.shape[0]
    else:
        n_con = 0
    n_out_max = 2 ** n

    # Create list of output zonotopes
    Z_out = []

    # CREATE THE ZONOTOPES
    cG_tuples = n_tuple_torch(n, negative_slope)
    Ab_tuples = n_tuple_torch(n, 0)

    for i in range(n_out_max):
        D_i = torch.diag(cG_tuples[i].T[0]).to(device)
        H_i = torch.diag((Ab_tuples[i] * (-2) + 1).T[0]).to(device)

        # Get new center and generator matrices
        c_i = D_i @ c
        G_i = D_i @ G
        G_i = torch.cat((G_i, torch.zeros(n, n).to(device)), dim=1)

        # Get new constraint arrays
        HG = H_i @ G
        d_i = torch.abs(HG) @ torch.ones(n_gen, 1).to(device)
        Hc = H_i @ c
        d_i = 0.5 * (d_i - Hc)

        b_i = -Hc - d_i
        if type(b) == torch.Tensor:
            b_i = torch.cat((b, b_i), dim=0)

        A_i = torch.cat((HG, torch.diag(d_i.T[0])), dim=1)
        if n_con > 0:
            A_i = torch.cat((torch.cat((A, torch.zeros(n_con, n).to(device)), dim=1), A_i), dim=0)

        # # Emptiness check
        # f_cost, A_ineq, b_ineq, A_eq, b_eq = make_con_zono_empty_check_LP(A_i.cpu().detach().numpy(), b_i.cpu().detach().numpy())
        # test_value = emptiness_check(f_cost, A_ineq, b_ineq, A_eq, b_eq)
        # if test_value <= 1:
        Z_out.append(TorchConstrainedZonotope(c_i, G_i, A_i, b_i))

    return Z_out


def LReL_con_zono_torch(Z_in, negative_slope=0):
    """
    INPUT:
    Z_in: A list of constrained zonotopes of class TorchConstrainedZonotope from constrained_zonotope.py.
    negative_slope: A float that is the activation coefficient for the LReL network. Defaults as 0 (regular ReLU).
    OUTPUT:
    Z_out: A list of constrained zonotopes of class TorchConstrainedZonotope from constrained_zonotope.py as a result of
    ReLU-activating each element in Z_in.
    """
    # Get the number of input zonotopes
    n_in = len(Z_in)

    # Iterate through input zonotopes and generate output
    Z_out = []

    for i in range(n_in):
        Z_i = Z_in[i]
        Z_out_i = LReL_con_zono_single_torch(Z_i, negative_slope)
        for j in range(len(Z_out_i)):
            Z_out.append(Z_out_i[j])

    return Z_out


def linear_layer_con_zono_torch(Z_in, W, b):
    """
    INPUTS:
    Z_in: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py.
    W: A single weight matrix as numpy array.
    b: A single bias vector as numpy array.
    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    multiplying and adding W and b to each element in Z_in.
    """
    # Preallocate the output
    Z_out = []

    # Get the number of input zonotopes
    n_in = len(Z_in)

    for i in range(n_in):
        # Get the current zonotope
        Z = Z_in[i]

        # Apply the linear transformation
        c = W.clone() @ Z.c + b
        G = W.clone() @ Z.G

        # Update the output
        Z_out.append(TorchConstrainedZonotope(c, G, Z.A, Z.b))

    return Z_out


def forward_pass_NN_torch(Z_in, net, negative_slope=0):
    """
    INPUTS:
    Z_in: A single zonotope of class Zonotope from zonotope.py.
    net: torch network
    negative_slope: A float that is the activation coefficient for the LReL network. Defaults as 0 (regular ReLU).
    OUTPUT:
    Z_out: A list of constrained zonotopes of class ConstrainedZonotope from constrained_zonotope.py as a result of
    passing Z_in through a neural network defined by NN_weights and NN_biases
    """
    # extract weights and biases from network
    NN_weights = []
    NN_biases = []

    idx = 0
    for param in net.parameters():
        if idx % 2 == 0: # "even" parameters are weights
            NN_weights.append(param)
        else: # "odd" parameters are biases
            NN_biases.append(param[:,None])
        idx += 1

    # Get depth of neural network
    n_depth = len(NN_weights)

    # Convert input zonotope into a constrained zonotope
    Z_in = TorchConstrainedZonotope(Z_in.c, Z_in.G)

    # Run through layers and perform ReLU activations
    Z_out = [Z_in]
    for i in range(n_depth - 1):
        W = NN_weights[i]
        b = NN_biases[i]
        Z_out = linear_layer_con_zono_torch(Z_out, W, b)
        Z_out = LReL_con_zono_torch(Z_out, negative_slope)

    # Evaluate final layer
    Z_out = linear_layer_con_zono_torch(Z_out, NN_weights[-1], NN_biases[-1])

    return Z_out