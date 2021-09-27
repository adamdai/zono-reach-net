import numpy as np
import torch
from util.constrained_zonotope import ConstrainedZonotope, TorchConstrainedZonotope
from util.LReL_NN_conzono import linear_layer_con_zono_torch, n_tuple_torch
from scipy.optimize import linprog

device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")


def emptiness_check(f_cost, A_ineq, b_ineq, A_eq, b_eq):
    res = linprog(f_cost, A_ineq, b_ineq, A_eq, b_eq, (None, None))
    return res.x[-1]


def make_con_zono_empty_check_LP(A, b):
    """
    Given the constraint matrices A and b for a constrained zonotope, return the data matrices required to construct a
    linear program to solve for emptiness check of the constrained zonotope.
    """
    # Dimension of problem
    d = A.shape[1]

    # Cost
    f_cost = np.zeros((d, 1))
    f_cost = np.concatenate((f_cost, np.eye(1)), axis=0)

    # Inequality cons
    A_ineq = np.concatenate((-np.eye(d), -np.ones((d, 1))), axis=1)
    A_ineq = np.concatenate((A_ineq, np.concatenate((np.eye(d), -np.ones((d, 1))), axis=1)), axis=0)
    b_ineq = np.zeros((2 * d, 1))

    # Equality cons
    A_eq = np.concatenate((A, np.zeros((A.shape[0], 1))), axis=1)
    b_eq = b

    return f_cost, A_ineq, b_ineq, A_eq, b_eq


def sel_LReL_con_zono_single_torch(Z_in, sel_dim, negative_slope=0):
    """
    INPUT:
    Z_in: A single constrained zonotope of class TorchConstrainedZonotope from constrained_zonotope.py.
    sel_dim: An int. The first sel_dim dimensions will not be LReL-activated.
    negative_slope: A float that is the activation coefficient for the LReL network. Defaults as 0 (regular ReLU).
    OUTPUT:
    Z_out: A list of constrained zonotopes of class TorchConstrainedZonotope from constrained_zonotope.py as a result of
    ReLU-activating Z_in with the exception of the first sel_dim dimensions.
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

    # Do not activate the first sel_dim dimensions
    for i in range(len(cG_tuples)):
        cG_tuples[i] = torch.cat((torch.ones(sel_dim, 1).to(device), cG_tuples[i][sel_dim:]), 0)

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

        # Emptiness check
        f_cost, A_ineq, b_ineq, A_eq, b_eq = make_con_zono_empty_check_LP(A_i.cpu().detach().numpy(), b_i.cpu().detach().numpy())
        test_value = emptiness_check(f_cost, A_ineq, b_ineq, A_eq, b_eq)
        if test_value <= 1:
            Z_out.append(TorchConstrainedZonotope(c_i, G_i, A_i, b_i))

    return Z_out


# def sel_LReL_con_zono_single_torch(Z_in, sel_dim, negative_slope=0):
#     """
#     INPUT:
#     Z_in: A single constrained zonotope of class TorchConstrainedZonotope from constrained_zonotope.py.
#     sel_dim: An int. The first sel_dim dimensions will not be LReL-activated.
#     negative_slope: A float that is the activation coefficient for the LReL network. Defaults as 0 (regular ReLU).
#     OUTPUT:
#     Z_out: A list of constrained zonotopes of class TorchConstrainedZonotope from constrained_zonotope.py as a result of
#     ReLU-activating Z_in with the exception of the first sel_dim dimensions.
#     """
#     # SETUP
#     # Get the zonotope parameters
#     c_in = Z_in.c
#     n = c_in.shape[0]
#
#     # Create list of output zonotopes
#     Z_out = [Z_in]
#
#     for i in range(n - sel_dim):
#         Z_int = []
#
#         for j in range(len(Z_out)):
#             Z_j = Z_out[j]
#             c = Z_j.c
#             G = Z_j.G
#             A = Z_j.A
#             b = Z_j.b
#             n_gen = G.shape[1]
#             n_con = A.shape[0]
#
#             c_pos = c
#             G_pos = torch.cat((G, torch.zeros(n, n).to(device)), 1)
#             H_pos = torch.zeros(n, n).to(device)
#             H_pos[sel_dim + i, sel_dim + i] = -1.
#             d_pos = 0.5 * (torch.abs(G) @ torch.ones(n_gen, 1).to(device) - H_pos @ c)
#             b_pos = -H_pos @ c - d_pos
#             if type(b) == torch.Tensor:
#                 b_pos = torch.cat((b, b_pos), 0)
#             A_pos = torch.cat((H_pos @ G, torch.diag(d_pos.T[0])), 1)
#             if n_con > 0:
#                 A_pos = torch.cat((torch.cat((A, torch.zeros(n_con, n).to(device)), 1), A_pos), 0)
#
#             f_cost, A_ineq, b_ineq, A_eq, b_eq = make_con_zono_empty_check_LP(A_pos.cpu().detach().numpy(), b_pos.cpu().detach().numpy())
#             test_value = emptiness_check(f_cost, A_ineq, b_ineq, A_eq, b_eq)
#             if test_value <= 1.:
#                 Z_int.append(TorchConstrainedZonotope(c_pos, G_pos, A_pos, b_pos))
#
#             u_neg = torch.eye(n).to(device)
#             u_neg[sel_dim + i, sel_dim + i] = negative_slope
#             c_neg = u_neg @ c
#             G_neg = torch.cat((u_neg @ G, torch.zeros(n, n).to(device)), 1)
#             H_neg = torch.zeros(n, n).to(device)
#             H_neg[sel_dim + i, sel_dim + i] = 1.
#             d_neg = 0.5 * (torch.abs(G) @ torch.ones(n_gen, 1).to(device) - H_neg @ c)
#             b_neg = -H_neg @ c - d_neg
#             if type(b) == torch.Tensor:
#                 b_neg = torch.cat((b, b_neg), 0)
#             A_neg = torch.cat((H_neg @ G, torch.diag(d_neg.T[0])), 1)
#             if n_con > 0:
#                 A_neg = torch.cat((torch.cat((A, torch.zeros(n_con, n).to(device)), 1), A_neg), 0)
#
#             f_cost, A_ineq, b_ineq, A_eq, b_eq = make_con_zono_empty_check_LP(A_neg.cpu().detach().numpy(), b_neg.cpu().detach().numpy())
#             test_value = emptiness_check(f_cost, A_ineq, b_ineq, A_eq, b_eq)
#             if test_value <= 1.:
#                 Z_int.append(TorchConstrainedZonotope(c_neg, G_neg, A_neg, b_neg))
#
#         Z_out = Z_int.copy()
#
#     return Z_out


def sel_LReL_con_zono_torch(Z_in, sel_dim, negative_slope=0):
    """
    INPUT:
    Z_in: A list of constrained zonotopes of class TorchConstrainedZonotope from constrained_zonotope.py.
    sel_dim: An int. The first sel_dim dimensions will not be LReL-activated.
    negative_slope: A float that is the activation coefficient for the LReL network. Defaults as 0 (regular ReLU).
    OUTPUT:
    Z_out: A list of constrained zonotopes of class TorchConstrainedZonotope from constrained_zonotope.py as a result of
    ReLU-activating Z_in with the exception of the first sel_dim dimensions.
    """
    # Get the number of input zonotopes
    n_in = len(Z_in)

    # Iterate through input zonotopes and generate output
    Z_out = []

    for i in range(n_in):
        Z_i = Z_in[i]
        Z_out_i = sel_LReL_con_zono_single_torch(Z_i, sel_dim, negative_slope)
        for j in range(len(Z_out_i)):
            Z_out.append(Z_out_i[j])

    return Z_out


def expand_NN_biases_torch(NN_biases_in, dim_in):
    """
    INPUTS:
    NN_biases_in: A list of torch vectors that are the biases of the original NN
    dim_in: An int, dimension of the input state-space
    OUTPUT:
    NN_biases_out: A list of torch matrices that are the "expanded" biases of the NN. See comments for
    expand_NN_weights_torch() for definition of an "expanded" NN.
    """

    # preallocate the output
    NN_biases_out = []

    # biases of the "expanded" NN is just the orignal bias with zeros of length dim_in stacked on top
    for i in range(len(NN_biases_in)):
        NN_biases_out.append(torch.cat((torch.zeros(dim_in, 1), NN_biases_in[i]), 0))

    return NN_biases_out


def expand_NN_weights_torch(NN_weights_in, dim_in):
    """
    INPUTS:
    NN_weights_in: A list of torch matrices that are the weights of the original NN
    dim_in: An int, dimension of the input state-space
    OUTPUT:
    NN_weights_out: A list of torch matrices that are the "expanded" weights of the NN. The output of an "expanded" NN
    is the input stacked with the output of the orignal NN. For example, if the original NN has an input of [x_1; x_2]
    and output of [u], the output of the "expanded" NN would be [x_1; x_2; u]
    """

    # The first weight has to be treated differently to increase the dimension of subsequent affine transformation
    cur_weight = NN_weights_in[0]
    col_dim = cur_weight.shape[1]

    dim_in_eye = torch.eye(dim_in).to(device)
    upper_weight = dim_in_eye
    if col_dim > dim_in:
        upper_weight = torch.cat((upper_weight, torch.zeros(dim_in, col_dim - dim_in).to(device)), 1)

    NN_weights_out = [torch.cat((upper_weight, cur_weight), 0)]

    # Expand the rest of the weights
    for i in range(len(NN_weights_in) - 1):
        cur_weight = NN_weights_in[i + 1]
        col_dim = cur_weight.shape[1]
        row_dim = cur_weight.shape[0]
        upper_weight = torch.cat((dim_in_eye, torch.zeros(dim_in, col_dim).to(device)), 1)
        lower_weight = torch.cat((torch.zeros(row_dim, dim_in).to(device), cur_weight), 1)
        NN_weights_out.append(torch.cat((upper_weight, lower_weight), 0))

    return NN_weights_out


def forward_pass_NN_controller_dynamics_torch(Z_in, c_net, d_net, x_star, u_star, delta_t, c_a=0, d_a=0):

    # CONTROLLER NETWORK
    # extract weights and biases from controller network
    cNN_weights = []
    cNN_biases = []

    idx = 0
    for param in c_net.parameters():
        if idx % 2 == 0:  # "even" parameters are weights
            cNN_weights.append(param)
        else:  # "odd" parameters are biases
            cNN_biases.append(param[:, None])
        idx += 1

    dim_in = cNN_weights[0].shape[1]
    n_depth = len(cNN_weights)

    # expand the controller weights and biases, so we can keep track of the controls corresponding to the state
    exp_cNN_weights = expand_NN_weights_torch(cNN_weights, dim_in)
    exp_cNN_biases = expand_NN_biases_torch(cNN_biases, dim_in)

    # Convert input zonotope into a constrained zonotope
    Z_in = TorchConstrainedZonotope(Z_in.c, Z_in.G)

    # Run through layers and perform ReLU activations
    cZ_out = [Z_in]
    for i in range(n_depth - 1):
        W = exp_cNN_weights[i]
        b = exp_cNN_biases[i]
        cZ_out = linear_layer_con_zono_torch(cZ_out, W, b)
        cZ_out = sel_LReL_con_zono_torch(cZ_out, dim_in, c_a)  # "selective" LReL activation

    # Evaluate final layer
    cZ_out = linear_layer_con_zono_torch(cZ_out, exp_cNN_weights[-1], exp_cNN_biases[-1])
    cstar_out = c_net(x_star)
    exp_cstar_out = torch.cat((torch.zeros(dim_in), cstar_out))
    exp_cstar_out = torch.reshape(exp_cstar_out, (exp_cstar_out.shape[0], 1))

    exp_ustar = torch.cat((torch.zeros(dim_in), u_star))
    exp_ustar = torch.reshape(exp_ustar, (exp_ustar.shape[0], 1))

    cn_out = len(cZ_out)
    for i in range(cn_out):
        cZ_out[i].c = cZ_out[i].c - exp_cstar_out + exp_ustar

    # TODO: Implement clamp function

    # DYNAMICS NETWORK
    dstar_out = d_net(torch.cat((x_star, u_star))).detach()
    dstar_out = torch.reshape(dstar_out, (1, 1))

    # extract weights and biases from controller network
    dNN_weights = []
    dNN_biases = []

    idx = 0
    for param in d_net.parameters():
        if idx % 2 == 0:  # "even" parameters are weights
            dNN_weights.append(param)
        else:  # "odd" parameters are biases
            dNN_biases.append(param[:, None])
        idx += 1

    exp_dNN_weights = expand_NN_weights_torch(dNN_weights, dim_in)
    exp_dNN_biases = expand_NN_biases_torch(dNN_biases, dim_in)

    dZ_out = []
    for i in range(cn_out):
        dZ_in = [cZ_out[i]]
        for j in range(len(exp_dNN_weights) - 1):
            W = exp_dNN_weights[j]
            b = exp_dNN_biases[j]
            dZ_in = linear_layer_con_zono_torch(dZ_in, W, b)
            dZ_in = sel_LReL_con_zono_torch(dZ_in, dim_in, d_a)
        dZ_in = linear_layer_con_zono_torch(dZ_in, exp_dNN_weights[-1], exp_dNN_biases[-1])

        # Approximate next state
        # NOTE: This currently only works for 2-D state-space
        # TODO: Generalize this for higher dimensions
        dZ_in = linear_layer_con_zono_torch(dZ_in, torch.tensor([[1, 0.5 * delta_t, 0.5 * delta_t], [0, 0, 1]]).to(device), torch.cat((-0.5 * delta_t * dstar_out, -1 * dstar_out), 0))

        for j in range(len(dZ_in)):
            dZ_out.append(dZ_in[j])

    return dZ_out