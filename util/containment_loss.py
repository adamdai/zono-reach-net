import numpy as np
import torch
import cvxpy as cp
from cvxpylayers.torch import CvxpyLayer

from .NN_controller_dynamics_reach import forward_pass_NN_controller_dynamics_torch


def my_nullspace(At, rcond=None):

    # ut, st, vht = torch.Tensor.svd(At, some=False,compute_uv=True)
    ut, st, vht = torch.linalg.svd(At, full_matrices=True)
    vht = vht.transpose(-2, -1).conj()

    vht=vht.T
    Mt, Nt = ut.shape[0], vht.shape[1]
    if rcond is None:
        rcondt = torch.finfo(st.dtype).eps * max(Mt, Nt)
    tolt = torch.max(st) * rcondt
    numt= torch.sum(st > tolt, dtype=int)
    nullspace = vht[numt:,:].T.cpu().conj()
    # nullspace.backward(torch.ones_like(nullspace),retain_graph=True)
    return nullspace


def torch_containment_check(Z_1, Z_2):
    c_1 = Z_1.c
    G_1 = Z_1.G
    A_1 = Z_1.A
    b_1 = Z_1.b
    c_2 = Z_2.c
    G_2 = Z_2.G
    A_2 = Z_2.A
    b_2 = Z_2.b

    n_gen_1 = G_1.shape[1]
    n_gen_2 = G_2.shape[1]
    n_con_1 = A_1.shape[0]
    n_con_2 = A_2.shape[0]

    s_1 = torch.linalg.pinv(A_1) @ b_1
    s_2 = torch.linalg.pinv(A_2) @ b_2
    k_1 = torch.cat((torch.ones(n_gen_1, 1) - s_1, torch.ones(n_gen_1, 1) + s_1), 0)
    k_2 = torch.cat((torch.ones(n_gen_2, 1) - s_2, torch.ones(n_gen_2, 1) + s_2), 0)

    T_1 = my_nullspace(A_1)
    T_2 = my_nullspace(A_2)
    H_1 = torch.cat((T_1, -T_1), 0)
    H_2 = torch.cat((T_2, -T_2), 0)

    G_2_s_2 = G_2 @ s_2
    G_1_s_1 = G_1 @ s_1
    G_2_T_2 = G_2 @ T_2
    G_1_T_1 = G_1 @ T_1

    # CVX Problem
    gamma_var = cp.Variable((n_gen_2 - n_con_2, n_gen_1 - n_con_1))
    beta_var = cp.Variable((n_gen_2 - n_con_2, 1))
    lambda_var = cp.Variable((2 * n_gen_2, 2 * n_gen_1))

    c_a = cp.Parameter(c_1.shape)
    c_b = cp.Parameter(c_2.shape)
    k_a = cp.Parameter(k_1.shape)
    k_b = cp.Parameter(k_2.shape)
    H_a = cp.Parameter(H_1.shape)
    H_b = cp.Parameter(H_2.shape)

    G_b_s_b = cp.Parameter(G_2_s_2.shape)
    G_a_s_a = cp.Parameter(G_1_s_1.shape)
    G_b_T_b = cp.Parameter(G_2_T_2.shape)
    G_a_T_a = cp.Parameter(G_1_T_1.shape)

    constraints = [lambda_var @ k_a <= k_b + H_b @ beta_var,
                   lambda_var >= np.zeros((2 * n_gen_2, 2 * n_gen_1))]
    objective = cp.Minimize(cp.norm(c_b + G_b_s_b - c_a - G_a_s_a - G_b_T_b @ beta_var) + cp.norm(G_a_T_a - G_b_T_b @ gamma_var, 'fro') + cp.norm(lambda_var @ H_a - H_b @ gamma_var, 'fro'))
    problem = cp.Problem(objective, constraints)
    assert problem.is_dpp()

    cvxpylayer = CvxpyLayer(problem, parameters=[c_a, c_b, k_a, k_b, H_a, H_b, G_b_s_b, G_a_s_a, G_b_T_b, G_a_T_a], variables=[gamma_var, beta_var, lambda_var])

    # solve the problem
    gamma_opt, beta_opt, lambda_opt = cvxpylayer(c_1, c_2, k_1, k_2, H_1, H_2, G_2_s_2, G_1_s_1, G_2_T_2, G_1_T_1)

    return torch.norm(c_2 + G_2_s_2 - c_1 - G_1_s_1 - G_2_T_2 @ beta_opt) + torch.norm(G_1_T_1 - G_2_T_2 @ gamma_opt) + torch.norm(lambda_opt @ H_1 - H_2 @ gamma_opt)


def dynamic_NN_contain_step(Z_in, Z_obs, c_net, d_net, x_star, u_star, delta_t, con_opt, c_a=0, d_a=0):

    con_opt.zero_grad()

    Z_out = forward_pass_NN_controller_dynamics_torch(Z_in, c_net, d_net, x_star, u_star, delta_t, c_a, d_a)

    losses = []

    for z in Z_out:
        v = torch_containment_check(z, Z_obs)
        loss = v
        losses.append(loss)

    if losses:
        total_loss = sum(losses)
        print('loss: ', total_loss)
        total_loss.backward() # backprop
        print(c_net.fc1.weight.grad)
        con_opt.step()