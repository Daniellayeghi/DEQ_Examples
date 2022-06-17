import cvxpy as cp
import numpy as np
import torch
from scipy.linalg import solve_discrete_are
from scipy.linalg import sqrtm

from cvxpylayers.torch import CvxpyLayer


'''

'''

# Generate data
torch.manual_seed(1)
np.random.seed(1)

n = 2
m = 3

A = np.eye(n) + 1e-2 * np.random.randn(n, n)
B = 1e-2 / 3 * np.random.randn(n, m)
Q = np.eye(n)
R = np.eye(m)

# Compute LQR control policy
P_lqr = solve_discrete_are(A, B, Q, R)
P = R + B.T@P_lqr@B
P_sqrt_lqr = sqrtm(P)

# Construct CVXPY problem and layer
x_cvxpy = cp.Parameter((n, 1))
P_sqrt_cvxpy = cp.Parameter((m, m))
P_21_cvxpy = cp.Parameter((n, m))
q_cvxpy = cp.Parameter((m, 1))

u_cvxpy = cp.Variable((m, 1))
y_cvxpy = cp.Variable((n, 1))

## Setup the problem using affine variables
objective = .5 * cp.sum_squares(P_sqrt_cvxpy @ u_cvxpy) + x_cvxpy.T @ y_cvxpy + q_cvxpy.T @ u_cvxpy
problem = cp.Problem(cp.Minimize(objective), [cp.norm(u_cvxpy) <= 1, y_cvxpy == P_21_cvxpy @ u_cvxpy])
assert problem.is_dpp()
policy = CvxpyLayer(
    problem, [x_cvxpy, P_sqrt_cvxpy, P_21_cvxpy, q_cvxpy], [u_cvxpy]
)

import matplotlib.pyplot as plt

def train(iters):
    # Initialize with LQR control lyapunov function
    P_sqrt = torch.from_numpy(P_sqrt_lqr).requires_grad_(True)
    P_21 = torch.from_numpy(A.T @ P_lqr @ B).requires_grad_(True)
    q = torch.zeros((m, 1), dtype=torch.double, requires_grad=True)
    variables = [P_sqrt, P_21, q]
    A_tch, B_tch, Q_tch, R_tch = map(torch.from_numpy, [A, B, Q, R])

    def g(x, u):
        return (x.t() @ Q_tch @ x + u.t() @ R_tch @ u).squeeze()

    def evaluate(x0, P_sqrt, P_21, q, T):
        x = x0
        cost = 0.
        for _ in range(T):
            u, = policy(x, P_sqrt, P_21, q)
            cost += g(x, u) / T
            x = A_tch @ x + B_tch @ u + .2 * torch.randn(n, 1).double()
        return cost

    def eval_loss(N=8, T=25):
        return sum([evaluate(torch.zeros(n, 1).double(), P_sqrt, P_21, q, T=T)
                    for _ in range(N)]) / N

    results = []
    optimizer = torch.optim.SGD(variables, lr=.02, momentum=.9)
    for i in range(iters):
        # use same seeds each iteration to get pretty training plot
        torch.manual_seed(1)
        np.random.seed(1)
        optimizer.zero_grad()
        loss = eval_loss()
        loss.backward()
        optimizer.step()
        results.append(loss.item())
        print("(iter %d) loss: %g " % (i, results[-1]))
    return results


results = train(iters=100)

plt.figure()
plt.plot(results)
plt.xlabel('iteration')
plt.ylabel('average cost')
plt.savefig("adp.pdf")
plt.show()

