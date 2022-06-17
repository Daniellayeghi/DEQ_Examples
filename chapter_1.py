import torch
import torch.nn as nn


'''
This is a basic implementation of the tanh fixed point solver.
We want to compute z* the solution to z = tanh(Wz* + x).
'''

import torch
import torch.nn as nn


class TanhFixedPointLayer(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        # initialize output z to be zero
        z = torch.zeros_like(x)
        self.iterations = 0

        # iterate until convergence
        while self.iterations < self.max_iter:
            z_next = torch.tanh(self.linear(z) + x)
            self.err = torch.norm(z - z_next)
            z = z_next
            self.iterations += 1
            if self.err < self.tol:
                break

        return z


class TanhNewtonLayer(nn.Module):
    def __init__(self, out_features, tol=1e-4, max_iter=50):
        super().__init__()
        self.linear = nn.Linear(out_features, out_features, bias=False)
        self.tol = tol
        self.max_iter = max_iter

    def forward(self, x):
        # initialize output z to be zero
        z = torch.tanh(x)
        self.iterations = 0

        # iterate until convergence
        while self.iterations < self.max_iter:
            z_linear = self.linear(z) + x
            g = z - torch.tanh(z_linear)
            self.err = torch.norm(g)
            if self.err < self.tol:
                break

            # newton step
            J = torch.eye(z.shape[1])[None, :, :] - (1 / torch.cosh(z_linear) ** 2)[:, :, None] * self.linear.weight[None, :, :]
            z = z - torch.solve(g[:, :, None], J)[0][:, :, 0]
            self.iterations += 1

        g = z - torch.tanh(self.linear(z) + x)
        z[torch.norm(g, dim=1) > self.tol, :] = 0
        return z


if __name__ == "__main__":

    samps, nout = 10, 10
    x = torch.randn(samps, nout)
    tanhFP = TanhFixedPointLayer(nout)
    z = tanhFP(x)
    print(f"FI Terminated after {tanhFP.iterations} iterations with error {tanhFP.err}")
    print(f"z* = {z}")

    tanhFPNWT = TanhNewtonLayer(nout)
    z = tanhFPNWT(x)
    print(f"NWT Terminated after {tanhFPNWT.iterations} iterations with error {tanhFPNWT.err}")
    print(f"z* = {z}")
