import jax
from jax import random
import jax.numpy as jnp


'''
In this section we implement a simple differentiation of fixed point layers w.r.t W
'''


def fwd_solver(f, z_init):
    z_prev, z = z_init, f(z_init)
    while jnp.linalg.norm(z_prev - z) > 1e-5:
        z_prev, z = z, f(z)
    return z


def newton_solver(f, z_init):
    f_root = lambda z: f(z) - z
    g = lambda z: z - jnp.linalg.solve(jax.jacobian(f_root)(z), f_root(z))
    return fwd_solver(g, z_init)


def fixed_point_layer(sovler, f, params, x):
    z_star = sovler(lambda z: f(params, x, z), x)
    return z_star


if __name__ == "__main__":
    f = lambda W, x, z: jnp.tanh(jnp.dot(W, z) + x)
    ndim = 10
    W = random.normal(random.PRNGKey(0), (ndim, ndim)) / jnp.sqrt(ndim)
    x = random.normal(random.PRNGKey(1), (ndim,))

    z_star = fixed_point_layer(newton_solver, f, W, x)
    print(f"z_star =\n{z_star}")

    # Naive differentiation w.r.t W
    g = jax.grad(lambda W: fixed_point_layer(newton_solver, f, W, x).sum())(W)
    print(f"Naive dz/dw =\n{g[0]}")

    # dz/dw using implicit function theorem
    from functools import partial

    @partial(jax.custom_vjp, nondiff_argnums=(0, 1))
    def fixed_point_layer(solver, f, params, x):
      z_star = solver(lambda z: f(params, x, z), z_init=jnp.zeros_like(x))
      return z_star

    def fixed_point_layer_fwd(solver, f, params, x):
      z_star = fixed_point_layer(solver, f, params, x)
      return z_star, (params, x, z_star)

    def fixed_point_layer_bwd(solver, f, res, z_star_bar):
      params, x, z_star = res
      _, vjp_a = jax.vjp(lambda params, x: f(params, x, z_star), params, x)
      _, vjp_z = jax.vjp(lambda z: f(params, x, z), z_star)
      return vjp_a(solver(lambda u: vjp_z(u)[0] + z_star_bar, z_init=jnp.zeros_like(z_star)))

    fixed_point_layer.defvjp(fixed_point_layer_fwd, fixed_point_layer_bwd)

    g = jax.grad(lambda W: fixed_point_layer(newton_solver, f, W, x).sum())(W)
    print(f"IF dz/dw =\n{g[0]}")

