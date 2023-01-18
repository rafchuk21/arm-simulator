import numpy as np
from scipy.linalg import solve_discrete_are, expm
import time

def linearize(f, x: np.matrix, u: np.matrix, eps = 1e-4) -> tuple[np.matrix, np.matrix]:
    """Linearize a system to the form x-dot = A(x-x0) + B(u-u0), where:
        x is the state and u is the input,
        x0, u0 are form stationary point.
            This can be guaranteed with u0 = u_ff(x0)

        Can pass these A, B into LQR and get u = K(r-x) + u_ff
    
    Arguments:
        x: State to linearize at
        u: voltage input to linearize at
    """

    jac_x = jacobian_with_x(f, x, u, eps)
    jac_u = jacobian_with_u(f, x, u, eps)

    return (jac_x, jac_u)

def discretize_ab(A, B, dt):
    states = A.shape[0]
    inputs = B.shape[1]

    M = np.matrix(expm(np.block([[A, B], [np.zeros((inputs, states)), np.zeros((inputs, inputs))]]) * dt))
    return M[:states, :states], M[:states, states:]

def jacobian_with_x(f, x: np.matrix, u: np.matrix, eps = 1e-4) -> np.matrix:
    """Get the jacobian of f with respect to x evaluated at x, u"""
    Jx = np.matrix(np.zeros((x.shape[0], x.shape[0])))

    for i in range(x.shape[0]): # each column is dx-dot/dx[i]
        ux = x.copy()
        ux[i] += eps            # upper x
        lx = x.copy()
        lx[i] -= eps            # lower x
        col = (f(ux, u) - f(lx, u)) / (2*eps)
        Jx[:,i] = col

    return Jx

def jacobian_with_u(f, x: np.matrix, u:np.matrix, eps = 1e-4) -> np.matrix:
    """Get the jacobian of f with respect to u evaluated at x, u"""
    Ju = np.matrix(np.zeros((x.shape[0], u.shape[0])))

    for i in range(u.shape[0]): # each column is dx-dot/du[i]
        uu = u.copy()
        uu[i] += eps            # upper u
        lu = u.copy()
        lu[i] -= eps            # lower u
        Ju[:,i] = (f(x, uu) - f(x, lu)) / (2*eps)

    return Ju

def lqr(A: np.matrix, B: np.matrix, Q: np.matrix, R: np.matrix) -> np.matrix:
    S = solve_discrete_are(A, B, Q, R)
    return (R + B.T * S * B).I * B.T * S * A