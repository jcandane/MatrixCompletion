import numpy as np

from cvxpy import Minimize, Problem, Variable, SCS
from cvxpy import norm as cvxnorm
from cvxpy import vec as cvxvec

from scipy.sparse.linalg import svds

def solveV(U, M_Ω, IJ, shaper, solver=SCS, output_error=False):
    """
    Given factor U, solve for V using
        min FrobNorm( (U * V.T - M) )
    """

    # Parameters
    n = shaper[1]
    r = U.shape[1]
    I, J = IJ
    
    # Problem
    V_T  = Variable((r, n)) ## V_T is an object!! not a physical tensor!!
    obj  = Minimize(cvxnorm(cvxvec((U @ V_T)[I, J]) - M_Ω))
    prob = Problem(obj)
    prob.solve(solver=solver, verbose=False)
    V = V_T.value.T
    if output_error:
        return (V, prob.value)
    else:
        return V

def solveU(V, M_Ω, IJ, shaper, solver=SCS, output_error=False):
    """
    Given factor V, solve for U
    """

    # Parameters
    m = shaper[0]
    if V.shape[0] < V.shape[1]:
        # make sure V_T is "short and fat"
        V = V.T
    r = V.shape[1]
    I, J = IJ

    # Problem
    U    = Variable((m, r))
    obj  = Minimize(cvxnorm(cvxvec((U @ V.T)[I, J]) - M_Ω))
    prob = Problem(obj)
    prob.solve(solver=solver, verbose=False)

    if output_error:
        return (U.value, prob.value)
    else:
        return U.value

def altMinSense(S, r, max_iters=50, tol=1e-4, optCond=lambda x,y: np.abs(x - y)):
    """
    Alternating minimization algorithm for a matrix completion.
    INPUTs:
    S : scipy.sparse.coo_matrix -- subsampled sparse matrix
    r : int -- rank
    RETURN:
    U,V : numpy.ndarray -- matrix-decomposition factors
    """

    #optCond = kwargs.get('optCond', lambda x, y: np.abs(x - y))
    
    IJ     = [S.row, S.col]
    shaper = S.shape
    M_Ω    = 1.*S.data 

    S.data = S.data * (S.size / np.prod(shaper)) ## unbias for sparse SVD

    S_left, S_sing, S_right = svds(S, k=r)
    U = S_left[:, :r]
    objPrevious = np.inf
    for T in range(max_iters):
        V = solveV(U, M_Ω, IJ, shaper, solver=SCS)
        U, output_error = solveU(V, M_Ω, IJ, shaper, solver=SCS, output_error=True)
        
        if optCond(output_error, objPrevious) < tol:
            print('Output Error = {:5.3g}'.format(output_error))
            break
        else:
            objPrevious = output_error
    return U, V