#### modification of asberk's https://github.com/asberk/matrix-completion-whirlwind

import numpy as np
import matplotlib.pyplot as plt

from sklearn.metrics import mean_squared_error

from cvxpy import Minimize, Problem, Variable, SCS
from cvxpy import norm as cvxnorm
from cvxpy import vec as cvxvec


def rmse(A, B):
    """
    rmse(A, B) is a function to compute the relative mean-squared
    error between two matrices A and B, given by
    1/(m*n) * sum_ij (A_ij - B_ij)^2
    where A and B are m-by-n numpy arrays.
    """
    return np.sqrt(mean_squared_error(A,B))

def plot_comparison(M, M_rec, show_error=True, **kwargs):
    """
    plot_comparison(M, M_rec, show_error=True, show_colorbars=True, **kwargs)
        plots two matrices, M and M_rec, side-by-side along with
        optionally a plot of the error between the two (given by some
        error map error_map)
    Input:
    M : the original matrix M
    M_rec : the recovered matrix M_rec
    show_error : if True (default) produce a third plot showing a 
                 mapping of (M, M_rec) 
                 (default: absolute error |M-M_rec|)
    show_colorbars : 
    kwargs:
    figsize : the figure size (w, h); default is (15,8)
    cmap : a string denoting the colormap to use for the plots,
           defined according to plt.cm.cmaps_lsited (default: 'viridis')
    fontsize : the font size for the labels on the plot (default: 16)
    plot_titles : the plot titles for the first two plots 
                  (default: ['$M$', '$M^*$'])
    error_map : the error map to use for the third plot 
                (only used if show_error=True). Default value is absolute
                error, |M - M_rec|.
    error_title : the title to use for the third plot. 
    cb : if True (default) plot colorbars beneath the plots (recommended)
    cb_orient : whether to show 'horizontal' or 'vertical' orientation 
                of colourbar (default horizontal)
    cb_pad : padding between plot and colourbar (default .05)
    """
    n_plots = 2
    if show_error:
        n_plots += 1
    figsize=kwargs.get('figsize', (15,8))
    cmap = kwargs.get('cmap', 'viridis')
    fontsize = kwargs.get('size', 16)
    plot_titles = kwargs.get('plot_titles', ['$M$', '$M^*$'])
    error_map = kwargs.get('error_map', lambda x,y: np.abs(x-y))
    error_title = kwargs.get('error_title', 'absolute error $|M - M^*|$')
    cb = kwargs.get('show_colorbars', True)
    
    fig, axes = plt.subplots(1, n_plots, figsize=figsize)
    matrices = [M, M_rec]
    if n_plots == 3:
        matrices.append(error_map(M, M_rec))
        plot_titles.append(error_title)
    for ax, mat, ptitle in zip(axes.flat, matrices, plot_titles):
        I = ax.matshow(mat, cmap=cmap)
        ax.set_title(ptitle, size=fontsize)
        ax.axis('off')
        if cb:
            cb_orient = kwargs.get('orientation', 'horizontal')
            cb_pad = kwargs.get('pad', 0.05)
            fig.colorbar(I, ax=ax, orientation=cb_orient, pad=cb_pad)
    print('RMSE = {}'.format(rmse(M, M_rec)))
    return


def matIndicesFromMask(mask):
    """
    matIndicesFromMask(mask) returns the matrix-indices 
    corresponding to mask == 1. This operation returns a 
    tuple containing a list of row indices and a list of 
    column indices.
    """
    return np.where(mask.T==1)[::-1]

def multiplyFromMatIdxList(U, V, Omega):
    """
    multiplyFromMatIdxList(U, V, Omega) returns a vector M_Omega 
    where each entry is given by 
        M_jk := < U_j, V_k >, for (j,k) \in Omega

    Input:
        U : The m-by-r left low-rank matrix
        V : The n-by-r right low-rank matrix
    Omega : A tuple of vectors, the first representing a list of 
            row indices, the second column indices. The tuple formed
            by the i-th element of each vector corresponds to an
            observed element of the low-rank matrix U @ V.T
    """
    return np.array([U[j,:] @ V[k,:] for j,k in zip(*Omega)])

def mcFrobSolveRightFactor_cvx(U, M_Omega, mask, **kwargs):
    """
    A solver for the right factor, V, in the problem 
        min FrobNorm( P_Omega(U * V.T - M) )
    where U is an m-by-r matrix, V an n-by-r matrix.
    M_Omega is the set of observed entries in matrix form, while
    mask is a Boolean array with 1/True-valued entries corresponding 
    to those indices that were observed.

    This function is computed using the CVXPY package (and 
    thus is likely to be slower than a straight iterative 
    least squares solver).
    """
    # Options
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    solver = kwargs.get('solver', SCS)
    verbose = kwargs.get('verbose', False)

    if isinstance(verbose, int):
        if verbose > 1:
            verbose = True
        else:
            verbose = False

    # Parameters
    n = mask.shape[1]
    r = U.shape[1]

    Omega_i, Omega_j = matIndicesFromMask(mask)
    
    # Problem
    V_T  = Variable((r, n))
    obj  = Minimize(cvxnorm(cvxvec((U @ V_T)[Omega_i, Omega_j]) - M_Omega))
    prob = Problem(obj)
    prob.solve(solver=solver, verbose=verbose)
    V = V_T.value.T
    if returnObjectiveValue:
        return (V, prob.value)
    else:
        return V

def mcFrobSolveLeftFactor_cvx(V, M_Omega, mask, **kwargs):
    """
    mcFrobSolveLeftFactor_cvx(V, M_Omega, mask, **kwargs)
    A solver for the left factor, U, in the problem
        min FrobNorm( P_Omega(U * V.T - M) )
    where U is an m-by-r matrix, V an n-by-r matrix.
    M_Omega is the set of observed entries in matrix form, while
    mask is a Boolean array with 1/True-valued entries corresponding 
    to those indices that were observed.

    This function is computed using the CVXPY package (and 
    thus is likely to be slower than a straight iterative 
    least squares solver).
    """
    # Options
    returnObjectiveValue = kwargs.get('returnObjectiveValue', False)
    solver = kwargs.get('solver', SCS)
    verbose = kwargs.get('verbose', False)

    if isinstance(verbose, int):
        if verbose > 1:
            verbose = True
        else:
            verbose = False

    # Parameters
    m = mask.shape[0]
    if V.shape[0] < V.shape[1]:
        # make sure V_T is "short and fat"
        V = V.T
    r = V.shape[1]

    Omega_i, Omega_j = matIndicesFromMask(mask)

    # Problem
    U    = Variable((m, r))
    obj  = Minimize(cvxnorm(cvxvec((U @ V.T)[Omega_i, Omega_j]) - M_Omega))
    prob = Problem(obj)
    prob.solve(solver=solver, verbose=verbose)
    if returnObjectiveValue:
        return (U.value, prob.value)
    else:
        return U.value
    
def matrixCompletionSetup(r, m, n=None, p=None):
    """
    matrixCompletionSetup(m, n, r, p) computes everything necessary to
                                      be set-up for a matrix
                                      completion problem using the
                                      structural assumption that M has
                                      low rank and each entry is
                                      observed independently with
                                      probability p.
    Input
    m : the number of rows of the output matrix M
    n : the number of columns of the output matrix M
    r : the rank of the output matrix M. 
        Note that this value can be a vector of values. 
    p : the observation probability for entries of M. 
        Note that this value can be a vector of values.
    k : the number of iterates to compute (default 1)

    Output
             U : the left m-by-r matrix
             V : the right n-by-r matrix
         Omega : the list of indices of M that were observed
    Omega_mask : the mask corresponding to observed entries Omega of
                 the matrix M
    """
    print('There has been a re-write of this function. Please ' +
          'check documentation or source for more information.' +
          ' (cf. sparseMatComSetup for a sparse version of this ' + 
          'function.)')
    if n is None:
        n = m
    if p is None:
        p = .5

    U = np.random.randint(0, 5, size=(m,r))
    V = np.random.randint(0, 5, size=(n,r))
    M = U @ V.T # size=(m,n)
    Omega_mask = (np.random.rand(m,n) <= p)
    Omega      = matIndicesFromMask(Omega_mask)
    M_Omega    = multiplyFromMatIdxList(U,V,Omega)
    return (U, V, M_Omega, Omega, Omega_mask)

def altMinSense(M_Omega, Omega_mask, r, **kwargs):
    """
    altMinSense(M_Omega, Omega_mask, r, **kwargs)
    The alternating minimization algorithm for a matrix completion
    version of the matrix sensing problem
    
    Input
    max_iters : the maximum allowable number of iterations of the algorithm
    optCond : the optimality conditions that is measured 
              (default: absolute difference)
    optTol : the optimality tolerance used to determine stopping conditions
    solveLeft : a function to solve for the left matrix, Uj, on iteration j
                (default: mcFrobSolveLeftFactor_cvxpy)
    solveRight : a function to solve for the right matrix, Vj, on iteration j
                (default: mcFrobSolveRightFactor_cvxpy)
    solver : which solver to use (for cvxpy only) (default: SCS)
    verbose : 0 (none), 1 (light, default) or 2 (full) level of verbosity

    Ouptut
    U : the left m-by-r factor
    V : the right n-by-r factor
    """
    max_iters = kwargs.get('max_iters', 50)
    method = kwargs.get('method', 'cvx')
    optCond = kwargs.get('optCond', lambda x, y: np.abs(x - y))
    optTol = kwargs.get('optTol', 1e-4)
    solveLeft = kwargs.get('leftSolve', None)
    solveRight = kwargs.get('rightSolve', None)
    opts = kwargs.get('methodOptions', None)
    verbose = kwargs.get('verbose', 1)

    if method == 'cvx':
        solveLeft = mcFrobSolveLeftFactor_cvx
        solveRight = mcFrobSolveRightFactor_cvx
        if opts is None:
            opts = {'solver': SCS, 'verbose': verbose}
        elif opts.get('solver') is None:
            opts['solver'] = SCS

    if not verbose:
        verbose = False
        verbose_solve = False
    elif (verbose is True) or (verbose == 1):
        verbose = True
        verbose_solve = False
    elif (verbose == 2):
        verbose = True
        verbose_solve = True

    m, n = Omega_mask.shape
    # # Create initial guess from unbiased estimator # #
    # Set initial entries of estimator
    unbiased = np.zeros(Omega_mask.shape)
    unbiased[matIndicesFromMask(Omega_mask)] = M_Omega
    # scale entries of estimator by an estimate on the sampling
    # probability p so that this estimator is unbiased
    unbiased /= (M_Omega.size / Omega_mask.size)
    # compute svd of the unbiased estimator ### perhaps sparse SVD??
    unbiased_left, unbiased_sing, unbiased_right = np.linalg.svd(unbiased)
    U = unbiased_left[:, :r]
    objPrevious = np.inf
    for T in range(max_iters):
        V = solveRight(U, M_Omega, Omega_mask, **opts)
        U, objValue = solveLeft(V, M_Omega, Omega_mask, **opts,
                                returnObjectiveValue=True)
        
        if optCond(objValue, objPrevious) < optTol:
            print()
            print('Optimality conditions satisfied.')
            print('Objective value = {:5.3g}'.format(objValue))
            break
        else:
            if verbose:
                print('Iteration {}: Objective = {}'.format(T, objValue), end='\r')
            objPrevious = objValue
    return U, V
