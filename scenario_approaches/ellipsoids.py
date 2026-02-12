import numpy as np 
import cvxpy as cp
from scipy.special import gamma
from scipy.optimize import minimize, NonlinearConstraint
from numpy.linalg import det, inv

class Ellipsoid:
    # Save Ellipsoid in form (x-c)'Q(x-c) 
    def __init__(self, n = 2):
        self.Q = np.eye(n)
        self.c = np.zeros((n, 1))
        self.r = 1

    # Given a dataset of xs and norm degree, fit ellipse to dataset
    # Ellipse in form ||Ax - b|| <= 1
    def fit_p_ball(self, xs, normp, verbose=False):
        xs = np.array(xs).T
        n = np.shape(xs)[0]
        m = np.shape(xs)[1]

        A = cp.Variable((n, n), symmetric=True)
        b = cp.Variable((n,1))

        # ------------------------------------------------------------------
        # objective --------------------------------------------------------
        objective = cp.Maximize(cp.log_det(A))

        # ------------------------------------------------------------------
        # constraints ------------------------------------------------------
        residual = A @ xs - b @ np.ones((1,m))
        col_norm_constraints = [
            cp.norm(residual[:, j], normp) <= 1
            for j in range(m)
        ]

        constraints = col_norm_constraints

        # ------------------------------------------------------------------
        # solve ------------------------------------------------------------
        problem = cp.Problem(objective, constraints)
        problem.solve(solver=cp.SCS)

        if(verbose):
            print("Optimal value :", problem.value)
            print("A* =", A.value)
            print("b* =", b.value)
            print("volume proxy:", np.log(np.linalg.det(A.value)))

            # Check if the optimization was successful
            if problem.status not in ["optimal", "optimal_inaccurate"]:
                print("Optimization failed with status:", problem.status)
            else:
                print("Optimization succeeded.")

        Q, c, r = self.convert_A_b_to_Q_c_r(A.value, b.value) 
        Q /= r
        self.Q = Q
        self.c = c
        # self.A = A.value
        # self.b = b.value
        self.L = np.linalg.cholesky((self.Q + self.Q.T)/2)
        return problem.value
    
    @staticmethod 
    def convert_A_b_to_Q_c_r(A, b):
        """
        Convert ellipsoid of form ||A x - b|| <= 1
        into standard ellipsoid form: (x - c)^T Q (x - c) <= r
        Returns (Q, c, r)
        """
        Q = A.T @ A
        Q_inv = np.linalg.inv(Q)
        c = Q_inv @ A.T @ b
        r = (b.T @ A @ Q_inv @ A.T @ b) - (b.T @ b) + 1
        return Q, c, r

    @staticmethod
    def in_ellipsoid(A, b, p):
        """
        Check if point p is in ellipsoid defined by ||A x - b|| <= 1
        Returns True or False
        """
        p = np.asarray(p,  dtype=float).ravel()
        b = np.asarray(b,  dtype=float).ravel()  
        return np.linalg.norm(A @ p - b) - 1 <= 0
    
    @staticmethod
    def plot_ellipse(Q, c, r, ax, alpha=0.75, color='green', **kwargs):
        """
        Plot ellipse defined by (x-c)^T Q (x-c) <= r in 2D
        """
        eigvals, eigvecs = np.linalg.eigh(Q)
        theta = np.linspace(0, 2 * np.pi, 200)
        circle = np.stack([np.cos(theta), np.sin(theta)])
        # Scale the ellipse by sqrt(r)
        ellipse = eigvecs @ np.diag(np.sqrt(r) / np.sqrt(eigvals)) @ circle
        ellipse = ellipse.T + c
        ax.plot(ellipse[:, 0], ellipse[:, 1], **kwargs, color=color, alpha=alpha)

        return ax
    

