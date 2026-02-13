import numpy as np 
import cvxpy as cp
from sklearn.cluster import KMeans
import matplotlib.pyplot as plt
from scipy.optimize import minimize, NonlinearConstraint
from itertools import product
from scipy.spatial import ConvexHull
from matplotlib.patches import Polygon

class Zonotope:
    def __init__(self):
       pass 

    @staticmethod
    def get_random_orthonormal_G(m, n):
        '''
        Generate an orthonormal matrix G to help "quick start" the optimization.
        '''
        Q, _ = np.linalg.qr(np.random.randn(n, m))
        return Q.T