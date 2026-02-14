import numpy as np 
from scipy.optimize import minimize, NonlinearConstraint


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
    
    @staticmethod
    def evaluate_generator_matrix(d, xs, centers, G):
        '''
        Constraint function for optimization of generation matrix initial guess.
        
        :param d: dimension of generator matrix
        :param xs: data points to fit
        :param centers: centers of zonotopes
        :param G: generator matrix
        '''
        xs = np.array(xs).T
        z_list = []
        for x in range(xs.shape[0]):
            z_list.append(np.linalg.norm((np.linalg.pinv(np.diag(d) @ G).T @ (xs[x] - centers.squeeze())), ord=np.inf))

        return z_list
    
    @staticmethod
    def evaluate_zonotope(xs, centers, G, m):
        '''
        Constraint function for optimization of zonotope.
        
        :param xs: data points to fit
        :param centers: centers of zonotopes
        :param G: generator matrix
        :param m: number of generators
        '''
        xs = np.array(xs).T
        n = np.shape(xs)[1]
        G = G.reshape((m, n))
        z_list = []
        for x in range(len(xs)):
            z_list.append(np.linalg.norm((np.linalg.pinv(G).T @ (xs[x] - centers).T), ord=np.inf))

        z_top = sorted(z_list, reverse=True)[:100]
        return z_top


    # Given a dataset of xs and centers of zonotope, fit zonotope to dataset
    def fit_zonotope(self, xs, centers):
        '''
        Fit a zonotope to the given data points xs and centers. The optimization is performed in two stages:        
            1. First, we optimize over the diagonal of the generator matrix G to find a good initial guess for G.
            2. Then, we perform full optimization over the entire generator matrix G with regularization to find the optimal zonotope.

        :param xs: data points to fit
        :param centers: centers of zonotopes
        '''

        xs = np.array(xs).T
        n = np.shape(xs)[0]

        ## Initial G matrix guesses:

        # Option 1:
        G = np.array([[0, 1],[1,0], [np.sqrt(2), np.sqrt(2)],[np.sqrt(2), -np.sqrt(2)]])

        # Option 2:
        # Generate G matrix to be more interesting (using Hanna's generator)
        # # G = generate_symmetric_G_matrix()

        # ##################################
        ## Quick start optimization by finding a good guess for G by optimizing over the diagonal of G (i.e. scaling each generator)
        g = G.shape[0]
        d0 = np.ones(g)

        obj = lambda d: np.log(np.linalg.det((np.diag(d) @ G).T @ (np.diag(d) @ G)))
        nlc = NonlinearConstraint(lambda d : Zonotope.evaluate_generator_matrix(d, xs, centers, G), -np.inf, 1)

        d0 = np.ones(g)

        res = minimize(obj, d0, constraints=(nlc), method="SLSQP", options = {'disp': True, 'maxiter': 1500})
        while hasattr(res, 'status') and res.status == 8:
            print("Exit Mode 8 detected: Positive directional derivative for linesearch.")
            res = minimize(obj, d0, constraints=(nlc), method="SLSQP", options = {'disp': True, 'maxiter': 1500})

        d_opt = res.x

        G0 = np.array(np.diag(d_opt) @ G).flatten()
        print("initial value: ", G0)

        # ##################################
        ## Perform full optimization, over the whole G matrix with regularization 
        m = 4
        def obj(G_flat):
            G = G_flat.reshape((m, n))
            epsilon = 1e-5
            return np.log(np.linalg.det(G.T @ G) + epsilon)
        nlc = NonlinearConstraint(lambda G : Zonotope.evaluate_zonotope(xs, centers, G, m), -np.inf, 1)

        res = minimize(obj, G0, constraints=(nlc), method="SLSQP", options = {'disp': True, 'maxiter': 1500})

        ## If the optimization fails, try again with a different initial G matrix (either by adding noise or by generating a new random orthonormal G matrix)
        has_failed = 0
        while hasattr(res, 'status') and res.status == 8:
            print("Exit Mode 8 detected: Positive directional derivative for linesearch.")
            has_failed += 1
            if has_failed > 3:
                G0 = Zonotope.get_random_orthonormal_G(m, n).flatten()
                has_failed=0
            else: 
                G0 = G0 + 1e-2 * np.random.randn(*G0.shape)
            res = minimize(obj, G0, constraints=(nlc), method="SLSQP", options = {'disp': True, 'maxiter': 1500})
            
        G_opt = res.x.reshape((m, n))
        return G_opt
