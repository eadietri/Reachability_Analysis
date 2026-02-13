import math
import numpy as np
from scipy.optimize import minimize, NonlinearConstraint
from scipy.stats import binom 

# Binomial tail inversion to calculate accuracy level given a test datasest
def binomial_tail(misses: int, num_samples: int) -> float:
    """
    Compute the largest p such that:
        BinomialCDF(k=misses; n=num_samples, p) >= 1e-9
    """
    k = misses
    n = num_samples

    def objective(p):
        return -p  # maximize p
    
    def binom_cdf(e, k, l):
        return binom.cdf(k, l, e)
    
    def binom_cdf_if_fails(e, k, l):
        binom_sum = 0
        for j in range(0, k+1):
            if not math.isnan(binom.pmf(j, l, e)[0]):
                binom_sum += binom.pmf(j, l, e)
        return binom_sum
    
    lc = NonlinearConstraint((lambda p : binom_cdf(p, k, n) - 0.000000001), lb=0, ub=1)
    result = minimize(objective, x0=k/n, constraints=lc, method='SLSQP')

    if not result.success:
        print("Optimization failed:", result.message, " re-running with different cdf function")
        lc = NonlinearConstraint((lambda p : binom_cdf_if_fails(p, k, n) - 0.000000001), lb=0, ub=1)
        result = minimize(objective, x0=k/n, constraints=lc, method='SLSQP')

    return result.x[0]

if __name__ == '__main__':
    # Test cases
    print("8 :", binomial_tail(8, 1500))