from ellipsoid_utils import in_ellipsoid
from probabilistic_guarantees.binomial_utils import binomial_tail

# Calculating accuracy of ellipsoidal reachable set or tube given a test dataset, 
# using the binomial tail to get a probabilistic guarantee on the accuracy level.

################################################################################################################################################
################################################################################################################################################

# Ellipsoidal reachable set:

def epsilon_set(data, A_star, b_star, num_samples):
    # Check how many points fall outside a given ellipsoid given a test data set
    misses = 0
    for i in range(len(data)):
        test = in_ellipsoid(A_star, b_star, data[i])
        if test - 1 > 0:
            misses += 1

    # Calculate the binomial tail to get the accuracy level
    p_estimate = binomial_tail(misses, num_samples)
    return p_estimate

################################################################################################################################################
################################################################################################################################################

# Ellipsoidal tube:

# Check how many points fall outside the ellipsoid given a test data set
# Defined separately to allow for checking of every time step in a trajectory, or checking of a set of trajectories
def tube_misses(data, A_star, b_star, num_samples, ax):
        misses = 0
        for i in range(len(data)):
            test = in_ellipsoid(A_star, b_star, data[i])
            if test - 1 > 0:
                misses += 1
                ax.plot(data[i, 0], data[i, 1], marker=".", color="green", alpha=0.7, linestyle='None')
        return misses

# Compute the binomial tail to get the accuracy level
def epsilon_tube(misses, num_samples):
    p_estimate = binomial_tail(misses, num_samples)
    return p_estimate


