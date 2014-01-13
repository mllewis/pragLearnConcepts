from nips import *

from scipy.special import gammaln

# two agents each have dirichlet priors on a set of object -> word
# multinomial distributions.
# without pragmatics we can't condition on listener interpretation, so
# instead, they take turns speaking

SIZE = 2

def DM_dist(alpha):
    # Joint density for N samples from a dirichlet-multinomial
    # distribution with parameters alpha[i], and n_k
    # repeats of each outcome is
    #   Z * prod_k Gamma(n_k + alpha[k])
    # where:
    #   Z = Gamma(A) / Gamma(N + A) * prod_k 1/Gamma(alpha[k])
    #   A = sum_k(alpha[k])
    #   N = sum_k(n_k)
    # For us N = 1
    # So basically P(i|alpha) \propto gamma(1 + alpha[i])/gamma(alpha[i])
    dist = gammaln(1 + alpha) - gammaln(alpha)
    dist -= np.logaddexp.reduce(dist)
    return dist

def marginal_S_dist(alpha):
    dist = np.empty((SIZE, SIZE))
    for obj in xrange(SIZE):
        dist[:, obj] = DM_dist(alpha[:, obj])
    assert np.allclose(np.logaddexp.reduce(dist, axis=0), 1)
    return dist

def L_dist_for(S_dist):
    L_dist = S_dist.copy()
    L_dist -= np.logaddexp.reduce(L_dist, axis=1)[:, np.newaxis]
    assert np.allclose(np.logaddexp.reduce(L_dist, axis=1), 1)
    return L_dist

def sample_S_dist(r, target, alpha):
    S_dist = marginal_S_dist(alpha)
    target_dist = S_dist[:, target]
    linear = np.exp(target_dist)
    assert np.allclose(np.sum(linear), 1)
    return conpact2.weighted_choice(r, linear)

def simulate_non_pragmatic(r, turns, prior=None):
    a1_alpha = np.ones((SIZE, SIZE))
    a2_alpha = np.ones((SIZE, SIZE))

    for i in xrange(turns):
        for S_alpha, L_alpha in [(a1_alpha, a2_alpha),
                                 (a2_alpha, a1_alpha)]:
            S_dist = marginal_S_dist(S_alpha)
            L_dist = L_dist_for(marginal_S_dist(L_alpha))
            target = r.randint(SIZE)
            word = conpact2.weighted_choice(r, np.exp(S_dist[:, target]))
            L_alpha[word, target] += 1

            P_understood =

    # Want out:
    #   P(understood)
    #   P(obj|word) using both posteriors
