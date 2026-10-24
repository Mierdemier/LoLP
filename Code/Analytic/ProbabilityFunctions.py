from scipy.stats import binom, norm, skellam
import random

def scalefree_pmf(k: int, m: int) -> float:
    if k < m: return 0.0
    return ( 2 * m * (m + 1) ) / ( k * (k + 1) * (k + 2) )
def sample_scalefree(m: int, N: int):
    uniform = random.random()
    for k in range(m, N + 1):
        uniform -= scalefree_pmf(k, m)
        if uniform <= 0:
            return k
    return N

binom_cache = dict()
def binom_pmf(k: int, N: int, p: float) -> float:
    if (k, N, p) not in binom_cache:
        binom_cache[(k, N, p)] = binom.pmf(k, N, p)
    return binom_cache[(k, N, p)]

skellam_cache = dict()
def skellam_pmf(k: int, lmb: float) -> float:
    if (k, lmb) not in skellam_cache:
        skellam_cache[(k, lmb)] = skellam.pmf(k, lmb, lmb)
    return skellam_cache[(k, lmb)]

gaussian_cache = dict()
def gaussian_pmf(k, mu, sigma):
    if (k, mu, sigma) not in gaussian_cache:
        gaussian_cache[(k, mu, sigma)] = norm.cdf(k + 0.5, loc=mu, scale=sigma) - \
                                         norm.cdf(k - 0.5, loc=mu, scale=sigma)
    return gaussian_cache[(k, mu, sigma)]