import math
from typing import Dict, Tuple, Literal
from Analytic.ProbabilityFunctions import *

def sign(n: int):
    if n == 0: return 0
    elif n > 0: return 1
    return -1

#Calculates all relevant probabilities in one pass over the ER graph (baseline conditions).
def er_probabilities(p: float = 0.05, N: int = 101, frac_hidden: float = 0.25) -> float:
    max_edges = N - 2
    total_prob = 0.0
    P_concordant = 0.0
    P_discordant = 0.0
    P_tie = 0.0
    P_tie_true = 0.0
    P_tie_vis = 0.0
    for dvis_delta in range(-max_edges, max_edges + 1):
        for removed_delta in range(-max_edges, max_edges + 1):
            prob = skellam_pmf(dvis_delta, max_edges * p * (1.0 - frac_hidden)) * \
                   skellam_pmf(removed_delta, max_edges * p * frac_hidden)
            total_prob += prob
            sign_true = sign(dvis_delta + removed_delta)
            sign_vis = sign(dvis_delta)
            if sign_true == 0 or sign_vis == 0:
                P_tie += prob
                if sign_true == 0:
                    P_tie_true += prob
                if sign_vis == 0:
                    P_tie_vis += prob
            elif sign_true == sign_vis:
                P_concordant += prob
            else:
                P_discordant += prob
     #The idea is that the total_prob is a canary for numerical stability issues.
    print(f"Total prob (should be ~1): {total_prob} = {P_concordant + P_discordant + P_tie}")
    return {
        "concordant": P_concordant,
        "discordant": P_discordant,
        "tie": P_tie,
        "tie_true": P_tie_true,
        "tie_vis": P_tie_vis
    }

def cm_probabilities(m: int = 3, N: int = 101, frac_hidden: float = 0.25) -> float:
    total_prob = 0.0
    P_concordant = 0.0
    P_discordant = 0.0
    P_tie = 0.0
    P_tie_true = 0.0
    P_tie_vis = 0.0
    for d_u in range(m, N + 1):
        for d_v in range(m, N + 1):
            for dvis_u in range(d_u + 1):
                for dvis_v in range(d_v + 1):
                    prob = scalefree_pmf(d_u, m) * scalefree_pmf(d_v, m) \
                           * binom_pmf(dvis_u, d_u, 1.0 - frac_hidden) * binom_pmf(dvis_v, d_v, 1.0 - frac_hidden)
                    total_prob += prob
                    sign_true = sign(d_u - d_v)
                    sign_vis = sign(dvis_u - dvis_v)
                    if sign_true == 0 or sign_vis == 0:
                        P_tie += prob
                        if sign_true == 0:
                            P_tie_true += prob
                        if sign_vis == 0:
                            P_tie_vis += prob
                    elif sign_true == sign_vis:
                        P_concordant += prob
                    else:
                        P_discordant += prob
    #The idea is that the total_prob is a canary for numerical stability issues.
    print(f"Total prob (should be ~1): {total_prob} = {P_concordant + P_discordant + P_tie}")
    return {
        "concordant": P_concordant,
        "discordant": P_discordant,
        "tie": P_tie,
        "tie_true": P_tie_true,
        "tie_vis": P_tie_vis
    }

#Returns Kendall-Tau b.
def baseline(probabilities: Dict) -> float:
    expected_score = probabilities['concordant'] - probabilities['discordant']
    var_true = 1.0 - probabilities['tie_true']
    var_vis = 1.0 - probabilities['tie_vis']
    return expected_score / math.sqrt(var_true * var_vis)

def er_baseline(p: float = 0.05, N: int = 101, frac_hidden: float = 0.25) -> float:
    return baseline(er_probabilities(p, N, frac_hidden))
def cm_baseline(m: int = 3, N: int = 101, frac_hidden: float = 0.25) -> float:
    return baseline(cm_probabilities(m, N, frac_hidden))