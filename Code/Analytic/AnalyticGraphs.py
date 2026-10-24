import networkx as nx
import random

from Analytic.ProbabilityFunctions import scalefree_pmf, sample_scalefree

def load_er_graph() -> nx.Graph:
    return nx.erdos_renyi_graph(n=101, p=0.05)

def load_scalefree_graph() -> nx.MultiGraph:
    N = 101; m = 3
    deg_sequence = [sample_scalefree(m, N) for _ in range(N)]
    if sum(deg_sequence) % 2 == 1:
        #Fix with negligible change.
        deg_sequence[-1] -= 1 #Note: deg_sequence[-1] is guaranteed to be >= m.
    return nx.configuration_model(deg_sequence)

