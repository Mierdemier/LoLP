#Global methods.
import networkx as nx
import numpy as np
from typing import Generator, Tuple, List, Set

from LinkPredictors.GlobalSimilarity import transition_matrix, walk_randomly

def local_path_index(graph: nx.Graph, beta: float = 0.01, l: int = 3) -> Generator[Tuple[int, int, float], None, None]:
    A = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    damping = 1.0

    S = np.zeros_like(A, dtype=np.float64)
    path_matrix = A @ A #start at paths of length 2.
    for _ in range(1, l):
        S += damping * path_matrix
        path_matrix @= A
        damping *= beta
    
    for u, v in nx.non_edges(graph):
        yield u, v, S[u, v]

def friend_link(graph: nx.Graph, l: int = 3) -> Generator[Tuple[int, int, float], None, None]:
    A = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    path_matrix = A
    normaliser = 1

    S = np.zeros_like(A, dtype=np.float64)
    for i in range(2, l + 1):
        path_matrix @= A
        normaliser *= graph.number_of_nodes() - i
        S += (1.0 / (i - 1)) * (path_matrix / normaliser)
    
    for u,v in nx.non_edges(graph):
        yield u, v, S[u, v]

#The original L3 index (eq. 42 in Link Prediction Techniques)
def L3_index(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    for u, v in nx.non_edges(graph):
        score = 0
        for x in nx.neighbors(graph, u):
            for y in nx.neighbors(graph, v):
                if graph.has_edge(x, y): score += 1.0 / np.sqrt(graph.degree[x] * graph.degree[y])
        yield u, v, score

def local_random_walk(graph: nx.Graph, t: int = 3) -> Generator[Tuple[int, int, float], None, None]:
    M = transition_matrix(graph)
    P = walk_randomly(M, t)
    normaliser = 2.0 * graph.number_of_edges()

    for u, v in nx.non_edges(graph):
        score_uv = graph.degree[u] / normaliser * P[u, v]
        score_vu = graph.degree[v] / normaliser * P[v, u]
        yield u, v, score_uv + score_vu

def superposed_random_walk(graph: nx.Graph, t: int = 3) -> Generator[Tuple[int, int, float], None, None]:
    M = transition_matrix(graph)
    I = np.eye(M.shape[0], M.shape[1])
    normaliser = 2.0 * graph.number_of_edges()

    P = np.zeros((t+1, M.shape[0], M.shape[1]))
    P[0] = I.copy()
    for tau in range(1, t + 1):
        P[tau] = P[tau - 1] @ M.transpose()
    P_sum = P[1:].sum(axis=0)

    for u, v in nx.non_edges(graph):
        score_uv = graph.degree[u] / normaliser * P_sum[u, v]
        score_vu = graph.degree[v] / normaliser * P_sum[v, u]
        yield u, v, score_uv + score_vu


#Helper for propflow:
#Returns similarity between source and every other node within range l.
#Trying to look up similarity for a node more than l away from source will give 0.
def single_source_propflow(graph: nx.Graph, source: int, l: int) -> np.ndarray:
    found = {source}
    new_search = {source}
    similarity = np.array([1.0 if dst == source else 0.0 for dst in graph.nodes])

    for _ in range(l):
        old_search = new_search
        new_search = set()
        for node in old_search:
            for neighbour in graph.neighbors(node):
                similarity[neighbour] += similarity[node] / graph.degree[node]
                if neighbour not in found:
                    found.add(neighbour)
                    new_search.add(neighbour)
    return similarity

def propflow(graph: nx.Graph, l: int = 3) -> Generator[Tuple[int, int, float], None, None]:
    already_searched = dict()
    for u, v in nx.non_edges(graph):
        if u in already_searched: score = already_searched[u][v]
        elif v in already_searched: score = already_searched[v][u]
        else:
            similarity_u = single_source_propflow(graph, u, l)
            already_searched[u] = similarity_u
            score = similarity_u[v]
        yield u, v, score
