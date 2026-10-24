#Global methods.
import networkx as nx
import numpy as np
from typing import Generator, Tuple, List, Set

#Helpers

#Helpers for random walks.
#Transition matrix for a random walker.
def transition_matrix(graph: nx.Graph):
    result = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()), dtype=np.float64)
    for u in graph.nodes:
        for v in graph.nodes:
            result[u, v] = 1.0 / graph.degree[u] if graph.has_edge(u, v) else 0.0
    return result

#For calculating the distribution of where a random walker ends up from each starting point.
def walk_randomly(M: np.ndarray, n_iters: int, alpha: float = 1.0) -> np.ndarray:
    """
    M is a transition matrix. 
    Obtain with transition_matrix() for a generic random walker. Or make custom probabilities.
    Optionally include alpha: (1 - alpha) = probability of starting over.
    Returns P: a matrix where P[i, j] = probability of ending up at j, starting from i.
    """
    #Potential optimisations:
    #- Cache M.transpose()
    #- Use sparse matrices.
    I = np.eye(M.shape[0], M.shape[1])
    P = I.copy() #Starting state: 100% probability of being where you started.
    for t in range(n_iters):
        P = alpha * (P @ M.transpose()) + (1 - alpha) * I
    return P


#Helpers for working with disconnected graphs (an edge case many papers don't explicitly address :( )
def index_of_component(components: List[Set[int]], u: int) -> int:
    for i, component in enumerate(components):
        if u in component: return i
    return -1

#Determines whether there is a path between u and v based on a list of connected components.
#Much faster than calling nx.has_path on real-world graphs if you need to do multiple calls.
def fast_has_path(components: List[Set[int]], u: int, v: int) -> bool:
    for component in components:
        if u in component and v in component:
            return True
    return False



#Start of link prediction methods

def negative_shortest_path_lengths(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    path_lengths = dict(nx.all_pairs_shortest_path_length(graph))
    for u, v in nx.non_edges(graph):
        if v in path_lengths[u]:
            yield u, v, -1 * path_lengths[u][v]

def katz_index(graph: nx.Graph, alpha=0.01) -> Generator[Tuple[int, int, float], None, None]:
    A = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    I = np.eye(A.shape[0], A.shape[1])
    result = np.linalg.inv(I - alpha * A) - I
    for u in range(A.shape[0]):
        for v in range(A.shape[1]):
            yield (u, v, result[u, v])

def cosine_laplacian(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    components = list(nx.connected_components(graph))
    L = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    L_plus = np.linalg.pinv(L)
    for u, v in nx.non_edges(graph):
        if not fast_has_path(components, u, v): continue
        yield u, v, L_plus[u, v] / np.sqrt(L_plus[u, u] * L_plus[v, v])

#Note: average_commute_time does not use the random walk helpers because it has an analytical solution.
def average_commute_time(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    components = list(nx.connected_components(graph))
    L = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    L_plus = np.linalg.pinv(L)
    for u, v in nx.non_edges(graph):
        if not fast_has_path(components, u, v): continue
        time = L_plus[u, u] + L_plus[v, v] - 2*L_plus[u, v]
        yield u, v, 1.0 / time

def matrix_forest_index(graph: nx.Graph, alpha: float = 0.25) -> Generator[Tuple[int, int, float], None, None]:
    components = list(nx.connected_components(graph))
    L = nx.laplacian_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    I = np.eye(L.shape[0], L.shape[1])
    S = np.linalg.inv(I + alpha * L)
    for u, v in nx.non_edges(graph):
        if not fast_has_path(components, u, v): continue
        yield u, v, S[u, v]

def global_lhn(graph: nx.Graph, alpha: float = 0.10) -> Generator[Tuple[int, int, float], None, None]:
    D_inv = np.zeros(( graph.number_of_nodes(), graph.number_of_nodes() ))
    for node in graph.nodes: 
        if graph.degree[node] == 0:
            #The expected number of paths is 0, making the normalisation factor undefined.
            #The paper does not say anything about this.
            #But I assume that these nodes simply don't contribute similarity.
            D_inv[node, node] = 0.0
        else:
            D_inv[node, node] = 1.0 / graph.degree[node]

    A = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    lmb = np.linalg.eigvalsh(A)[-1] #A is real symmetric so we can use this fast method.
    I = np.eye(A.shape[0], A.shape[1])

    S = D_inv @ np.linalg.inv(I - (alpha / lmb) * A) @ D_inv
    for u, v in nx.non_edges(graph):
        yield (u, v, S[u, v])

def normalised_average_commute_time(graph: nx.Graph, n_iters: int = 10) -> Generator[Tuple[int, int, float], None, None]:
    M = transition_matrix(graph)
    P = walk_randomly(M, n_iters)
    #The stationary distribution is not well-defined for an unconnected graph.
    #So I will calculate a separate stationary distribution for each component.
    components = list(nx.connected_components(graph))
    sum_of_degrees = [sum(graph.degree[u] for u in comp) for comp in components]

    for u, v in nx.non_edges(graph):
        if not fast_has_path(components, u, v): continue
        if P[u, v] == 0 or P[v, u] == 0: continue #Can happen if n_iters is lower than graph diameter.
        prob_u = graph.degree[u] / sum_of_degrees[index_of_component(components, u)]
        prob_v = graph.degree[v] / sum_of_degrees[index_of_component(components, v)]
        steps_uv = 1.0 / P[u, v]; steps_vu = 1.0 / P[v, u]
        yield u, v, 1.0 / (steps_uv * prob_v + steps_vu * prob_u)

def random_walk_restarts(graph: nx.Graph, alpha: float = 0.50, n_iters: int = 60) -> Generator[Tuple[int, int, float], None, None]:
    M = transition_matrix(graph)
    P = walk_randomly(M, n_iters, alpha)
    for u, v in nx.non_edges(graph):
        yield u, v, P[u, v] + P[v, u]

def max_entropy_random_walk(graph: nx.Graph, alpha: float = 0.50, n_iters: int = 60) -> Generator[Tuple[int, int, float], None, None]:
    A = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    M = np.zeros_like(A, dtype=np.float64)
    
    #Unlike the original method, we need to deal with the case where the graph is not connected.
    #I do this by treating each component as a separate graph and then stitching the Ms together.
    for component in nx.connected_components(graph):
        idx = np.array(sorted(component))
        if len(idx) == 1:
            continue #Isolated node. Skip.
        A_local = A[np.ix_(idx, idx)]
        eigenvalues, eigenvectors = np.linalg.eigh(A_local)
        lmb = eigenvalues[-1]; psi = np.abs(eigenvectors[:, -1])
        M_local = (A_local / lmb) * np.outer(1.0 / psi, psi)
        M[np.ix_(idx, idx)] = M_local

    P = walk_randomly(M, n_iters, alpha)
    for u, v in nx.non_edges(graph):
        yield u, v, P[u, v] + P[v, u]

def flow_propagation(graph: nx.Graph, alpha: float = 0.50, n_iters: int = 60) -> Generator[Tuple[int, int, float], None, None]:
    D = np.zeros((graph.number_of_nodes(), graph.number_of_nodes()))
    for node in graph.nodes:
        if graph.degree[node] != 0:
            D[node, node] = 1.0 / np.sqrt(graph.degree[node])
    #Note: D_left = D_right in an undirected graph bc the in-degree == the out-degree.
    M = D @ nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray() @ D
    P = walk_randomly(M, n_iters, alpha)
    for u, v in nx.non_edges(graph):
        yield u, v, P[u, v] + P[v, u]

def blondel_index(graph: nx.Graph, n_iters: int = 51) -> Generator[Tuple[int, int, float], None, None]:
    if n_iters % 2 == 0:
        raise ValueError(f"The blondel index only converges on odd iterations. Cannot calculate the blondel index for {n_iters} iterations.")
   
    A = nx.adjacency_matrix(graph, nodelist=sorted(graph.nodes())).toarray()
    S = np.eye(A.shape[0], A.shape[1])
    for _ in range(1, n_iters):
        S = (A @ S @ A.transpose()) + (A.transpose() @ S @ A)
        S /= np.linalg.norm(S)

    for u, v in nx.non_edges(graph):
        yield u, v, S[u, v]