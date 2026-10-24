#Local/Neighbour-based methods.
import networkx as nx
import numpy as np
from typing import Callable, Generator, Tuple

#Helper function
def pairs(items):
    for i in range(len(items)):
        for j in range(i + 1, len(items)):
            yield items[i], items[j]

# A lot of local neighbourhood based methods follow this pattern:
def degree_based_index(degree_func: Callable) -> Callable:
    def index(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
        for (u, v, common_neighbours) in nx.common_neighbor_centrality(graph, alpha=1):
            denominator = degree_func(graph.degree[u], graph.degree[v])
            if denominator == 0:
                yield u, v, 0
            else:
                yield u, v, common_neighbours / denominator
    return index
salton_index = degree_based_index(lambda u, v: np.sqrt(u * v))
sorenson_index = degree_based_index(lambda u, v: (u + v) / 2)
hub_promoted_index = degree_based_index(min)
hub_depressed_index = degree_based_index(max)
lhn_index = degree_based_index(lambda u, v: u * v)

def CAR_common_neighbours(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    for u, v in nx.non_edges(graph):
        neighborhood = graph.subgraph(nx.common_neighbors(graph, u, v))
        yield u, v, neighborhood.number_of_nodes() * neighborhood.number_of_edges()
def CAR_resource_allocation(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    for u, v in nx.non_edges(graph):
        neighborhood = graph.subgraph(nx.common_neighbors(graph, u, v))
        score = sum(neighborhood.degree[node] / graph.degree[node] for node in neighborhood.nodes)
        yield u, v, score
def CAR_adamic_adar(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    for u, v in nx.non_edges(graph):
        neighborhood = graph.subgraph(nx.common_neighbors(graph, u, v))
        score = sum(neighborhood.degree[node] / np.log2(graph.degree[node]) for node in neighborhood.nodes)
        yield u, v, score
def CAR_preferential_attachment(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    for u, v in nx.non_edges(graph):
        neighborhood = graph.subgraph(nx.common_neighbors(graph, u, v))
        car = neighborhood.number_of_nodes() * neighborhood.number_of_edges()
        e_u = graph.degree[u] - neighborhood.number_of_nodes()
        e_v = graph.degree[v] - neighborhood.number_of_nodes()
        yield u, v,  e_u * e_v  +  e_u * car  +  e_v * car  +  car * car
def individual_attraction(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    for u, v in nx.non_edges(graph):
        neighborhood = graph.subgraph(nx.common_neighbors(graph, u, v))
        score = sum((neighborhood.degree[node] + 2) / graph.degree[node] for node in neighborhood.nodes)
        yield u, v, score

def local_naive_bayes(influence_func: Callable) -> Callable:
    def bayesian_index(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
        V = graph.number_of_nodes(); E = graph.number_of_edges()
        prior = 0.5 * V * (V - 1) / E   - 1

        #Optimization: remember node scores from last time they were needed.
        node_scores = dict()
        for u, v in nx.non_edges(graph):
            score = 0
            for neighbour in nx.common_neighbors(graph, u, v):
                if neighbour not in node_scores:
                    ego = graph.subgraph(graph.neighbors(neighbour))
                    ego_edges = ego.number_of_edges()
                    ego_non_edges = ego.number_of_nodes() * (ego.number_of_nodes() - 1) // 2  -  ego_edges
                    influence = (2 * ego_edges + 1) / (2 * ego_non_edges + 1)
                    node_scores[neighbour] = influence_func(graph.degree[neighbour]) * np.log2(prior * influence)
                score += node_scores[neighbour]
            yield u, v, score
    return bayesian_index
bayesian_common_neighbours = local_naive_bayes(lambda d: 1)
bayesian_adamic_adar = local_naive_bayes(lambda d: 1 / np.log2(d))
bayesian_resource_allocation = local_naive_bayes(lambda d: 1 / d)

def functional_similarity_weight(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    degrees = [degree for (node, degree) in graph.degree]
    avg_degree = np.array(degrees).mean()
    for u, v in nx.non_edges(graph):
        neighbours_u = set(graph[u])
        neighbours_v = set(graph[v])
        num_cn = len(neighbours_u & neighbours_v)
        num_ucn = len(neighbours_u - neighbours_v)
        lmb = max(0, avg_degree - (num_ucn + num_cn))

        yield u, v, ( 2*num_cn / (num_ucn + 2*num_cn + lmb) )**2

def local_interacting_score(graph: nx.Graph, n_iterations=2) -> Generator[Tuple[int, int, float], None, None]:
    scores = dict()
    for u, v in pairs(list(graph.nodes)):
        scores[(u, v)] = scores[(v, u)] = 1 if graph.has_edge(u, v) else 0

    for _ in range(n_iterations):
        new_scores = dict()
        global_score = 2 * sum(scores[edge] for edge in graph.edges) / graph.number_of_nodes()
        for u, v in pairs(list(graph.nodes)):
            common_score = sum(scores[(u, n)] + scores[(v, n)] for n in nx.common_neighbors(graph, u, v))
            u_score = sum(scores[(u, n)] for n in graph[u])
            v_score = sum(scores[(v, n)] for n in graph[v])
            lmb_u = max(0, global_score - u_score)
            lmb_v = max(0, global_score - v_score)
            new_score = common_score / (u_score + v_score + lmb_u + lmb_v)
            new_scores[(u, v)] = new_scores[(v, u)] = new_score
        scores = new_scores

    for u, v in nx.non_edges(graph):
        yield u, v, scores[(u, v)]

def clustering_coefficient_index(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    clusterings = nx.clustering(graph)
    for u, v in nx.non_edges(graph):
        score = sum(clusterings[n] for n in nx.common_neighbors(graph, u, v))
        yield u, v, score
def link_clustering_index(graph: nx.Graph) -> Generator[Tuple[int, int, float], None, None]:
    clusterings = nx.clustering(graph)
    num_common_neighbours = dict()
    for u, v in pairs(list(graph.nodes)):
        num_common_neighbours[(u, v)] = num_common_neighbours[(v, u)] = sum(1 for _ in nx.common_neighbors(graph, u, v))
    for u, v in nx.non_edges(graph):
        score = 0
        for n in nx.common_neighbors(graph, u, v):
            score += (num_common_neighbours[(u, n)] + num_common_neighbours[(v, n)]) \
                     * clusterings[n] \
                     / (graph.degree[n] - 1)
        yield u, v, score