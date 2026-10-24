import networkx as nx
from typing import Dict, Any
import numpy as np

from Analytic.ProbabilityFunctions import *

class DegreeReconstructor:
    def reconstruct_degree(self, graph: nx.Graph) -> Dict[Any, float]:
        raise NotImplementedError("Subclasses should implement a method for reconstructing the degree in a graph with missing edges. Return a dictionary that maps each node to its degree.")
    def update_model(self, removed_node: int) -> None:
        raise NotImplementedError("Subclasses should implement a method for updating the reconstructor's mental model of the graph when a node is removed. (e.g. to remember that this node is deliberately gone and should not be predicted back into existence)")
    def reset_model(self) -> None:
        raise NotImplementedError("Subclasses should implement a method for resetting the mental model of the graph, putting the reconstructor back into a fresh state.")

class BaselineDegreeReconstructor(DegreeReconstructor):
    def reconstruct_degree(self, graph: nx.Graph) -> Dict[Any, float]:
        #Just return the degree in the new graph and hope it still correlates well with the old.
        return nx.degree_centrality(graph)
    def update_model(self, removed_node): pass
    def reset_model(self): pass

#Reconstructs degree using mathematically optimal link predictor for ER networks.
class EROptimalLinkPrediction(DegreeReconstructor):
    def __init__(self, threshold: float = 0.25):
        self.threshold = threshold
        self.reconstructed_graph = None

    def link_predictor(self, graph: nx.Graph):
        #There is only one link predictor possible for random graph, see Theory.md.
        p = 0.05
        for u, v in nx.non_edges(graph):
            yield u, v, p / (4 - 3*p)

    def add_edges(self, graph: nx.Graph):
        for (u, v, link_strength) in self.link_predictor(graph):
            if link_strength >= self.threshold:
                graph.add_edge(u, v)
    
    def reconstruct_degree(self, graph: nx.Graph) -> Dict[Any, float]:
        if not self.reconstructed_graph:
            self.reconstructed_graph = graph.copy()
            for (u, v, link_strength) in self.link_predictor(graph):
                if link_strength >= self.threshold:
                    self.reconstructed_graph.add_edge(u, v)
        return nx.degree_centrality(self.reconstructed_graph)
    
    def update_model(self, removed_node):
        self.reconstructed_graph.remove_node(removed_node)
    def reset_model(self):
        self.reconstructed_graph = None


class CMOptimalLinkPrediction(DegreeReconstructor):
    def __init__(self, threshold: float = 0.05):
        self.threshold = threshold
        self.reconstructed_graph = None

    def link_predictor(self, graph: nx.Graph):
        #This is the optimal link predictor for a scale-free graph, see Theory.md.
        N = 101; m = 3
        for (u, v) in nx.non_edges(graph):
            dvis_u = graph.degree[u]; dvis_v = graph.degree[v]
            dvis_rst = graph.number_of_edges() * 2 - dvis_u - dvis_v
            d_rst = dvis_rst / 0.75 #point estimate for d_rst

            #Precompute separable parts to get as much out of the O(N^2) loop as possible.
            du_cache = np.array([scalefree_pmf(d_u, m) * binom_pmf(dvis_u, d_u, 0.75) for d_u in range(N+1)])
            dv_cache = np.array([scalefree_pmf(d_v, m) * binom_pmf(dvis_v, d_v, 0.75) for d_v in range(N+1)])

            total_prob_d = 0.0
            total_prob_edge = 0.0
            for d_u in range(max(m, dvis_u), N+1):
                for d_v in range(max(m, dvis_v), N+1):
                    prob_d = du_cache[d_u] * dv_cache[d_v] #* ~1 for d_rst
                    
                    r_u = d_u - dvis_u; r_v = d_v - dvis_v; r_tot = r_u + r_v + d_rst - dvis_rst
                    if r_tot <= 1:
                        prob_edge = 0
                    else:
                        prob_edge = prob_d * (r_u * r_v) / (r_tot - 1)

                    total_prob_d += prob_d
                    total_prob_edge += prob_edge
            
            # Normalisation constant = 1 / total[P(d | Gvis)]
            # Final prob = normalisation constant * total[P(d | Gvis) * P(edge | d, Gvis)]
            # == total[P(d | Gvis) * P(edge | d, Gvis)] / total[P(d | Gvis)]
            prob_uv = total_prob_edge / total_prob_d
            yield u, v, prob_uv

    def add_edges(self, graph: nx.Graph):
        for (u, v, link_strength) in self.link_predictor(graph):
            if link_strength >= self.threshold:
                graph.add_edge(u, v)
    
    def reconstruct_degree(self, graph: nx.Graph) -> Dict[Any, float]:
        if not self.reconstructed_graph:
            self.reconstructed_graph = graph.copy()
            for (u, v, link_strength) in self.link_predictor(graph):
                if link_strength >= self.threshold:
                    self.reconstructed_graph.add_edge(u, v)
        return nx.degree_centrality(self.reconstructed_graph)
    
    def update_model(self, removed_node):
        self.reconstructed_graph.remove_node(removed_node)
    def reset_model(self):
        self.reconstructed_graph = None