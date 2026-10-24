import networkx as nx
import numpy as np
from typing import Dict, Any, Callable, Tuple

class Reconstructor:
    def reconstruct_betweenness(self, graph: nx.Graph) -> Dict[Any, float]:
        raise NotImplementedError("Subclasses should implement a method for reconstructing the betweenness in a graph with missing edges. Return a dictionary that maps each node to its betweenness.")
    def update_model(self, removed_node: int) -> None:
        raise NotImplementedError("Subclasses should implement a method for updating the reconstructor's mental model of the graph when a node is removed. (e.g. to remember that this node is deliberately gone and should not be predicted back into existence)")
    def reset_model(self) -> None:
        raise NotImplementedError("Subclasses should implement a method for resetting the mental model of the graph, putting the reconstructor back into a fresh state.")
    
class BaselineReconstructor(Reconstructor):
    def reconstruct_betweenness(self, graph: nx.Graph) -> Dict[Any, float]:
        #Just return the betweenness in the new graph and hope it still correlates well with the old.
        return nx.betweenness_centrality(graph)
    def update_model(self, removed_node): pass
    def reset_model(self): pass
    
#Results of link prediction search (link_predictor, suggested threshold.)
link_predictors: Tuple[Callable, float] = [
    (nx.jaccard_coefficient, 0.25),
    (lambda g: nx.common_neighbor_centrality(g, alpha=1), 4),
    (nx.resource_allocation_index, 0.475),
    (nx.adamic_adar_index, 1.50),
]

class LinkPredictionReconstructor(Reconstructor):
    def __init__(self, link_predictor: Callable = nx.jaccard_coefficient, threshold: float = 0.25):
        self.link_predictor = link_predictor
        self.threshold = threshold
        self.reconstructed_graph = None

    def add_edges(self, graph: nx.Graph):
        for (u, v, link_strength) in self.link_predictor(graph):
            if link_strength >= self.threshold:
                graph.add_edge(u, v)
    
    def reconstruct_betweenness(self, graph: nx.Graph) -> Dict[Any, float]:
        if not self.reconstructed_graph:
            self.reconstructed_graph = graph.copy()
            for (u, v, link_strength) in self.link_predictor(graph):
                if link_strength >= self.threshold:
                    self.reconstructed_graph.add_edge(u, v)
        return nx.betweenness_centrality(self.reconstructed_graph)
    
    def update_model(self, removed_node):
        self.reconstructed_graph.remove_node(removed_node)

    def reset_model(self):
        self.reconstructed_graph = None