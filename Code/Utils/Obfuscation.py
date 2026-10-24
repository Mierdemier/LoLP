import networkx as nx
import numpy as np

class ObfuscationAlgorithm:
    def obscure(self, graph: nx.Graph, fraction_removed: float) -> nx.Graph:
        raise NotImplementedError("Subclasses should implement a method that removes roughly fraction_removed of the edges and returns the new graph. Do not modify the original in place.")
    

class RandomEdgeObfuscation(ObfuscationAlgorithm):
    def obscure(self, graph: nx.Graph, fraction_removed: float) -> nx.Graph:
        num_to_remove = int(graph.number_of_edges() * fraction_removed)
        edges_to_remove = np.random.choice(np.fromiter(graph.edges(), object), size=num_to_remove, replace=False)
        return_val = graph.copy()
        return_val.remove_edges_from(edges_to_remove)
        return return_val