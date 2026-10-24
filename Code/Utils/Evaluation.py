import numpy as np
from scipy.stats import kendalltau
import matplotlib.pyplot as plt
import networkx as nx
from typing import Dict, Any, List

from Utils.Reconstruction import Reconstructor

#Kendall correlation between centrality rankings.
def kendall_correlation(dict1: Dict[Any, float], dict2: Dict[Any, float]) -> float:
    #Rank all nodes
    nodes = list(dict1.keys())
    if set(nodes) != set(dict2.keys()):
        raise ValueError("Both dictionaries must have the same keys.")
    
    vals1 = [dict1[node] for node in nodes]
    vals2 = [dict2[node] for node in nodes]

    #Note: kendalltau() already converts from vals to ranks.
    #We just have to make sure they correspond to the same node.
    stat = kendalltau(vals1, vals2).statistic
    if np.isnan(stat):
        return 0.0 # Counts as not correlating.
    return stat


def disruption_robustness(original_graph: nx.Graph, visible_graph: nx.Graph, reconstructor: Reconstructor, plot: bool = False) -> float:
    """
    Attempts to disrupt a graph using reconstructed betweenness and returns Robustness.
    """
    original_graph = original_graph.copy(); visible_graph = visible_graph.copy()
    n_nodes = original_graph.number_of_nodes()
    lcc_sizes = []
    for _ in range(n_nodes):
        #Get most central node according to the reconstructor.
        reconstructed_betweenness = reconstructor.reconstruct_betweenness(visible_graph)
        most_central_node = max(reconstructed_betweenness, key=reconstructed_betweenness.get)
        #Remove the most central node from both the ground-truth graph and the visible subgraph.
        original_graph.remove_node(most_central_node)
        visible_graph.remove_node(most_central_node)
        reconstructor.update_model(removed_node=most_central_node)
        #Measure ground-truth LCC size.
        if original_graph.number_of_nodes() > 0:
            largest_component_size = max([len(c) for c in nx.connected_components(original_graph)])
        else:
            largest_component_size = 0
        lcc_sizes.append(largest_component_size)
    robustness = sum(lcc_sizes) / (n_nodes**2)
    
    if plot:
        print(f"Robustness is {robustness} (AUC is {sum(lcc_sizes)})")
        plt.plot(range(n_nodes), lcc_sizes)
        plt.title("Disruption")
        plt.xlabel("num nodes removed")
        plt.ylabel("LCC size")
        plt.show()

    return robustness #Lower robustness => *better* disruption.

#Lots of code duplication going on here with the disruption process.
#If you ever need to change it, rewrite it to put disruption in a common place somewhere to
#prevent mistakes where you update one function but not all others.
def get_raw_lcc_sizes(original_graph: nx.Graph, visible_graph: nx.Graph, reconstructor: Reconstructor) -> List[float]:
    original_graph = original_graph.copy(); visible_graph = visible_graph.copy()
    n_nodes = original_graph.number_of_nodes()
    lcc_sizes = []
    for _ in range(n_nodes):
        #Get most central node according to the reconstructor.
        reconstructed_betweenness = reconstructor.reconstruct_betweenness(visible_graph)
        most_central_node = max(reconstructed_betweenness, key=reconstructed_betweenness.get)
        #Remove the most central node from both the ground-truth graph and the visible subgraph.
        original_graph.remove_node(most_central_node)
        visible_graph.remove_node(most_central_node)
        reconstructor.update_model(removed_node=most_central_node)
        #Measure ground-truth LCC size.
        if original_graph.number_of_nodes() > 0:
            largest_component_size = max([len(c) for c in nx.connected_components(original_graph)])
        else:
            largest_component_size = 0
        lcc_sizes.append(largest_component_size)
    return lcc_sizes

def disruption_accuracy(original_graph: nx.Graph, visible_graph: nx.Graph, reconstructor: Reconstructor, limit: int = -1) -> float:
    """
    Measures how often the reconstructor picks out the correct most central node over the course of a single disruption.
    """
    original_graph = original_graph.copy(); visible_graph = visible_graph.copy()
    n_nodes = original_graph.number_of_nodes()
    if limit < 0: limit = n_nodes
    n_correct = 0
    for _ in range(limit):
        #Get most central node according to the reconstructor.
        reconstructed_betweenness = reconstructor.reconstruct_betweenness(visible_graph)
        most_central_node = max(reconstructed_betweenness, key=reconstructed_betweenness.get)
        #Get true most central node.
        betweenness = nx.betweenness_centrality(original_graph)
        true_most_central_node = max(betweenness, key=betweenness.get)
                                                          #Getting a node of exactly equal betweenness counts.
        if most_central_node == true_most_central_node or betweenness[true_most_central_node] == betweenness[most_central_node]:
            n_correct += 1
        #Remove the most central node from both the ground-truth graph and the visible subgraph.
        original_graph.remove_node(most_central_node)
        visible_graph.remove_node(most_central_node)
        reconstructor.update_model(removed_node=most_central_node)
    return n_correct / limit