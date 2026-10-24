from Utils.Obfuscation import RandomEdgeObfuscation
from Utils.Reconstruction import *
from Utils.Evaluation import *

import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
from typing import List, Callable

def run_main_experiment(
        graph: nx.Graph, reconstructor: Reconstructor, fraction_removed: float, 
        num_repetitions: int = 1, skip_robustness: bool = False) -> float:
    """
    Runs the betweenness reconstruction experiment and evaluates using both correlation and disruption robustness.

    Args:
        graph (nx.Graph): Complete network to do reconstruction on.
        reconstructor (Reconstructor): Algorithm for reconstructing betweenness centrality.
        fraction_removed (float): fraction of edges that are removed.
        num_repetitions (int): number of times to repeat experiment (result is averaged).
        skip_robustness (bool): If true, don't compute disruption robustness. Use for efficiency when you only care about correlation. Optionala, default = False.

    Returns:
        if skip_robustness:
            float: Kendall tau correlation between reconstruction and true betweenness.
        else:
            Tuple[float, float]: Kendall tau correlation, disruption robustness.

    """
    original_betweenness = nx.betweenness_centrality(graph)

    corr = 0; robustness = 0
    for _ in range(num_repetitions):
        visible_graph = RandomEdgeObfuscation().obscure(graph, fraction_removed)
        reconstructor.reset_model()
        reconstructed_betweenness = reconstructor.reconstruct_betweenness(visible_graph)

        corr += kendall_correlation(original_betweenness, reconstructed_betweenness) / num_repetitions
        if not skip_robustness:
            robustness += disruption_robustness(graph, visible_graph, reconstructor) / num_repetitions
    reconstructor.reset_model() #Give back  clean reconstructor in case caller want to use it again.
    return corr if skip_robustness else (corr, robustness)

def top_node_accuracy(reconstructor: Reconstructor, graph: nx.Graph, num_repetitions: int = 300):
    """
    Measures how often it gets the exact top node correct.
    Useful because this might correlate better w/ disruption performance than overall Kendall-Tau.
    """
    accuracy = 0
    for _ in range(num_repetitions):
        visible_graph = RandomEdgeObfuscation().obscure(graph, 0.25)
        reconstructor.reset_model()
        accuracy += disruption_accuracy(graph, visible_graph, reconstructor, limit=-1) / num_repetitions
    reconstructor.reset_model()
    return accuracy