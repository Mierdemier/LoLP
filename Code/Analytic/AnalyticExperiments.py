from Analytic.AnalyticBaselines import er_baseline, cm_baseline
from Analytic.AnalyticGraphs import load_er_graph, load_scalefree_graph
from Analytic.AnalyticReconstruction import *

from Utils.Obfuscation import RandomEdgeObfuscation
from Utils.Evaluation import kendall_correlation

import networkx as nx
from typing import Callable
from sklearn.calibration import calibration_curve
import matplotlib.pyplot as plt

#Calculates baseline kendall-tau-b correlation of the degree centrality.
#Useful for checking analytic computations.
def monte_carlo_kendall(graph_function: Callable, reconstructor: DegreeReconstructor, fraction_edges_removed: float = 0.25, num_repetitions: int = 1000) -> float:
    kendall_corr = 0
    for n in range(num_repetitions):
        graph = graph_function()
        reconstructor.reset_model()

        #Note: Analytic calculation use degree, not betweenness.
        #So original run_main_experiment function is NOT applicable!
        original_degree = nx.degree_centrality(graph)
        visible_graph = RandomEdgeObfuscation().obscure(graph, fraction_edges_removed)
        reconstructed_degree = reconstructor.reconstruct_degree(visible_graph)
        kendall_corr += kendall_correlation(original_degree, reconstructed_degree)
    reconstructor.reset_model()
    return kendall_corr / num_repetitions


#Used to check that analytic baseline for the ER network is approximately equal to mean observed baseline.
#(It won't be exactly equal due to sample vs. population kendall-tau being slightly different)
def er_baseline_reality_check(num_repetitions: int = 1000):
    expected_kendall_corr = er_baseline()
    print(f"Expecting a kendall-tau-b correlation of {expected_kendall_corr}")

    kendall_corr = monte_carlo_kendall(load_er_graph, BaselineDegreeReconstructor(), num_repetitions=num_repetitions)
    print(f"Actual average kendall-tau-b is {kendall_corr}")

#Used to check that analytic baseline for the Configuration Model network is approximately equal to mean observed baseline.
#(It won't be exactly equal due to sample vs. population kendall-tau being slightly different)
def cm_baseline_reality_check(num_repetitions: int = 1000):
    expected_kendall_corr = cm_baseline()
    print(f"Expecting a kendall-tau-b correlation of {expected_kendall_corr}")
    
    kendall_corr = monte_carlo_kendall(load_scalefree_graph, BaselineDegreeReconstructor(), num_repetitions=num_repetitions)
    print(f"Actual average kendall-tau-b is {kendall_corr}")

#Checks that CMOptimalLinkPrediction is indeed optimal by comparing predicted probability
#to empirically observed frequency of edge.
def cm_calibration_proof(num_edges_tested: int = 100_000):
    probs = []
    edge_exists = []
    for _ in range(num_edges_tested):
        graph = load_scalefree_graph()
        graph_vis = RandomEdgeObfuscation().obscure(graph, 0.25)
        for u, v, score in CMOptimalLinkPrediction().link_predictor(graph_vis):
            probs.append(score)
            edge_exists.append(1 if graph.has_edge(u, v) else 0)
            break #One edge per graph for maximally diverse observations.
    
    prob_true, prob_pred = calibration_curve(
        edge_exists, probs, n_bins=10, strategy='quantile'
    )
    plt.plot(prob_pred, prob_true, "o-")
    plt.plot([0,0.1], [0,0.1], "--")
    plt.xlabel("Predicted probability")
    plt.ylabel("Observed frequency")
    plt.show()

#Checks that indeed no threshold exists that allows link prediction to be significantly better
#than doing nothing on CM graph.
#(useful bc the corresponding theoretical argument is intuitive, not rigorously mathematical)
def cm_try_all_thresholds(num_thresholds: int = 100, num_repetitions_per_threshold: int = 100):
    best_corr = -1; best_t = -1
    for t in np.linspace(0, 1, num_thresholds):
        corr = monte_carlo_kendall(load_scalefree_graph, CMOptimalLinkPrediction(threshold=t), num_repetitions=num_repetitions_per_threshold)
        if corr > best_corr:
            print(f"Best so far: {t} with {corr}")
            best_corr = corr; best_t = t

    print(f"Best threshold is {best_t}. Running final experiment to eliminate selection bias.")
    corr = monte_carlo_kendall(load_scalefree_graph, CMOptimalLinkPrediction(threshold=best_t), num_repetitions=num_repetitions_per_threshold)
    print(f"Corresponding Kendall-Tau is {corr}.")
