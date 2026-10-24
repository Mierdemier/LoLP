#This file contains the search procedure for link predictors.
import networkx as nx
import numpy as np
import matplotlib.pyplot as plt
import math
from typing import Callable, Tuple
from sklearn.metrics import roc_curve, roc_auc_score
from scipy.stats import binom

from Data.Graphs import *
from LinkPredictors.LocalSimilarity import *
from LinkPredictors.GlobalSimilarity import *
from LinkPredictors.QuasiLocalSimilarity import *
from Utils.Evaluation import disruption_robustness, disruption_accuracy
from Utils.Reconstruction import LinkPredictionReconstructor, BaselineReconstructor
from Utils.Obfuscation import RandomEdgeObfuscation
from Experiments.General import *

#Search/Evaluation functions.
def roc_auc_investigation(link_predictor: Callable, graph: nx.Graph, num_iterations: int) -> Tuple[np.ndarray, np.ndarray, np.ndarray]:
    """
    Investigates the kind of scores a link predictor returns and its ROC AUC.
    Returns:
        ROC AUC Score
        Example thresholds (useful for determining the general range of scores a predictor can return)
    """
    auc = 0
    thresholds = None
    for i in range(num_iterations):
        subgraph = RandomEdgeObfuscation().obscure(graph, 0.25)
        scores = []; true_labels = []
        for (u, v, score) in link_predictor(subgraph):
            scores.append(score)
            true_labels.append(1 if graph.has_edge(u, v) else 0)
        if i == num_iterations - 1: 
            _, _, thresholds = roc_curve(true_labels, scores)
        auc += roc_auc_score(true_labels, scores) / num_iterations
    print("ROC-AUC-Score: ", auc)
    return auc, thresholds

def find_best_threshold(link_predictor: Callable, graph: nx.Graph, lower_bound: float, upper_bound: float, log_scale: bool = False) -> float:
    """
    Approximates the optimal threshold through (log-)linear search.
    Optimal is defined as resulting in the highest Kendall-Tau correlation in the main experiment.
    Plots and prints the search results. Returns optimal threshold.
    """
    if log_scale:
        lower_bound = np.log10(lower_bound); upper_bound = np.log10(upper_bound)
        search_range = 10 ** np.linspace(lower_bound, upper_bound, 100)
    else:
        search_range = np.linspace(lower_bound, upper_bound, 100)

    corrs = []
    for threshold in search_range:
        reconstructor = LinkPredictionReconstructor(link_predictor, threshold)
        corrs.append(
            run_main_experiment(graph, reconstructor, 0.25, num_repetitions=10, skip_robustness=True)
        )
    best_corr = max(corrs)
    best_threshold = search_range[corrs.index(best_corr)]
    print(f"Best correlation is {best_corr:.2f} using threshold {best_threshold}") #threshold not rounded bc we don't know how small it could be.
    #plt.plot(search_range, corrs)
    #plt.title(f"{link_predictor.__name__} Effectiveness")
    #plt.ylabel("Kendall Correlation w/ original betweennesss")
    #plt.xlabel("Threshold")
    #if log_scale:
    #    plt.xscale('log', base=10)
    #plt.show()
    return best_threshold

def disruption_win_prob(link_predictor: Callable, threshold: float, graph: nx.Graph, num_repetitions: int = 300) -> float:
    """
    Empirically finds P(link_predictor better at disruption than baseline | not a tie)
    Useful for investigating whether Kendall-Tau gains actually translate into better disruption.
    """
    num_baseline_wins = 0; ties = 0
    for _ in range(num_repetitions):
        visible_graph = RandomEdgeObfuscation().obscure(graph, 0.25)
        lp_reconstructor = LinkPredictionReconstructor(link_predictor, threshold)
        lp_robustness = disruption_robustness(graph, visible_graph, lp_reconstructor)
        bl_robustness = disruption_robustness(graph, visible_graph, BaselineReconstructor())
        if lp_robustness == bl_robustness:
            ties += 1
        elif lp_robustness > bl_robustness:
            num_baseline_wins += 1
    p_value = binom.cdf(num_baseline_wins, num_repetitions - ties, 0.5)

    if num_repetitions == ties:
        print(f"Link prediction method was exactly identical to baseline!!!")
        return 0.50 #Counts as neither winning nor losing.

    print(f"Link Predictor won {num_repetitions - ties - num_baseline_wins}/{num_repetitions - ties} times ({(num_repetitions - ties - num_baseline_wins) / (num_repetitions - ties) * 100:.2f}%)")
    print("P-value that it is better than baseline: ", p_value)
    return (num_repetitions - ties - num_baseline_wins) / (num_repetitions - ties)

def evaluate_link_predictor(graph: nx.Graph, link_predictor: Callable) -> float:
    """
    Quick check to see if a link predictor seems viable.
    Returns Kendall-Tau correlation. Prints and plots results.
    Quick search overfits to montagna graph as a deliberate price paid for time efficiency.
    Use it to find promising candidates and then do a more thorough evaluation.
    """
    #Step 1. Measure ROC-AUC Score for analysis purposes.
    print(f"Evaluating {link_predictor.__name__}:")
    auc, thresholds = roc_auc_investigation(link_predictor, graph, 100)

    #Step 2: Systematic search for best threshold.
    thresholds = [t for t in thresholds if math.isfinite(t) and not math.isnan(t)]
    thresholds.sort()
    lower_bound = thresholds[0]; upper_bound = thresholds[-1]
    if lower_bound > 0:
        lowest_nonzero = lower_bound
    elif thresholds[-1] > 0:
        lowest_nonzero = next(i for i,t in enumerate(thresholds) if t > 0)
    else: 
        lowest_nonzero = upper_bound #NSP never returns > 0. Don't use log-scale for NSP.
    if upper_bound // lowest_nonzero >= 1000:
        #If spans at least three orders of magnitude, use a log scale.
        print(f"Using log scale for {link_predictor.__name__}.")
        log_scale = True
        lower_bound = lowest_nonzero
    else: log_scale = False
    best_threshold = find_best_threshold(link_predictor, graph, lower_bound, upper_bound, log_scale)

    #Step 3. Take new independent measurement to avoid selection bias on the threshold.
    reconstructor = LinkPredictionReconstructor(link_predictor, best_threshold)
    corr = run_main_experiment(graph, reconstructor, 0.25, num_repetitions=100, skip_robustness=True)
    print(f"Final Kendall-Tau: {corr:.2f}")

    #Step 4. Measure Top Node Accuracy to see if it results in an accurate disruption.
    tna = top_node_accuracy(LinkPredictionReconstructor(link_predictor, best_threshold), graph)
    print(f"{link_predictor.__name__} got the exact correct node {100.0 * tna:.2f}% of the time.")

    #Step 5. Take robustness measurement to see if it's actually better at disruption.
    win_prob = disruption_win_prob(link_predictor, best_threshold, graph)
    print()

    return auc, corr, tna, win_prob



#Search==========================================================================================
TO_SEARCH = [
    #Local Similarity indices.
    nx.adamic_adar_index,
    lambda g: nx.common_neighbor_centrality(g, alpha=1),
    nx.jaccard_coefficient,
    nx.preferential_attachment,
    nx.resource_allocation_index,
    salton_index,
    sorenson_index,
    hub_promoted_index,
    hub_depressed_index,
    lhn_index,
    CAR_common_neighbours,
    CAR_adamic_adar,
    CAR_resource_allocation,
    CAR_preferential_attachment,
    individual_attraction,
    bayesian_common_neighbours,
    bayesian_adamic_adar,
    bayesian_resource_allocation,
    functional_similarity_weight,
    local_interacting_score,
    clustering_coefficient_index,
    link_clustering_index,
    
    #Global Similarity Indices.
    #P(win) for these can be very dependent on the precise threshold chosen by random search
    negative_shortest_path_lengths,
    katz_index,
    cosine_laplacian,
    average_commute_time,
    normalised_average_commute_time,
    matrix_forest_index,
    global_lhn,
    random_walk_restarts,
    max_entropy_random_walk,
    flow_propagation,
    blondel_index,
    
    #Quasi-local similarity methods.
    local_path_index,
    L3_index,
    local_random_walk,
    superposed_random_walk,
    friend_link,
    propflow,   
]

def big_search():
    for name, graph in empirical_graphs():
        print("USING GRAPH: ", name)
        aucs = []; corrs = []; tnas = []; win_probs = []
        for link_predictor in TO_SEARCH:
            auc, corr, tna, win_prob = evaluate_link_predictor(graph, link_predictor)
            aucs.append(auc); corrs.append(corr); tnas.append(tna); win_probs.append(win_prob)
        print("On ", name)
        print("AUCs:")
        print('\n'.join(map(str, aucs)))
        print()
        print("Corrs:")
        print('\n'.join(map(str, corrs)))
        print()
        print("TNAs:")
        print('\n'.join(map(str, tnas)))
        print()
        print("Win Probs:")
        print('\n'.join(map(str, win_probs)))
        print('\n\n\n')

#For the qualitative investigations into succesful link predictors:
def plot_cumulative_advantage(graph: nx.Graph, link_predictor: Callable, threshold: float, num_repetitions: int = 1000):
    baseline_lccs = np.zeros((num_repetitions, graph.number_of_nodes()))
    lp_lccs = np.zeros_like(baseline_lccs)

    for n in range(num_repetitions):
        visible_graph = RandomEdgeObfuscation().obscure(graph, 0.25)
        baseline_lccs[n] = get_raw_lcc_sizes(graph, visible_graph, BaselineReconstructor())
        lp_reconstructor = LinkPredictionReconstructor(link_predictor, threshold)
        lp_lccs[n] = get_raw_lcc_sizes(graph, visible_graph, lp_reconstructor)

    avg_advantage = np.cumsum(baseline_lccs.mean(axis=0) - lp_lccs.mean(axis=0))
    advantage_distribution = baseline_lccs.sum(axis=1) - lp_lccs.sum(axis=1)
    #Convert to units of robustness:
    nodes_squared = graph.number_of_nodes() * graph.number_of_nodes()
    avg_advantage /= nodes_squared; advantage_distribution /= nodes_squared

    plt.plot(avg_advantage)
    plt.xlabel("Nodes Removed")
    plt.ylabel("Avg cumulative advantage of link predictor")
    print("Final advantage (avg):", avg_advantage[-1])
    plt.show()

    plt.hist(advantage_distribution, log=False)
    plt.title("Histogram of final RWR advantage")
    plt.show()

    print("Avg Advantage: ")
    print('\n'.join(map(str, avg_advantage)))
    print('\n\n\n')
    print("Advantage distribution: ")
    print('\n'.join(map(str, advantage_distribution)))