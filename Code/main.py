from Data.Graphs import empirical_graphs
from Utils.Reconstruction import BaselineReconstructor
from Experiments.General import *
from Experiments.LinkPredictionSearch import big_search, plot_cumulative_advantage
from Analytic.AnalyticExperiments import *

if __name__ == "__main__":
    #Checks to do:
    #v Baseline returns sensible values.
    #v Big Search returns sensible values on at least one graph.
    #- Check that at least one deep dive works.
    #- ER analytic baseline is roughly correct.
    #- CM analytic baseline is roughly correct.
    #- CM link predictor is calibrated.
    #- CM has no threshold that improves over baseline.

    print("Select which experiment you want to replicate using a numeric code: ")
    print("1: Obtain baseline")
    print("2: Search over link predictors")
    print("3: Investigation of a successful link predictor")
    print("4: Calculate analytic ER baseline")
    print("5: Calculate analytic config model baseline")
    print("6: Check optimal config model link predictor calibration")
    print("7: Check that no threshold improves link prediction on config model")
    code = int(input("Enter your number: "))
    print('\n\n')

    match code:
        case 1: 
            print("Running baseline experiment:")
            for name, graph in empirical_graphs():
                corr, robustness = run_main_experiment(graph, BaselineReconstructor(), 0.25, num_repetitions=100)
                tna = top_node_accuracy(BaselineReconstructor(), graph)
                print(f"On {name}: kendall-tau={corr:.2f}; robustness={robustness:.2f}; top node acc={tna:.2f}")

        case 2:
            print("Running link predictor search:")
            big_search()  

        case 3:
            from LinkPredictors.LocalSimilarity import *
            from LinkPredictors.GlobalSimilarity import *
            from LinkPredictors.QuasiLocalSimilarity import *
            from Data.Graphs import *
            link_predictor = eval(input("Enter link predictor function as a Python expression (e.g. 'random_walk_restarts'): "))
            threshold = float(input("Enter threshold (e.g. 0.1457)): "))
            graph = eval(input("Enter graph function as a Python expression (e.g. load_montagna_graph): "))
            print(f"Running investigation into {link_predictor.__name__} on {graph.__name__} at t = {threshold}")
            plot_cumulative_advantage(graph(), link_predictor, threshold)

        case 4:
            print("Calculating analytic ER baseline and giving a Monte-Carlo estimate to check against:")
            er_baseline_reality_check()
        case 5:
            print("Calculating analytic config model baseline and giving a Monte-Carlo estimate to check against:")
            cm_baseline_reality_check()

        case 6:
            print("Producing config model calibration plot.")
            cm_calibration_proof()

        case 7:
            print("Testing if a threshold exists s.t. link prediction improves over baseline in config model.")
            print("Warning: This can take a very long time.")
            cm_try_all_thresholds()

        case _:
            print("That is not a valid experiment code.")



