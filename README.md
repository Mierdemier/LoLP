# Limits of Link Prediction in Disrupting Incomplete Criminal Networks

This GitHub repository contains the code for the Master's thesis: Limits of Link Prediction in Disrupting Incomplete Criminal Networks. See the thesis for an explanation of what this code does and which results it uncovers. All scripts can be found in the ```Code``` folder. Many of them take a very long time to run, so there is also a ```Results``` folder that has pre-computed results stored in a computer-friendly format.

The rest of this README file will explain the following. First, it has instructions on how to run the code as-is using ```main.py```. Next, it gives an overview of the ```Results``` folder, and how the values therein are formatted. Finally, there is an explanation of the ```Code``` folder for those who wish to understand and/or modify the code.

## How to Run the Code
The codebase is designed to make it easy to replicate the experiments discussed in the thesis. To run an experiment:
1. Ensure that the active directory is the ```Code``` folder.
2. Run ```python main.py```
3. Read the instructions given in the console and enter any information it asks for.

```main.py``` will ask the user to enter a numerical code corresponding to the experiment they wish to replicate. The print statement itself gives all the codes, but here they are listed again with more elaboration:
1. "Obtain baseline". This experiment empirically finds the baseline values of the Kendall-Tau, robustness, and top node accuracy on all graphs. Recall that the baseline refers to conditions where part of the network is hidden, but no link prediction is attempted. Read the Empirical Analysis chapter of the thesis for more information.

2. "Search over link predictors". This experiment uses all link predictors on all graphs, and reports the ROC-AUC score, Kendall Tau correlation, top node accuracy, and probability of exceeding baseline performance in terms of disruption effectiveness for each one. Read the Empirical Analysis chapter of the thesis for more details.

3. "Investigation of a successful link predictor". This refers to the special qualitative investigations defined in Section 5.2.4 of the thesis, where it is determined when and how often a link predictor gains an advantage over the baseline during disruption. After entering this code, the user is asked to give some additional information: the name (in the codebase!) of the link predictor and the graph they wish to use, as well as the desired threshold. To find the names of link predictors, look at the ```TO_SEARCH``` list in ```Experiments/LinkPredictionSearch.py```. To find the names of graphs, look at ```Code/Data/Graphs.py```.

4. "Calculate analytic ER baseline". This function calculates the baseline Kendall-Tau of the synthetic Erdos-Renyi (ER) graph using the analytic method described in the theoretical analysis chapter. It then computes it again using a simple Monte-Carlo estimate to provide supporting evidence that the analytic computation is correct (they should be *approximately* the same). See the Theoretical Analysis chapter, particularly Section 4.2 for more information.

5. "Calculate analytic config model baseline". This experiment is the same as #4, except it applies to the configuration model instead of the ER model. See the Theoretical Analysis chapter, particularly Section 4.3 for more information.

6. "Check optimal config mode link predictor calibration". This experiment shows that the optimal link predictor is indeed optimal by comparing predicted probability to the empirically observed frequency of edges. See Section 4.3 for more information; the experiment should reproduce the same information found in the figure of the calibration plot.

7. "Check that no threshold improves link prediction on config model". Finally, this experiment checks that indeed no threshold exists that allows link prediction to be significantly better than doing nothing on CM graph. Recall that this is useful, because the corresponding theoretical argument is intuitive, not rigorously mathematical. See the end of Section 4.3 for more information.

Using these codes, the user can replicate the experiments without requiring complete knowledge of the codebase. However, keep in mind that many of the experiments can run for very long. Every once in a while, they should keep the user updated by printing some intermediate information. Note also that all of the experiments are stochastic. This is partially addressed by repeating all stochastic observations many times, but the user's results may deviate by a small margin from the exact numbers reported in the thesis.

## How to Use Results

Since the experiments are computationally expensive, pre-computed results are already provided in the ```Results``` folder. This section explains the format of every file in that folder. Note that the full results are also presented in a human-readable format in Appendix B in the thesis. This folder is primarily intended to be easily read by automated data-processing scripts.

Inside ```Results```, there is one folder corresponding to each empirical network:
- ```Caviar``` for the Drugs Dealers.
- ```Hijackers``` for the 9/11 Hijackers.
- ```Montagna``` refers to the Mafia Clans.
- ```Ndragheta``` refers to *La Lombardia*.
- ```Iceland```, ```High School```, and ```Physicians``` use the same names as in the thesis.

Inside each folder, there are five text files containing newline separated floating point numbers (sometimes with asterisks after them, see below). Each value corresponds to one link predictor that was tested. See Section 5.2.4 in the thesis for more information on these evaluation metrics that were measured during the simulation. The files are:
1. ```KendallTau.txt``` has the Kendall-Tau correlations. There are no asterisks.
2. ```Pwin.txt``` has the win probabilities during disruption. Values ending in an asterisk are $P(\text{win} | \neg\text{tie})$ above 0.50, but not statistically significant (just like in the tables in Appendix B). This asterisk is meant to warn the user that it may look like the link predictor outperformed the baseline, but that this has not been statistically established.
3. ```ROCAUC.txt``` has the ROC-AUC scores. There are no asterisks.
4. ```TopNodeAccuracy.txt``` has the accuracy with which the correct highest-betweenness node was selected during disruption. There are no asterisks.
5. ```thresholds.txt``` contains the threshold on link prediction scores selected during the search. An asterisk after the value indicates the threshold was found in a log-spaced search (just like in the tables in Appendix B).

Each line consistently refers to the same link predictor. ```Results/names.txt``` shows which one. For example, line 5 in ```names.txt``` says "Resource Allocation", so in all files, the fifth line always refers to Resource Allocation. Scripts are encouraged to load both ```names``` and the desired evaluation metric into memory and combine them into a DataFrame.

The ```Results/DeepDives``` subfolder contains the results of the qualitative investigations into seemingly succesful link predictors. These investigations and the metrics obtained therein are described in more detail in Section 5.2.4. \
Each file corresponds to a single link predictor, which it is named after. Note that there are no link predictors that were considered succesful on multiple graphs, so the name of the link predictor already implies which graph it was measured on. \
The text files contain two lists of newline separated values, split by an empty line in-between. The first list consists of floating point numbers indicating the average cumulative advantage; the $i$ th line is $\textbf{Adv}(i)$. This can be used to reconstruct the line plots in the thesis. In the second list, each value is floating point number indicating one observed value for the final cumulative advantage $\textbf{Adv}(|V|)$. This can be used to reconstruct the histograms from the thesis.

## Organisation of the Code
This section explains how the code is organised. It is not necessary reading for users who just wish to replicate the existing experiments, but may be useful for those who wish to make modifications.

The ```Code``` folder contains the following items:
- ```main.py``` allows the user to select a experiment and then runs it. Generally, it refers to the other files for the actual implementation of the experiment. It is helpful to read ```main.py``` to learn how to call the experiment functions in the other files.
- ```Data``` contains the edge lists describing each empirical network.
    - ```Data/Graphs.py``` has the code for loading an edge list into a Python network object. When ```main.py``` asks for a name of a graph, it refers to the name of the loader function in this file, *e.g.* ```load_iceland_graph```. This file can also be run directly (```python Data/Graphs.py```) in which case it will present some information about each graph in a table.
    - The other files in the ```Data``` folder represent one edge list each. Examine the code in  ```Graphs.py``` to see how to read them.
- ```Experiments``` has code for running the empirical analysis.
    - ```Experiments/General.py``` contains experiments that refer to centrality reconstruction and robustness in general, without making specific reference to link prediction.
    - ```Experiments/LinkPredictionSearch.py``` contains experiments that are specific to the method of link prediction: the search over all link predictors as well as the qualitative investigations into specific ones. The ```TO_SEARCH``` variable is particularly useful to inspect, as it is a list of all link predictors used in the research. The user can refer to it when *e.g.* ```main.py``` asks for the name of a link predictor they wish to use.
- ```LinkPredictors``` implements all link predictors used in the thesis that are not already implemented in ```networkx```.
    - ```LinkPredictors/LocalSimilarity.py``` has local methods.
    - ```LinkPredictors/GlobalSimilarity.py``` has global methods.
    - ```LinkPredictors/QuasiLocalSimilarity.py``` has quasi-local methods.
- ```Utils``` has helper functions used by ```Experiments``` (other than link prediction methods, see ```LinkPredictors```).
    - ```Utils/Evaluation.py``` contains helper functions that compute evaluation metrics such as the Kendall Tau.
    - ```Utils/Obfuscation.py``` has code for hiding the edges in a graph.
    - ```Utils/Reconstruction.py``` contains code for reconstructing a network using link prediction scores. It also includes the trivial baseline "reconstructor" which does nothing.
- ```Analytic``` is a folder containing everything related to the Theoretical Analysis. It is separate from the rest of the code, because the Theoretical Analysis sometimes makes different assumptions than the Empirical one (*e.g.* degree instead of betweenness), so it is important to avoid unintended code re-use between the two analyses.
    - ```Analytic/AnalyticBaselines.py``` specifies the analytic calculation of baseline Kendall-Tau for both example networks.
    - ```AnalyticExperiments.py``` has code for all the empirical experiments/checks that are used the Theoretical Analysis chapter to double-check the accuracy of arguments and calculations. It uses the other files in this folder as helpers.
    - ```AnalyticGraphs.py``` has code for sampling networks from both the example distributions.
    - ```AnalyticReconstruction.py``` defines the optimal link predictors for both network models, and how to (try to) use them to reconstruct the degree.
    - ```ProbabilityFunctions.py``` has helper functions for sampling from and calculating the probability mass function of the random distributions used (except for distributions that generate entire graphs, see ```AnalyticGraphs.py```)

Further details can be found in the comments of the code.