import networkx as nx
from typing import Iterator, Tuple
import pandas as pd
from itertools import product

class GraphFormat:
    def __init__(self, 
                    weighted: bool = False, 
                    separator: str = '', 
                    has_header: bool = False,
                    node1_col: int = 0,
                    node2_col: int = 1,
                    weight_col: int = 2
                 ):
        self.weighted = weighted
        self.separator = separator
        self.has_header = has_header
        self.node1_col = node1_col
        self.node2_col = node2_col
        self.weight_col = weight_col
    
    def load_graph(self, file_path: str) -> nx.Graph:
        G = nx.Graph()
        with open(file_path, 'r') as f:
            if self.has_header:
                next(f)  # Skip header
            for line in f:
                r = line.strip().split(self.separator) if self.separator else line.strip().split()
                node1, node2 = int(r[self.node1_col]), int(r[self.node2_col])
                if self.weighted:
                    weight = float(r[self.weight_col])
                    G.add_edge(node1, node2, weight=weight)
                else:
                    G.add_edge(node1, node2)
                    
        G = nx.convert_node_labels_to_integers(
            G,
            first_label=0,
            ordering="sorted"
        )
        return G

#Specific graphs.
def load_iceland_graph() -> nx.Graph:
    return GraphFormat(has_header=True).load_graph("Data/out.iceland")

def load_highschool_graph() -> nx.Graph:
    #Note: *is* weighted, but we ignore the weights in our analysis.
    return GraphFormat(has_header=True).load_graph("Data/out.moreno_highschool_highschool")

def load_physicians_graph() -> nx.Graph:
    return GraphFormat(has_header=True).load_graph("Data/out.moreno_innovation_innovation")

def load_9_11_graph() -> nx.Graph:
    return GraphFormat(separator=',').load_graph("Data/9_11 Edgelist.csv")

def load_caviar_graph() -> nx.Graph:
    #Note: *is* weighted, but we ignore the weights in our analysis.
    return GraphFormat(separator=',').load_graph("Data/Caviar Edgelist.csv")

def load_montagna_graph() -> nx.Graph:
    return GraphFormat().load_graph("Data/Montagna_meetings_edgelist.csv")

def load_ndrangheta_graph() -> nx.Graph:
    return GraphFormat(separator=',').load_graph("Data/Ndrangheta Edgelist.csv")

def empirical_graphs() -> Iterator[Tuple[str, nx.Graph]]:
    #Implemented as a generator to avoid loading all graphs into memory at once.
    #Small social
    yield "Iceland", load_iceland_graph()
    yield "High School", load_highschool_graph()
    yield "Physicians", load_physicians_graph()

    #Criminal.
    yield "9/11 Hijackers", load_9_11_graph()
    yield "Operation Caviar", load_caviar_graph()

    #Mafia.
    yield "Montagna meetings", load_montagna_graph()
    yield "Ndrangheta", load_ndrangheta_graph()

if __name__ == "__main__":
    #Print out all of the networks and their characteristics.
    table = pd.DataFrame(columns=["Name", "#Nodes", "#Edges", "Clustering", "Average Distance", "Density", "Diameter"])
    for i, (name, graph) in enumerate(empirical_graphs()):
        N = graph.number_of_nodes()
        E = graph.number_of_edges()
        clustering = nx.average_clustering(graph)
        density = nx.density(graph)

        avg_dist = 0; num_paths = 0
        for node1, node2 in product(graph.nodes, graph.nodes):
            if node2 <= node1: continue #Assumes nodes are integers, which is true for our data.
            if not nx.has_path(graph, node1, node2): continue
            avg_dist += nx.shortest_path_length(graph, node1, node2)
            num_paths += 1
        avg_dist /= num_paths

        diameter = 0
        for component in nx.connected_components(graph):
            subgraph = graph.subgraph(component)
            diameter = max(diameter, nx.diameter(subgraph))
        
        table.loc[i] = [name, N, E, clustering, avg_dist, density, diameter]
    
    print(table)