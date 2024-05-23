
import networkx as nx
import time
import numpy as np

class GraphAnalysis:
    def __init__(self, graph_data):
        """
        Initialize the GraphAnalysis with the provided graph data.
        """
        self.graph = nx.Graph()
        self._create_graph(graph_data)

    def _create_graph(self, graph_data):
        """
        Create an undirected graph from the provided data.
        """
        for edge in graph_data:
            self.graph.add_edge(edge[0], edge[2])

    def count_connected_components(self):
        """
        Count the total number of connected components in the graph.
        """
        return nx.number_connected_components(self.graph)

    def count_small_subgraphs(self, max_nodes):
        """
        Count the number of sub-graphs with no more than 'max_nodes' nodes.
        """
        count = 0
        for component in nx.connected_components(self.graph):
            if len(component) <= max_nodes:
                count += 1
        return count

# # Example usage
# graph_data = [
#     ['B', 'C', 'workflow'],
#     ['B', 'D', 'related to'],
#     ['A', 'E', 'workflow'],
#     ['A', 'F', 'duplicate'],
#     ['F', 'G', 'duplicate'],
#     ['E', 'G', 'workflow'],
#     ['H', 'I', 'duplicate'],
#     ['H', 'M', 'related to']
# ]
#
# analyzer = GraphAnalysis(graph_data)
# total_components = analyzer.count_connected_components()
# small_subgraphs = analyzer.count_small_subgraphs(3)
#
# print("Total number of connected components:", total_components)
# print("Number of sub-graphs with no more than 3 nodes:", small_subgraphs)

def read_files(rank_path):
    f = open(rank_path)
    f.readline()

    x_obj = []
    for d in f:
        d = d.strip()
        if d:
            d = d.split('\t')

            elements = []
            for n in d:
                elements.append(n.strip())
            d = elements
            x_obj.append(d)
    f.close()

    return np.array(x_obj)

if __name__ == '__main__':

    t0= time.time()
    total_example = read_files("../data/Jira/total_triples_with_created_time.txt")
    analyzer = GraphAnalysis(total_example)
    t1 = time.time()

    total_components = analyzer.count_connected_components()
    small_subgraphs = analyzer.count_small_subgraphs(3)
    #
    print("Total number of connected components:", total_components)
    print("Number of sub-graphs with no more than 3 nodes:", small_subgraphs)
    print("rate :", small_subgraphs/total_components)

    t2 = time.time()



# Apache
# Total number of connected components: 59136
# Number of sub-graphs with no more than 3 nodes: 49222
# rate : 0.8323525432900433

# QT
# Total number of connected components: 8391
# Number of sub-graphs with no more than 3 nodes: 6991
# rate : 0.8331545703730188

# RedHat
# Total number of connected components: 30152
# Number of sub-graphs with no more than 3 nodes: 24909
# rate : 0.8261143539400372

# MongoDB
# Total number of connected components: 10676
# Number of sub-graphs with no more than 3 nodes: 8438
# rate : 0.7903709254402398

# Jira
# Total number of connected components: 40631
# Number of sub-graphs with no more than 3 nodes: 35440
# rate : 0.8722404075705742

# Mojang
# Total number of connected components: 16629
# Number of sub-graphs with no more than 3 nodes: 9963
# rate : 0.5991340429370378
