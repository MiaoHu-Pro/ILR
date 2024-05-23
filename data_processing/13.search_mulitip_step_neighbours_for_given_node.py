import time

import networkx as nx
from collections import deque

import numpy as np


class GraphAnalyzer:
    def __init__(self, graph_data):
        """
        Initialize the GraphAnalyzer with the provided graph data.
        """
        self.G = nx.DiGraph()
        self._create_graph(graph_data)
        self.G_undirected = self.G.to_undirected()

    def _create_graph(self, graph_data):
        """
        Create a graph from the provided data.
        """
        for edge in graph_data:
            self.G.add_edge(edge[0], edge[2], label=edge[1])

    def find_neighbors_by_relationship(self, start_node, relationship):
        """
        Find all multi-step neighbors for a given node in the graph (undirected),
        connected through a specific relationship.
        Returns a set of neighbors connected through the specified relationship type.
        """
        visited = set()  # Keep track of visited nodes
        queue = deque([start_node])  # Queue for BFS

        neighbors = set()  # Store neighbors connected through the specified relationship

        while queue:
            current_node = queue.popleft()

            if current_node not in visited:
                visited.add(current_node)

                # Check each neighbor of the current node
                for neighbor in self.G_undirected.neighbors(current_node):
                    if neighbor not in visited:
                        # Check if the edge has the specified relationship
                        if self.G_undirected.get_edge_data(current_node, neighbor)['label'] == relationship:
                            neighbors.add(neighbor)
                            queue.append(neighbor)

        return neighbors

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
    # Example usage
    # graph_example = [
    #     ['A', 'B', 'workflow'],
    #     ['B', 'C', 'workflow'],
    #     ['B', 'D', 'related to'],
    #     ['A', 'D', 'workflow'],
    #     ['A', 'E', 'workflow'],
    #     ['A', 'F', 'duplicate'],
    #     ['F', 'G', 'duplicate'],
    #     ['E', 'G', 'workflow'],
    #     ['G', 'H', 'duplicate'],
    #     ['H', 'I', 'duplicate'],
    #     ['H', 'M', 'related to']
    # ]
    t0= time.time()
    total_example = read_files("../data/Apache/total_triples_with_created_time.txt")
    analyzer = GraphAnalyzer(total_example)
    t1 = time.time()


    neighbors_workflow_c = analyzer.find_neighbors_by_relationship('HBASE-25304', 'general relation')

    t2 = time.time()
    print("time: ", t1 - t0)
    print("time: ", t2 - t1)


    print(neighbors_workflow_c)


