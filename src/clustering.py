"""Container for clustering"""

from src.utils import Paths
import networkx as nx


class Clustering:
    """
    Should take an input graph, ground truth clusters, method. Generate clustering and report NMI, modularity scores
    """
    _implemented_methods_ = ['graph-walk_dis', 'graph-walk_cont', 'random-walk_dis', 'random-walk_cont', 'diffusion']

    def __init__(self, paths: Paths, input_g: nx.Graph, method: str, order: str):
        self.paths: Paths = paths
        self.input_g: nx.Graph = input_g
        self.method: str = method  # name of the method
        self.order: str = order  # either first or second
        return

