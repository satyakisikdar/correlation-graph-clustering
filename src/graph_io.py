"""
File for general graph io and synthetic graph generation
"""
import subprocess
from typing import Tuple

import networkx as nx
from cdlib import NodeClustering
from cdlib.evaluation import normalized_mutual_information, adjusted_mutual_information
from matplotlib import pyplot as plt

from src.utils import remove_files, plot_network_clusters, Paths


class LFRGraph:
    """
    Container for LFR graphs - store both the graph object and the node community labels
    """

    def __init__(self, paths: Paths, n: int, mu: float, avg_k: int, max_k: int = 50):
        """
        n: number of nodes
        mu: mixing parameter
        avg_k: average degree
        """
        self.paths = paths
        self.n = n
        self.mu = round(mu, 2)
        self.avg_k = avg_k
        self.max_k = max_k
        self.name = f'LFR-{self.n // 1000}k-{max_k}-{avg_k}-{int(self.mu * 100)}'

        self.graph_basepath = self.paths.data_dir / 'LFR' / f'{self.n // 1000}k-{max_k}/{avg_k}'
        # self.input_g = None
        # self.pos = None  # store the node positions
        # self.ground_truth_clustering: NodeClustering = self.generate_ground_truth_clustering()
        return

    def plot_clustering(self, method: str = 'ground_truth', ix: int = -1, figsize: Tuple[int, int] = (25, 25),
                        node_size: int = 150, clustering=None):
        if self.pos is None:
            self.pos = nx.spring_layout(self.input_graph, iterations=100)

        ncols = 1
        if method == 'ground_truth':
            clustering = self.ground_truth_clustering
            ncols = 1
        elif clustering is None:
            clustering = self.generated_clusterings[method][ix]  # plot the latest by default
            ncols = 2

        modularity = clustering.newman_girvan_modularity().score
        num_clusters = len(clustering.communities)

        title = f'{self.name!r} \t |V| = {self.input_graph.order():,} \t |E| = {self.input_graph.size():,}\n' \
                f'method = {method!r} \t clusters = {num_clusters:,} \t modularity = {modularity:.3g}'

        if method != 'ground_truth':
            nmi = normalized_mutual_information(self.ground_truth_clustering, clustering).score
            ami = adjusted_mutual_information(self.ground_truth_clustering, clustering).score
            title += f'\t NMI = {nmi:.3g} \t AMI = {ami:.3g}'

        fig, *axs = plt.subplots(figsize=figsize, layout='constrained', ncols=ncols)

        plot_network_clusters(self.input_graph, figsize=figsize, node_size=node_size, cmap='tab20',
                              partition=clustering, position=self.pos, ax=axs[0], title=title)

        plt.show()
        return

    def __str__(self) -> str:
        ground_truth_cluster_count = len(self.ground_truth_clustering.communities)
        ground_truth_modularity = self.ground_truth_clustering.newman_girvan_modularity().score
        st = f'<{self.name!r} \t |V|={self.input_graph.order():,} \t |E| = {self.input_graph.size():,} \t ' \
             f'GT comms: {ground_truth_cluster_count:,} \t GT mod: {ground_truth_modularity:.4g}>'
        return st

    def _read_graph(self) -> nx.Graph:
        graph_path = self.paths.data_dir / 'LFR' / f'{self.name}_edgelist.txt'
        g = nx.read_edgelist(graph_path, nodetype=int)
        num_comp = nx.number_connected_components(g)
        if num_comp > 1:
            lcc_nodes = max(nx.connected_components(g), key=len)
            frac = len(lcc_nodes) / g.order() * 100
            print(f'{self.name!r} has {num_comp:,} components. LCC has {frac:.2g}% nodes')
        # assert num_comp == 1, f'{self.name!r} has {num_comp} components'
        assert min(g.nodes()) == 0 and max(
            g.nodes()) == g.order() - 1, f'{self.name!r} node labels are not continuous from 0'
        g.name = self.name
        return g

    def get_graph_and_clustering(self, trial: int):
        output_graph_path = self.graph_basepath / f'{self.name}_{trial}.gml'
        lfr_g = nx.read_gml(output_graph_path, destringizer=int)

        # set graph level attributes
        lfr_g.name = self.name + f'_{trial}'  # add trial number for output
        lfr_g.graph['mu'] = self.mu
        lfr_g.graph['avg_k'] = self.avg_k
        lfr_g.graph['max_k'] = self.max_k

        ground_truth_labels = nx.get_node_attributes(lfr_g, name='ground_truth')
        clusters_dict = {}  # key: community label, value: list of member nodes

        for u, c_u in ground_truth_labels.items():
            if c_u not in clusters_dict:
                clusters_dict[c_u] = []
            clusters_dict[c_u].append(u)

        clustering = NodeClustering(communities=list(clusters_dict.values()), graph=lfr_g,
                                    method_name='ground_truth')
        return lfr_g, clustering

    def generate_graphs(self, num_graphs: int, overwrite: bool = False):
        """
         Generate num_graphs graphs with the same parameters
        """
        tau1, tau2 = 2, 3
        avg_degree = self.avg_k
        max_degree = self.max_k
        c_min = 10
        c_max = 100
        self.graph_basepath.mkdir(exist_ok=True)

        for trial in range(1, num_graphs + 1):
            output_graph = self.graph_basepath / f'{self.name}_{trial}.gml'
            if output_graph.exists() and not overwrite:
                continue
            args = f'cd {self.paths.lfr_dir}; ./benchmark -N {self.n} -k {avg_degree} -maxk {max_degree} -mu {self.mu} ' \
                   f'-t1 {tau1} -t2 {tau2} -minc {c_min} -maxc {c_max}'
            subprocess.run(args=args, shell=True)

            generated_graph_path, generated_cluster_path = self.paths.lfr_dir / 'network.dat', self.paths.lfr_dir / 'community.dat'

            # preprocess the grpah - reindex from 0, adjust cluster labels accordingly
            g = nx.read_edgelist(generated_graph_path, nodetype=int)

            clusters = {}
            with open(generated_cluster_path) as reader:
                for line in reader:
                    u, c_u = map(int, line.split())
                    clusters[u] = c_u
            nx.set_node_attributes(g, name='ground_truth', values=clusters)
            relabeled_g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='sorted')

            nx.write_gml(relabeled_g, path=output_graph)
            remove_files([generated_graph_path, generated_cluster_path, self.paths.lfr_dir / 'statistics.dat',
                          self.paths.lfr_dir / 'time_seed.dat'])
        return
