"""Container for clustering"""
import csv
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union

import cdlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
from cdlib import NodeClustering, algorithms
from cdlib.evaluation import normalized_mutual_information, adjusted_mutual_information
from tqdm.auto import tqdm

from src.utils import Paths, plot_network_clusters, write_clustering


class LFRGraph:
    """
    Container for LFR graphs - store both the graph object and the node community labels
    """

    def __init__(self, paths: Paths, n: int, mu: float):
        self.paths = paths
        self.n = n
        self.mu = mu
        self.name = f'LFR-{self.n}-{int(self.mu * 100)}'

        self.input_g = self._read_graph()
        self.pos = None  # store the node positions
        self.ground_truth_clustering: NodeClustering = self._read_clustering(method='ground_truth')

        self.generated_clusterings = {}  # stores the generated clustering objects
        self.stats = {}  # store the stats
        return

    def run_method(self, method: str, num_trials: int = 1, write_results: bool = False, overwrite: bool = False):
        """
        runs the clustering using CDlib, write the results in a CSV, and store the clustering as a text file
        """
        new_run = False  # true if it's a new run
        for trial in range(1, num_trials + 1):
            output_cluster_path = self.paths.results_dir / 'LFR' / method / f'{self.name}_{method}_{trial}_clusters.txt'

            if output_cluster_path.exists() and not overwrite:
                clustering = self._read_clustering(method=method, cluster_path=output_cluster_path)
                time_taken = np.NAN

            else:  # compute clustering
                new_run = True
                if method in ('leiden', 'louvain', 'infomap', 'greedy_modularity', 'label_propagation'):
                    cdlib_alg = getattr(cdlib.algorithms, method)
                    tic = time.time()
                    clustering = cdlib_alg(self.input_g)
                    time_taken = time.time() - tic
                else:
                    raise NotImplementedError(f'{method=!r} not implemented')

            if method not in self.generated_clusterings:
                self.generated_clusterings[method] = []
            self.generated_clusterings[method].append(clustering)

            modularity = clustering.newman_girvan_modularity().score
            num_clusters = len(clustering.communities)
            nmi = normalized_mutual_information(self.ground_truth_clustering, clustering).score
            ami = adjusted_mutual_information(self.ground_truth_clustering, clustering).score

            stats = dict(timestamp=datetime.now(), name=self.name, method=method, trial=trial,
                         time_taken=time_taken, num_clusters=num_clusters, modularity=modularity,
                         NMI=nmi, AMI=ami, output_cluster_path=output_cluster_path)
            # rich.print(stats)

            if method not in self.stats:
                self.stats[method] = []
            self.stats[method].append(stats)

            if new_run:
                # write clustering to file
                write_clustering(clustering=clustering, cluster_path=output_cluster_path)
                self.write_stats(stats)

        return

    def write_stats(self, stats):
        stats_path = self.paths.results_dir / 'stats.csv'
        file_exists = stats_path.exists()

        columns = ['timestamp', 'name', 'method', 'trial', 'time_taken', 'num_clusters', 'modularity', 'NMI', 'AMI',
                   'output_cluster_path']

        with open(stats_path, 'a') as f:
            csv_writer = csv.DictWriter(f, fieldnames=columns)
            if not file_exists:
                csv_writer.writeheader()
            csv_writer.writerow(stats)

        return

    def plot_clustering(self, method: str = 'ground_truth', ix: int = -1, figsize: Tuple[int, int] = (25, 25),
                        node_size: int = 150):
        if self.pos is None:
            self.pos = nx.spring_layout(self.input_g, iterations=100)
        ncols = 1
        if method == 'ground_truth':
            clustering = self.ground_truth_clustering
        else:
            clustering = self.generated_clusterings[method][ix]  # plot the latest by default
            # ncol = 2

        modularity = clustering.newman_girvan_modularity().score
        num_clusters = len(clustering.communities)

        title = f'{self.name!r} \t |V| = {self.input_g.order():,} \t |E| = {self.input_g.size():,}\n' \
                f'method = {method!r} \t clusters = {num_clusters:,} \t modularity = {modularity:.3g}'

        if method != 'ground_truth':
            nmi = normalized_mutual_information(self.ground_truth_clustering, clustering).score
            ami = adjusted_mutual_information(self.ground_truth_clustering, clustering).score
            title += f'\t NMI = {nmi:.3g} \t AMI = {ami:.3g}'

        fig, *axs = plt.subplots(figsize=figsize, layout='constrained', ncols=ncols)

        plot_network_clusters(self.input_g, figsize=figsize, node_size=node_size, cmap='tab20',
                              partition=clustering, position=self.pos, ax=axs[0], title=title)

        plt.show()
        return

    def __str__(self) -> str:
        ground_truth_cluster_count = len(self.ground_truth_clustering.communities)
        ground_truth_modularity = self.ground_truth_clustering.newman_girvan_modularity().score
        st = f'<{self.name!r} \t |V|={self.input_g.order():,} \t |E| = {self.input_g.size():,} \t ' \
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

    def _read_clustering(self, method: str, cluster_path: Union[str, Path] = None) -> NodeClustering:
        if method == 'ground_truth':
            cluster_path = self.paths.data_dir / 'LFR' / f'{self.name}_clusters.txt'

        with open(cluster_path) as reader:
            clusters_dict = {}  # key: community label, value: list of member nodes
            for line in reader:
                u, c_u = map(int, line.split())
                if c_u not in clusters_dict:
                    clusters_dict[c_u] = []

                clusters_dict[c_u].append(u)

        clustering = NodeClustering(communities=list(clusters_dict.values()), graph=self.input_g,
                                    method_name=method)
        modularity = clustering.newman_girvan_modularity().score
        if method == 'ground_truth':
            tqdm.write(f'{self.name!r} {method!r} #communities: {len(clusters_dict):,} modularity: {modularity:.3g}')
        return clustering
