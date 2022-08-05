"""Container for clustering"""
import abc
import csv
import random
import time
from datetime import datetime
from pathlib import Path
from typing import Tuple, Union, Dict

import cdlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from cdlib import NodeClustering, algorithms
from cdlib.evaluation import normalized_mutual_information, adjusted_mutual_information
from tqdm.auto import tqdm

from src.utils import Paths, plot_network_clusters, write_clustering


class BaseClustering:
    def __init__(self, paths: Paths, name: str, lfr_graph, method: str):
        self.paths = paths
        self.name = name
        self.method = method
        self.lfr_graph = lfr_graph
        self.output_cluster_base = self.paths.results_dir / self.name / self.method
        return

    @abc.abstractmethod
    def _run_method(self, **params) -> NodeClustering:
        pass

    def write_stats(self, stats_dict, stats_path):
        file_exists = stats_path.exists()

        if isinstance(self, CorrelationClustering):
            columns = ['timestamp', 'name', 'mu', 'method', 'tau', 'tau_max', 'trial', 'running_time', 'num_clusters',
                       'modularity',
                       'NMI', 'AMI', 'output_cluster_path']
        else:
            columns = ['timestamp', 'name', 'mu', 'method', 'trial', 'running_time', 'num_clusters',
                       'modularity', 'NMI', 'AMI', 'output_cluster_path']

        round_cols = ['mu', 'running_time', 'modularity', 'NMI', 'AMI']
        for col in round_cols:
            stats_dict[col] = round(stats_dict[col], 4)

        with open(stats_path, 'a') as f:
            csv_writer = csv.DictWriter(f, fieldnames=columns)
            if not file_exists:
                csv_writer.writeheader()
            csv_writer.writerow(stats_dict)

        return

    def run(self, num_trials: int, overwrite: bool):
        # running the method num_trials many times
        stats_filename = self.paths.results_dir / 'stats_others.csv'

        for trial in range(1, num_trials + 1):
            cluster_path = self.output_cluster_base / f'{self.name}_{self.method}_{trial}_clusters.txt'

            if cluster_path.exists() and not overwrite:  # clustering already exists
                print(f'Existing clustering found {self.method!r} {self.name!r} {trial}')
                continue

            tic = time.time()
            generate_clustering = self._run_method()
            time_taken = time.time() - tic
            clustering_stats = self.get_clustering_stats(generated_clustering=generate_clustering,
                                                         running_time=time_taken, mu=self.lfr_graph.mu,
                                                         timestamp=datetime.now(), trial=trial,
                                                         output_cluster_path=cluster_path)
            self.write_stats(stats_dict=clustering_stats, stats_path=stats_filename)
            write_clustering(clustering=generate_clustering, cluster_path=cluster_path)

        return

    def get_clustering_stats(self, generated_clustering: NodeClustering, **params) -> Dict:
        modularity = generated_clustering.newman_girvan_modularity().score
        num_clusters = len(generated_clustering.communities)
        nmi = normalized_mutual_information(self.lfr_graph.ground_truth_clustering, generated_clustering).score
        ami = adjusted_mutual_information(self.lfr_graph.ground_truth_clustering, generated_clustering).score

        stats = dict(name=self.name, method=self.method, modularity=modularity, num_clusters=num_clusters, NMI=nmi,
                     AMI=ami)
        stats.update(params)
        return stats


class CorrelationClustering(BaseClustering):
    """
    Container for running correlation clustering methods
    main idea: avoid recalculation of objects that don't change - esp matrices
    """

    _METHODS = ['GW', 'RW', 'Diff']  # CL = Correlation leakage, # GW = GraphWalk, RW = RandomWalk, Diff = Diffusion

    def __init__(self, method: str, continuous: bool, paths: Paths, name: str, lfr_graph):
        super().__init__(paths, name, lfr_graph, method)
        assert method in CorrelationClustering._METHODS, f'Incorrect {method!r} pick from {CorrelationClustering._METHODS}'
        self.method = method
        self.input_g = lfr_graph.input_g

        assert isinstance(continuous, bool)
        self.continuous = continuous  # true if continuous

        self.A, self.A_prime, self.d, self.D = self._init_matrices()
        self.R0, self.R1, self.tau_max = self.generate_R_matrices()
        return

    def _init_matrices(self):
        n, m = self.input_g.order(), self.input_g.size()

        A = nx.to_scipy_sparse_array(self.input_g, dtype='d', nodelist=list(range(n)))
        d = A @ np.ones((n, 1))  # n x 1 degree vector
        D = scipy.sparse.diags(d.flatten())

        A_prime = (d @ np.transpose(d)) / (2 * m)  # null model
        return A, A_prime, d, D

    def generate_R_matrices(self):
        if self.method == 'GW':
            if self.continuous:
                H = -self.A
                H_prime = -self.A_prime
            else:
                H = self.A
                H_prime = self.A_prime

        elif self.method == 'Diff':
            if self.continuous:
                H = self.D - self.A
                H_prime = self.D - self.A_prime  # D_prime should equal D
            else:
                raise NotImplementedError()

        elif self.method == 'RW':
            if self.continuous:
                n = self.input_g.order()
                I = scipy.sparse.eye(n, n)
                D_inv = scipy.sparse.diags((1 / self.d).flatten())

                H = I - (self.A @ D_inv)
                H_prime = I - (self.A_prime @ D_inv)  # I_prime and D_inv_prime should be the same

            else:
                D_inv = scipy.sparse.diags((1 / self.d).flatten())
                H = self.A @ D_inv
                H_prime = self.A_prime @ D_inv
        else:
            raise Exception()

        tau_max = 1 / abs(scipy.sparse.linalg.eigs(H, k=1)[0][0])  # to make sure it's non-negative
        H_t, H_prime_t = H.transpose(), H_prime.transpose()

        if self.continuous:  # discrete method
            R0 = H + H_t - H_prime - H_prime_t

            R1 = (H @ H + H_t @ H_t + 2 * H @ H_t) \
                 - 1 * (H_prime @ H + H_prime @ H_t + H @ H_prime_t + H_t @ H_prime_t)
        else:
            R0 = -H @ H_t + H_prime @ H_prime_t
            R1 = None
        return R0, R1, tau_max

    def run(self, num_trials: int, overwrite: bool):
        # running the method num_trials many times
        stats_filename = self.paths.results_dir / 'stats_CL.csv'

        if not self.continuous:
            taus = [0]
        else:
            taus = np.linspace(0, self.tau_max, 3)

        for trial in tqdm(range(1, num_trials + 1), desc=f'Correlation clustering {self.method}..'):
            for tau in taus:
                method = f'{self.method}-C' if self.continuous else f'{self.method}-D'
                cluster_path = self.output_cluster_base / f'{self.name}_{method}_{tau}_{trial}_clusters.txt'

                if cluster_path.exists() and not overwrite:  # clustering already exists
                    print(f'Existing clustering found {self.method!r} {self.name!r} {trial}')
                    continue

                tic = time.time()
                generated_clustering = self._run_method(tau=tau)
                time_taken = time.time() - tic
                clustering_stats = self.get_clustering_stats(generated_clustering=generated_clustering,
                                                             running_time=time_taken, tau=tau, tau_max=self.tau_max,
                                                             mu=self.lfr_graph.mu,
                                                             timestamp=datetime.now(), trial=trial,
                                                             output_cluster_path=cluster_path)
                clustering_stats['method'] = method  # methods with augmented 'D' or 'C"
                self.write_stats(stats_dict=clustering_stats, stats_path=stats_filename)
                write_clustering(clustering=generated_clustering, cluster_path=cluster_path)
        return

    def _run_method(self, **params):
        if self.R1 is None:
            R = self.R0
        else:
            tau = params['tau']
            R = self.R0 - (tau * self.R1)
        R = R * -1  # to maximize

        n = R.shape[0]
        communities = [{u} for u in range(n)]  # list of sets

        iters = 0
        changed = True
        nodes = list(range(n))
        random.shuffle(nodes)

        while changed:
            iters += 1
            changed = False

            for i in nodes:  # try all nodes
                current_comm_i_idx = [ix for ix, comm in enumerate(communities) if i in comm][0]

                loss_i_from_current_comm = 0  # loss from i leaving its current comm
                for j in communities[current_comm_i_idx]:  # members of i's community
                    if i != j:
                        loss_i_from_current_comm -= R[i, j]

                new_comm_i = None
                gain_i_to_new_comm = 0

                for comm_ix, community in enumerate(communities):  # iterate over all non-emptycommunities
                    if len(community) == 0:  # empty
                        continue
                    if comm_ix == current_comm_i_idx:  # skip over i's current community
                        continue
                    g = 0
                    for j in community:  # j is from a different community
                        g += R[i, j]

                    if g > gain_i_to_new_comm:
                        gain_i_to_new_comm = g
                        new_comm_i = comm_ix

                quality_change = gain_i_to_new_comm + loss_i_from_current_comm

                if new_comm_i is not None and quality_change > 0:
                    # remove i from old comm, add it to the new comm
                    changed = True
                    communities[new_comm_i].add(i)
                    communities[current_comm_i_idx].remove(i)

                    if len(communities[current_comm_i_idx]) == 0:
                        del communities[current_comm_i_idx]

        communities = [list(comm) for comm in communities if len(comm) > 0]
        clustering = NodeClustering(graph=self.input_g, method_name=self.method, communities=communities)

        return clustering

    def generate_R_matrices_old(self):
        if self.method == 'graph-walk_cont':
            H = -self.A
            R0 = -2 * (self.A - self.A_prime)  # eq 15
            R1 = 4 * (self.A @ self.A) - 2 * (self.A_prime @ self.A + self.A @ self.A_prime)

        elif self.method == 'diffusion_cont':
            H = self.D - self.A
            R0 = -2 * (self.A - self.A_prime)

            # eq 22
            R1 = 4 * (self.A @ self.A) - 2 * (self.D @ self.A) - 2 * (self.A @ self.D) - 2 * (self.A @ self.A_prime) \
                 - 2 * (self.A_prime @ self.A) + 2 * (self.D @ self.A_prime) + 2 * (self.A_prime @ self.D)

        elif self.method == 'random-walk_cont':
            n = self.input_g.order()
            # I = scipy.sparse.eye(n, n)
            # D_inv = scipy.sparse.diags((1 / self.d).flatten())
            I = np.eye(n)
            D_inv = np.eye(n) * (1 / self.d)

            H = I - (self.A @ D_inv)
            R0 = -1 * (self.A - self.A_prime) @ D_inv - D_inv @ (self.A - self.A_prime)

            # eq 32
            R1 = (self.A @ D_inv @ self.A @ D_inv) + (D_inv @ self.A @ D_inv @ self.A) \
                 + (self.A @ D_inv @ D_inv @ self.A) + (D_inv @ self.A @ self.A @ D_inv) \
                 - (self.A_prime @ D_inv @ self.A @ D_inv) - (self.A_prime @ D_inv @ D_inv @ self.A) \
                 - (self.A @ D_inv @ D_inv @ self.A_prime) - (D_inv @ self.A @ D_inv @ self.A_prime) - (2 * R0)

        elif self.method == 'graph-walk_disc':
            H = self.A
            # eq 38
            R0 = (-self.A @ self.A) + (self.A_prime @ self.A_prime)
            R1 = None

        elif self.method == 'random-walk_disc':
            n = self.input_g.order()
            D_inv = np.eye(n) * (1 / self.d)
            # D_inv = scipy.sparse.diags((1 / self.d).flatten())
            H = self.A @ D_inv

            R0 = -(self.A @ D_inv @ D_inv @ self.A) + (self.A_prime @ D_inv @ D_inv @ self.A_prime)
            R1 = None

        else:
            raise NotImplementedError()

        # tau_max = 1 / abs(scipy.sparse.linalg.eigsh(H, k=1)[0][0])  # to make sure it's non-negative
        tau_max = -1  # change later
        return R0, R1, tau_max


class OtherClustering(BaseClustering):
    _METHODS = ['louvain', 'greedy_modularity', 'label_propagation']

    def __init__(self, paths: Paths, name: str, lfr_graph, method: str):
        super().__init__(paths, name, lfr_graph, method)
        self.method = method
        assert method in OtherClustering._METHODS, f'Incorrect method: {method!r}, pick from {OtherClustering._METHODS}'
        self.input_g = lfr_graph.input_g
        return

    def _run_method(self, **params) -> NodeClustering:
        cdlib_alg = getattr(cdlib.algorithms, self.method)
        clustering = cdlib_alg(self.input_g)
        return clustering


class LFRGraph:
    """
    Container for LFR graphs - store both the graph object and the node community labels
    """

    def __init__(self, paths: Paths, n: int, mu: float):
        self.paths = paths
        self.n = n
        self.mu = round(mu, 2)
        self.name = f'LFR-{self.n}-{int(self.mu * 100)}'

        self.input_g = self._read_graph()
        self.pos = None  # store the node positions
        self.ground_truth_clustering: NodeClustering = self._read_clustering(method='ground_truth')
        return

    def _run_method(self, method: str, num_trials: int = 1, overwrite: bool = False, tau: float = np.nan,
                    corr_cluster=None):
        """
        runs the clustering using CDlib, write the results in a CSV, and store the clustering as a text file
        """
        if method == 'infomap':
            raise NotImplementedError(f'InfoMap not working')

        new_run = False  # true if it's a new run
        for trial in range(1, num_trials + 1):
            if tau is None:
                output_cluster_path = self.paths.results_dir / 'LFR' / method / f'{self.name}_{method}_{trial}_clusters.txt'
            else:  # for correlation leakage methods
                t = round(tau, 4)
                output_cluster_path = self.paths.results_dir / 'LFR' / method / f'{self.name}_{method}_{t}_{trial}_clusters.txt'

            if method == 'original':  # store the original values once
                clustering = self.ground_truth_clustering
                time_taken = np.NAN
                new_run = True
                if trial > 1:  # process the original graph only once
                    break

            elif output_cluster_path.exists() and not overwrite:
                clustering = self._read_clustering(method=method, cluster_path=output_cluster_path)
                time_taken = np.NAN

            else:  # compute clustering
                new_run = True
                if method in ('leiden', 'louvain', 'greedy_modularity', 'label_propagation', 'sbm_dl'):
                    cdlib_alg = getattr(cdlib.algorithms, method)
                    tic = time.time()
                    clustering = cdlib_alg(self.input_g)
                    time_taken = time.time() - tic

                elif method in (
                'graph-walk_cont', 'graph-walk_disc', 'random-walk_cont', 'random-walk_disc', 'diffusion_cont'):
                    tic = time.time()
                    assert corr_cluster.method == method, f'Method mismatch'
                    clustering = corr_cluster.get_clusters(tau=tau)
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

            stats = dict(timestamp=datetime.now(), name=self.name, mu=self.mu, method=method, trial=trial,
                         running_time=time_taken, num_clusters=num_clusters, modularity=modularity,
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


    def plot_clustering(self, method: str = 'ground_truth', ix: int = -1, figsize: Tuple[int, int] = (25, 25),
                        node_size: int = 150, clustering=None):
        if self.pos is None:
            self.pos = nx.spring_layout(self.input_g, iterations=100)

        ncols = 1
        if method == 'ground_truth':
            clustering = self.ground_truth_clustering
            ncols = 1
        elif clustering is None:
            clustering = self.generated_clusterings[method][ix]  # plot the latest by default
            ncols = 2

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






if __name__ == '__main__':
    test_g = nx.karate_club_graph()

    # test_g = nx.barbell_graph(4, 5)
    # test_g = nx.erdos_renyi_graph(n=20, p=0.75, seed=0)

    methods = ('graph-walk_cont', 'graph-walk_disc', 'random-walk_cont', 'random-walk_disc', 'diffusion_cont')
    for method in methods:
        corr_cluster_old = CorrelationClustering(input_g=test_g, method=method, old=True)
        corr_cluster_new = CorrelationClustering(input_g=test_g, method=method, old=False)

        assert np.allclose(corr_cluster_old.R0, corr_cluster_new.R0), f'{method!r} R0 mismatch'
        if corr_cluster_old.R1 is not None and corr_cluster_new.R1 is not None:
            if not np.allclose(corr_cluster_old.R1, corr_cluster_new.R1):
                print(f'{method!r} R1 mismatch')
        # if not np.isclose(corr_cluster_old.tau_max, corr_cluster_new.tau_max):
        #     print(f'{method!r} tau mismatch')
