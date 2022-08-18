"""Container for clustering"""
import abc
import csv
import itertools
import random
import subprocess
import sys
import time
from datetime import datetime
from typing import Tuple, Dict

import cdlib
import matplotlib.pyplot as plt
import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from cdlib import NodeClustering, algorithms
from cdlib.evaluation import normalized_mutual_information, adjusted_mutual_information
from tqdm.auto import tqdm

sys.path.extend(['./', '../'])

from src.utils import Paths, plot_network_clusters, write_clustering, remove_files, clusim_ec_score


class BaseClustering:
    def __init__(self, paths: Paths, input_graph: nx.Graph,
                 ground_truth_clustering: NodeClustering, method: str):
        self.paths = paths
        self.method = method
        self.input_graph = input_graph
        assert input_graph.name != '', f'Input graph name is blank.'
        self.name = input_graph.name
        self.ground_truth_clustering = ground_truth_clustering
        self.output_cluster_base = self.paths.results_dir / 'LFR' / self.method
        return

    @abc.abstractmethod
    def _run_method(self, **params) -> NodeClustering:
        pass

    def write_stats(self, stats_dict, stats_path):
        file_exists = stats_path.exists()
        round_cols = ['mu', 'running_time', 'modularity', 'NMI', 'EC', 'AMI']

        if isinstance(self, CorrelationClustering):
            columns = ['timestamp', 'name', 'mu', 'avg_k', 'max_k',
                       'method', 'tau_max', 'tau_diff', 'tau_ix', 'tau', 'orig_clusters', 'trial', 'running_time',
                       'num_clusters', 'before_merge', 'modularity', 'NMI', 'AMI', 'EC', 'output_cluster_path']
            round_cols += ['tau_max']
        else:
            columns = ['timestamp', 'name', 'mu', 'avg_k', 'max_k',
                       'method', 'trial', 'orig_clusters',
                       'running_time', 'num_clusters', 'modularity', 'NMI', 'AMI', 'EC', 'output_cluster_path']

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
        stats_path = self.paths.results_dir / 'stats_others.csv'

        for trial in tqdm(range(1, num_trials + 1), desc=f'{self.method!r}'):
            cluster_path = self.output_cluster_base / f'{self.name}_{self.method}_{trial}_clusters.txt'

            if cluster_path.exists() and not overwrite:  # clustering already exists
                # print(f'Existing clustering found {self.method!r} {self.name!r} {trial}')
                continue

            tic = time.time()
            generate_clustering = self._run_method()
            time_taken = time.time() - tic

            mu = self.input_graph.graph['mu']
            avg_k = self.input_graph.graph['avg_k']
            max_k = self.input_graph.graph['max_k']
            timestamp = datetime.now().strftime('%c')

            clustering_stats = self.get_clustering_stats(generated_clustering=generate_clustering,
                                                         running_time=time_taken, mu=mu, avg_k=avg_k, max_k=max_k,
                                                         timestamp=timestamp, trial=trial,
                                                         output_cluster_path=cluster_path)

            self.write_stats(stats_dict=clustering_stats, stats_path=stats_path)
            # write_clustering(clustering=generate_clustering, cluster_path=cluster_path)

        return

    def get_clustering_stats(self, generated_clustering: NodeClustering, **params) -> Dict:
        modularity = generated_clustering.newman_girvan_modularity().score
        num_clusters = len(generated_clustering.communities)
        orig_clusters = len(self.ground_truth_clustering.communities)
        nmi = normalized_mutual_information(self.ground_truth_clustering, generated_clustering).score
        ami = adjusted_mutual_information(self.ground_truth_clustering, generated_clustering).score
        ec = clusim_ec_score(self.ground_truth_clustering, generated_clustering)
        stats = dict(name=self.name, method=self.method, modularity=modularity, num_clusters=num_clusters, NMI=nmi,
                     AMI=ami, orig_clusters=orig_clusters, EC=ec)
        stats.update(params)
        return stats


class CorrelationClustering(BaseClustering):
    """
    Container for running correlation clustering methods
    main idea: avoid recalculation of objects that don't change - esp matrices
    """

    _METHODS = ['GW', 'RW', 'Diff']  # CL = Correlation leakage, # GW = GraphWalk, RW = RandomWalk, Diff = Diffusion

    def __init__(self, method: str, continuous: bool, paths: Paths, input_graph: nx.Graph,
                 ground_truth_clustering: NodeClustering, exact: bool = True):

        super().__init__(paths=paths, input_graph=input_graph, ground_truth_clustering=ground_truth_clustering,
                         method=method)
        assert method in CorrelationClustering._METHODS, f'Incorrect {method!r} pick from {CorrelationClustering._METHODS}'
        self.method = method

        assert isinstance(continuous, bool)
        self.is_continuous = continuous  # true if continuous
        self.exact = exact  # exact computation of R
        self._init_matrices()
        self.full_leakage_score = None
        return

    def _init_matrices(self):
        n, m = self.input_graph.order(), self.input_graph.size()

        self.A = nx.to_scipy_sparse_array(self.input_graph, dtype='d', nodelist=sorted(self.input_graph.nodes()))
        self.d = self.A @ np.ones((n, 1))  # n x 1 degree vector
        self.D = scipy.sparse.diags(self.d.flatten())

        self.A_prime = (self.d @ np.transpose(self.d)) / (2 * m)  # null model

        if self.method == 'GW':
            if self.is_continuous:
                self.H = -self.A
                self.H_prime = -self.A_prime
            else:
                self.H = self.A
                self.H_prime = self.A_prime

        elif self.method == 'Diff':
            if self.is_continuous:
                self.H = self.D - self.A
                self.H_prime = self.D - self.A_prime  # D_prime should equal D
            else:
                raise NotImplementedError()

        elif self.method == 'RW':
            if self.is_continuous:
                n = self.input_graph.order()
                I = scipy.sparse.eye(n, n)
                D_inv = scipy.sparse.diags((1 / self.d).flatten())

                self.H = I - (self.A @ D_inv)
                self.H_prime = I - (self.A_prime @ D_inv)  # I_prime and D_inv_prime should be the same

            else:
                D_inv = scipy.sparse.diags((1 / self.d).flatten())
                self.H = self.A @ D_inv
                self.H_prime = self.A_prime @ D_inv
        else:
            raise Exception()

        self.tau_max = 1 / abs(
            scipy.sparse.linalg.eigs(self.H, k=1, return_eigenvectors=False)[0])  # to make sure it's non-negative

        # tau_diff is inverse of the second smallest non-zero eigenvalue of H
        sorted_eigenvals = np.sort(np.abs(scipy.linalg.eig(self.H.toarray(), right=False)))
        second_smallest_eig = None
        for eig in sorted_eigenvals:
            if not np.isclose(eig, 0):
                second_smallest_eig = eig
                break
        assert second_smallest_eig is not None, f'second smallest eigenvalue is None!!'
        self.tau_diff = 1 / second_smallest_eig

        return

    def generate_R_matrix(self, tau: float = None):
        H, H_prime = self.H, self.H_prime
        H_t, H_prime_t = self.H.transpose(), self.H_prime.transpose()

        if self.is_continuous:
            if self.exact:
                U = scipy.sparse.linalg.expm(-tau * H)
                U = U @ U.transpose()
                R = H @ U + U @ H_t - H_prime @ U - U @ H_prime_t

            else:  # approx computation
                R0 = H + H_t - H_prime - H_prime_t

                R1 = (H @ H + H_t @ H_t + 2 * H @ H_t) \
                     - 1 * (H_prime @ H + H_prime @ H_t + H @ H_prime_t + H_t @ H_prime_t)

                R = R0 - (tau * R1)

        else:  # discrete method
            R = -H @ H_t + H_prime @ H_prime_t

        self.full_leakage_score = np.sum(R)  # every node is in its own community
        # assert not np.isclose(self.full_leakage_score, 0), f'Leakage score is 0 for {self.method} {tau=:.4g}'
        return R

    def run(self, num_trials: int, overwrite: bool, write_stats: bool = True, try_merging: bool = False):
        # running the method num_trials many times
        stats_path = self.paths.results_dir / 'stats_CL.csv'

        if not self.is_continuous:
            taus = [-1]
        else:
            # taus = np.linspace(0, self.tau_max, 5)  # change to log scale
            taus_1 = np.geomspace(np.finfo(float).eps, self.tau_max, num=5)
            taus_2 = np.geomspace(self.tau_max, self.tau_diff, num=5)[1:]  # prevent tau_max from getting double counted
            taus = np.concatenate([taus_1, taus_2])

        clusterings = {}
        for tau_ix, tau in enumerate(taus):
            R = self.generate_R_matrix(tau=tau)

            if tau not in clusterings:
                clusterings[tau] = []
            method_ = f'{self.method}-C' if self.is_continuous else f'{self.method}-D'
            method = f'{method_}-exact' if self.exact else method_

            desc = f'Corr {method} ..'

            for trial in tqdm(range(1, num_trials + 1), desc=desc, leave=False):
                cluster_path = self.output_cluster_base / f'{self.name}_{method}_{tau_ix}_{trial}_clusters.txt'

                if cluster_path.exists() and not overwrite:  # clustering already exists
                    # print(f'Existing clustering found {method!r} {self.name!r}')
                    continue

                tic = time.time()
                generated_clustering = self._run_method(R=R, try_merging=try_merging)
                time_taken = time.time() - tic
                generated_clustering.method_parameters.update({'method': method, 'tau': tau, 'tau_max': self.tau_max,
                                                               'tau_diff': self.tau_diff, 'trial': trial})
                clusterings[tau].append(generated_clustering)

                if write_stats:
                    mu = self.input_graph.graph.get('mu', np.nan)
                    avg_k = self.input_graph.graph.get('avg_k', np.nan)
                    max_k = self.input_graph.graph.get('max_k', np.nan)
                    before_merge = generated_clustering.method_parameters['before_merge']
                    timestamp = datetime.now().strftime('%c')

                    clustering_stats = self.get_clustering_stats(generated_clustering=generated_clustering,
                                                                 running_time=time_taken, tau=tau, tau_ix=tau_ix,
                                                                 tau_max=self.tau_max, tau_diff=self.tau_diff,
                                                                 mu=mu, avg_k=avg_k, max_k=max_k,
                                                                 timestamp=timestamp, trial=trial,
                                                                 before_merge=before_merge,
                                                                 output_cluster_path=cluster_path)
                    clustering_stats['method'] = method  # methods with augmented 'D' or 'C"
                    self.write_stats(stats_dict=clustering_stats, stats_path=stats_path)
                    write_clustering(clustering=generated_clustering, cluster_path=cluster_path)

        return clusterings

    def _run_method(self, R, try_merging=False):
        # store clusters as list of sets of nodes, flatten them out at the end of each iteration
        R = R * -1  # to maximize

        communities = [[{u}] for u in self.input_graph.nodes()]
        super_nodes = [{u} for u in self.input_graph.nodes()]  # list of flattened supernodes
        mapping = {u: i for i, u in enumerate(sorted(self.input_graph.nodes()))}

        changed = True
        while changed:
            changed = False
            random.shuffle(super_nodes)
            for s_i in super_nodes:
                # figure out the current community where s_i is in
                s_i_comm_ix = None
                for i, comm in enumerate(communities):
                    if s_i in comm:
                        s_i_comm_ix = i
                        break
                assert s_i_comm_ix is not None, f's_i index is None'

                d_i = communities[s_i_comm_ix]
                # loss of s_i leaving its own community
                loss = 0
                for s_j in d_i:
                    if s_i == s_j:
                        continue
                    for i in s_i:
                        for j in s_j:
                            loss -= 2 * R[mapping[i], mapping[j]]

                # check other communities for s_i to merge with
                max_gain = 0
                best_community_ix = None
                for ix_2, d_j in enumerate(communities):
                    if s_i_comm_ix == ix_2:  # skip over s_i's own community
                        continue
                    gain = 0
                    for i in s_i:
                        for s_j in d_j:
                            for j in s_j:
                                gain += 2 * R[mapping[i], mapping[j]]

                    if gain > max_gain:
                        max_gain = gain
                        best_community_ix = ix_2

                change_in_quality = max_gain + loss
                if best_community_ix is not None and change_in_quality > 0:
                    changed = True
                    communities[best_community_ix].append(s_i)  # add s_i to its new community

                    # move s_i from its previous community to its new community
                    communities[s_i_comm_ix].remove(s_i)
                    if len(communities[s_i_comm_ix]) == 0:  # the community is empty
                        del communities[s_i_comm_ix]

            # flatten the communities at the end
            super_nodes = [set.union(*s_i) for s_i in communities]  # note the lack of inner list
            communities = [[set.union(*s_i)] for s_i in
                           communities]  # notice the outer list inside: communities are list of sets

        communities = [list(comm[0]) for comm in communities if len(comm) > 0]
        # calculate leakage score based on the clustering found
        leakage_score = 0
        for comm in communities:
            for i, j in itertools.combinations_with_replacement(comm, 2):  # to account for diagonals
                leakage_score += 2 * R[mapping[i], mapping[j]]

        leakage_score -= np.trace(R)  # to prevent diagonals from getting counted twice
        leakage_diff = round(self.full_leakage_score - leakage_score, 5)

        clustering = NodeClustering(graph=self.input_graph, method_name=self.method, communities=communities,
                                    method_parameters={'full_leakage_score': self.full_leakage_score,
                                                       'leakage_score': leakage_score, 'leakage_diff': leakage_diff})

        return clustering

    def _run_method_old(self, R, try_merging: bool = False):
        """
        R is the quality matrix
        """
        # TODO: pick the clustering with the max R score

        R = R * -1  # to maximize

        n = R.shape[0]
        communities = [{u} for u in self.input_graph.nodes()]  # list of sets

        mapping = {u: i for i, u in enumerate(sorted(self.input_graph.nodes()))}

        iters = 0
        changed = True
        nodes = list(self.input_graph.nodes())
        random.shuffle(nodes)

        while changed:
            iters += 1
            changed = False

            for i in nodes:  # try all nodes
                current_comm_i_idx = [ix for ix, comm in enumerate(communities) if i in comm][0]

                loss_i_from_current_comm = 0  # loss from i leaving its current comm
                for j in communities[current_comm_i_idx]:  # members of i's community
                    if i != j:
                        loss_i_from_current_comm -= R[mapping[i], mapping[j]]

                new_comm_i = None
                gain_i_to_new_comm = 0

                for comm_ix, old_community in enumerate(communities):  # iterate over all non-emptycommunities
                    if len(old_community) == 0:  # empty
                        continue
                    if comm_ix == current_comm_i_idx:  # skip over i's current community
                        continue
                    g = 0
                    for j in old_community:  # j is from a different community
                        g += R[mapping[i], mapping[j]]

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

        # try merging clusters if flag is enabled
        if try_merging:
            before_merge = len([comm for comm in communities if len(comm) > 0])
            changed = True
            random.shuffle(communities)  # shuffle community labels / indices

            while changed:
                changed = False
                for i, old_community in enumerate(communities):
                    if len(old_community) == 0:
                        continue
                    merge_ix = None
                    max_gain = 0
                    for j, new_community in enumerate(communities):
                        if j == i:
                            continue

                        gain = 0
                        for u in old_community:
                            for v in new_community:
                                gain += R[mapping[u], mapping[v]]

                        if gain > max_gain:
                            max_gain = gain
                            merge_ix = j

                    if merge_ix is not None:  # all members of community becomes part of new community
                        assert max_gain > 0
                        changed = True
                        # merge community with new community
                        copy_old_community = set(communities[i])
                        for u in copy_old_community:
                            communities[i].remove(u)  # remove u from existing community
                            communities[merge_ix].add(u)  # add u to the merged community

                after_merge = len([comm for comm in communities if len(comm) > 0])
                # print(f'\n{self.method}\t {before_merge=}\t {after_merge=}')
        else:
            before_merge = None

        communities = [list(comm) for comm in communities if len(comm) > 0]
        clustering = NodeClustering(graph=self.input_graph, method_name=self.method, communities=communities,
                                    method_parameters={'before_merge': before_merge})

        return clustering


class OtherClustering(BaseClustering):
    _METHODS = ['louvain', 'greedy_modularity', 'label_propagation', 'infomap', 'leiden', 'eigenvector']

    def __init__(self, paths: Paths, input_graph: nx.Graph, method: str, ground_truth_clustering: NodeClustering):
        super().__init__(paths=paths, input_graph=input_graph, method=method,
                         ground_truth_clustering=ground_truth_clustering)
        self.method = method
        assert method in OtherClustering._METHODS, f'Incorrect method: {method!r}, pick from {OtherClustering._METHODS}'
        return

    def _run_method(self, **params) -> NodeClustering:
        cdlib_alg = getattr(cdlib.algorithms, self.method)
        clustering = cdlib_alg(self.input_graph)
        return clustering


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


if __name__ == '__main__':
    paths = Paths(basedir='../')
    avg_k = 20
    mu = 0.1
    # lfr = LFRGraph(paths=paths, n=100, mu=mu, avg_k=avg_k)
    # g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=1)
    sizes = [20, 20, 20]
    probs = [[0.25, 0.02, 0.02], [0.02, 0.35, 0.02], [0.02, 0.02, 0.40]]

    # g = nx.stochastic_block_model(sizes, probs, seed=1)
    # g = nx.ring_of_cliques(51, clique_size=4)
    g = nx.barbell_graph(4, 5);
    g.name = 'barbell'
    ground_truth_clustering = None

    method = 'Diff'

    corr_cluster = CorrelationClustering(paths=paths, input_graph=g, continuous=True, method=method,
                                         ground_truth_clustering=ground_truth_clustering, exact=True)

    clusterings = corr_cluster.run(num_trials=1, write_stats=False, overwrite=True)
    clusterings
