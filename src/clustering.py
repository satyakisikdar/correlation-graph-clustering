"""Container for clustering"""
import abc
import csv
import itertools
import random
import subprocess
import sys
import time
from collections import defaultdict, deque
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
from cdlib.utils import convert_graph_formats
from tqdm.auto import tqdm

sys.path.extend(['./', '../'])

from src.utils import Paths, plot_network_clusters, write_clustering, remove_files, clusim_ec_score


def infomap(g_original: object, flags: str = "") -> NodeClustering:
    """
    Infomap is based on ideas of information theory.
    The algorithm uses the probability flow of random walks on a network as a proxy for information flows in the real system and it decomposes the network into modules by compressing a description of the probability flow.


    **Supported Graph Types**

    ========== ======== ========
    Undirected Directed Weighted
    ========== ======== ========
    Yes        Yes      Yes
    ========== ======== ========

    :param g_original: a networkx/igraph object
    :param flags: str flags for Infomap
    :return: NodeClustering object

    :Example:

    >>> from cdlib import algorithms
    >>> import networkx as nx
    >>> G = nx.karate_club_graph()
    >>> coms = algorithms.infomap(G)

    :References:

    Rosvall M, Bergstrom CT (2008) `Maps of random walks on complex networks reveal community structure. <https://www.pnas.org/content/105/4/1118/>`_ Proc Natl Acad SciUSA 105(4):1118–1123

    .. note:: Reference implementation: https://pypi.org/project/infomap/

    .. note:: Infomap Python API documentation: https://mapequation.github.io/infomap/python/
    """
    global imp, pipes
    if imp is None:
        try:
            import infomap as imp
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Optional dependency not satisfied: install infomap to use the selected feature."
            )
    if pipes is None:
        try:
            from wurlitzer import pipes
        except ModuleNotFoundError:
            raise ModuleNotFoundError(
                "Optional dependency not satisfied: install package wurlitzer to use infomap."
            )

    g = convert_graph_formats(g_original, nx.Graph, directed=g_original.is_directed())

    if g_original.is_directed() and "-d" not in flags and "--directed" not in flags:
        flags += " -d"

    g1 = nx.convert_node_labels_to_integers(g, label_attribute="name")
    name_map = nx.get_node_attributes(g1, "name")
    coms_to_node = defaultdict(list)

    with pipes():
        im = imp.Infomap(flags)

        if tuple(map(int, imp.__version__.split("."))) >= (1, 7, 1):
            n_dict = {i: str(n) for i, n in enumerate(g1.nodes)}
            im.add_nodes(n_dict)
        else:
            im.add_nodes(g1.nodes)

        for source, target, data in g1.edges(data=True):
            if "weight" in data:
                im.add_link(source, target, data["weight"])
            else:
                im.add_link(source, target)
        im.run()

        for node_id, module_id in im.modules:
            node_name = name_map[node_id]
            coms_to_node[module_id].append(node_name)

    coms_infomap = [list(c) for c in coms_to_node.values()]
    return NodeClustering(
        coms_infomap, g_original, "Infomap", method_parameters={"flags": flags, 'codelength': im.getCodelength()}
    )


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
                       'method', 'tau_max', 'tau_diff', 'tau_ix', 'tau', 'orig_clusters', 'running_time',
                       'leakage_score',
                       'num_clusters', 'modularity', 'NMI', 'AMI', 'EC', 'output_cluster_path']
            round_cols += ['tau_max']
        else:
            columns = ['timestamp', 'name', 'mu', 'avg_k', 'max_k',
                       'method', 'orig_clusters',
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

        metric = 'modularity' if method in ['louvain', 'leiden'] else 'nmi'

        for trial in tqdm(range(1, num_trials + 1), desc=f'{self.method!r}'):
            cluster_path = self.output_cluster_base / f'{self.name}_{self.method}_best_clusters.txt'

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
                                                         timestamp=timestamp,
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
        self.mapping = self._init_matrices()
        self.full_leakage_score = None
        return

    def _init_matrices(self):
        mapping = {u: i for i, u in enumerate(sorted(self.input_graph.nodes))}
        n, m = self.input_graph.order(), self.input_graph.size()

        # self.A = nx.to_scipy_sparse_array(self.input_graph, dtype='d', weight=1,  # unweighted graphs only
        #                                   nodelist=sorted(self.input_graph.nodes()))
        self.A = nx.to_numpy_array(self.input_graph, dtype='d', weight=1, nodelist=sorted(self.input_graph.nodes))

        self.d = self.A @ np.ones((n, 1))  # n x 1 degree vector
        self.D = np.diag(self.d.flatten())
        # self.D = scipy.sparse.diags(self.d.flatten())

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
                # I = scipy.sparse.eye(n, n)
                I = np.eye(n, n)
                # D_inv = scipy.sparse.diags((1 / self.d).flatten())
                D_inv = np.diag((1 / self.d).flatten())

                self.H = I - (self.A @ D_inv)
                self.H_prime = I - (self.A_prime @ D_inv)  # I_prime and D_inv_prime should be the same

            else:
                # D_inv = scipy.sparse.diags((1 / self.d).flatten())
                D_inv = np.diag((1 / self.d).flatten())
                self.H = self.A @ D_inv
                self.H_prime = self.A_prime @ D_inv
        else:
            raise Exception()

        self.tau_max = 1 / abs(
            scipy.sparse.linalg.eigs(self.H, k=1, return_eigenvectors=False)[0])  # to make sure it's non-negative

        # tau_diff is inverse of the second smallest non-zero eigenvalue of H
        # sorted_eigenvals = np.sort(np.abs(scipy.linalg.eig(self.H.toarray(), right=False)))
        sorted_eigenvals = np.sort(np.abs(scipy.linalg.eig(self.H, right=False)))
        second_smallest_eig = None
        for eig in sorted_eigenvals:
            if not np.isclose(eig, 0):
                second_smallest_eig = eig
                break
        assert second_smallest_eig is not None, f'second smallest eigenvalue is None!!'
        self.tau_diff = 1 / second_smallest_eig

        return mapping

    def generate_R_matrix(self, tau: float = None):
        H, H_prime = self.H, self.H_prime
        H_t, H_prime_t = self.H.transpose(), self.H_prime.transpose()

        if self.is_continuous:
            if self.exact:
                # U = scipy.sparse.linalg.expm(-tau * H)
                U = scipy.linalg.expm(-tau * H)
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
        best_clusterings = {}  # store the best clustering for each tau

        G_ = nx.relabel_nodes(self.input_graph, mapping=self.mapping)
        for tau_ix, tau in enumerate(taus):
            R = self.generate_R_matrix(tau=tau)

            if tau not in clusterings:
                clusterings[tau] = []
            method_ = f'{self.method}-C' if self.is_continuous else f'{self.method}-D'
            method = f'{method_}-exact' if self.exact else method_

            desc = f'Corr {method} ..'
            best_clustering = None
            best_score = None

            for trial in tqdm(range(1, num_trials + 1), desc=desc, leave=False):
                cluster_path = self.output_cluster_base / f'{self.name}_{method}_{tau_ix}_best_clusters.txt'

                if cluster_path.exists() and not overwrite:  # clustering already exists
                    # print(f'Existing clustering found {method!r} {self.name!r}')
                    continue

                tic = time.time()
                generated_clustering = self._run_method(R=R, G=G_)
                cluster_score = generated_clustering.method_parameters['leakage_score']

                time_taken = time.time() - tic
                generated_clustering.method_parameters.update({'method': method, 'tau': tau, 'tau_max': self.tau_max,
                                                               'tau_diff': self.tau_diff, 'trial': trial,
                                                               'running_time': time_taken,
                                                               'cluster_path': cluster_path})
                clusterings[tau].append(generated_clustering)

                # pick best ones
                if best_score is None:
                    best_score = cluster_score
                    best_clustering = generated_clustering

                elif best_score < cluster_score:
                    best_score = cluster_score
                    best_clustering = generated_clustering

            best_clusterings[tau] = best_clustering

            if write_stats and best_clustering is not None:
                mu = self.input_graph.graph.get('mu', np.nan)
                avg_k = self.input_graph.graph.get('avg_k', np.nan)
                max_k = self.input_graph.graph.get('max_k', np.nan)
                timestamp = datetime.now().strftime('%c')
                time_taken = best_clustering.method_parameters['running_time']
                cluster_path = best_clustering.method_parameters['cluster_path']
                leakage_score = best_score

                clustering_stats = self.get_clustering_stats(generated_clustering=best_clustering,
                                                             running_time=time_taken, tau=tau, tau_ix=tau_ix,
                                                             tau_max=self.tau_max, tau_diff=self.tau_diff,
                                                             mu=mu, avg_k=avg_k, max_k=max_k,
                                                             timestamp=timestamp, leakage_score=leakage_score,
                                                             output_cluster_path=cluster_path)
                clustering_stats['method'] = method  # methods with augmented 'D' or 'C"
                self.write_stats(stats_dict=clustering_stats, stats_path=stats_path)
                write_clustering(clustering=best_clustering, cluster_path=cluster_path)
        return clusterings, best_clusterings

    def _run_method(self, G, R):
        R = R * -1
        d = corr_leakage_partitions(G=G, R=R)
        q = deque(d, maxlen=1)
        clustering = q.pop()
        inv_mapping = {v: k for k, v in self.mapping.items()}
        mapped_cluster = []
        for cluster in clustering:
            c = [inv_mapping[x] for x in cluster]
            mapped_cluster.append(c)

        leakage_score = quality_score(partition=clustering, R=R)
        leakage_diff = round(self.full_leakage_score - leakage_score, 5)

        clustering = NodeClustering(graph=self.input_graph, method_name=self.method, communities=mapped_cluster,
                                    method_parameters={'full_leakage_score': self.full_leakage_score,
                                                       'leakage_score': leakage_score, 'leakage_diff': leakage_diff})

        return clustering

    def _run_method_old(self, R, try_merging=False):
        # store clusters as list of sets of nodes, flatten them out at the end of each iteration
        R = R * -1  # to maximize

        communities = [[{u}] for u in self.input_graph.nodes()]
        super_nodes = [{u} for u in self.input_graph.nodes()]  # list of flattened supernodes
        mapping = {u: i for i, u in enumerate(sorted(self.input_graph.nodes()))}
        random.shuffle(super_nodes)

        changed = True
        while changed:
            changed = False

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

                # use networkx cuts?
                for ix_2, d_j in enumerate(communities):
                    if s_i_comm_ix == ix_2:  # skip over s_i's own community
                        continue
                    for s_j in d_j:
                        if nx.cut_size(self.input_graph, s_i, s_j) == 0:
                            continue

                        gain = 0
                        for i in s_i:
                            for j in s_j:
                                gain += 2 * R[mapping[i], mapping[j]]

                        if gain > max_gain:
                            max_gain = gain
                            best_community_ix = ix_2

                change_in_quality = max_gain + loss
                if best_community_ix is not None and change_in_quality > 0:
                    changed = True
                    communities[best_community_ix].append(s_i)  # add s_i to its new community

                    # move s_i from its previous community to its new community  # TODO: change this to
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

    def _run_method_old_old(self, R, try_merging=False):
        # store clusters as list of sets of nodes, flatten them out at the end of each iteration
        R = R * -1  # to maximize

        communities = [[{u}] for u in self.input_graph.nodes()]
        super_nodes = [{u} for u in self.input_graph.nodes()]  # list of flattened supernodes
        supernode2comms = {u: i for i, u in
                           enumerate(self.input_graph.nodes())}  # mapping between super nodes and their indices
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
                # find neighboring super nodes of s_i
                # use networkx cuts?

                for ix_2, d_j in enumerate(communities):  # TODO: change this to only look at neighboring super nodes
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

    def _run_method_old_old_old(self, R, try_merging: bool = False):
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


def quality_score(partition, R):
    """
    Given a graph, a partition, and leakage matrix R, compute the quality function
    partition: list of sets of nodes
    """
    leakage_score = 0
    for comm in partition:
        for i, j in itertools.combinations_with_replacement(comm, 2):  # to account for diagonals
            leakage_score += 2 * R[i, j]
    leakage_score -= np.trace(R)  # to prevent diagonals from getting double counted
    return leakage_score


def shrink_R_matrix(comm, R):
    """
    Shrink the R matrix by summing over rows and columns of the old matrix R
    """
    # comm: node label -> {cluster of nodes}
    new_clusters = len(comm)  # graph has supernodes as nodes
    new_R = np.zeros((new_clusters, new_clusters))

    for i in range(new_clusters):
        comm_i = comm[i]
        for j in range(i, new_clusters):
            comm_j = comm[j]
            val = 0
            for x in comm_i:
                for y in comm_j:
                    val += R[x, y]

            new_R[i, j] = val
            new_R[j, i] = val

    return new_R


def _one_level(G, partition, R):
    """
    Calculate one level of the Louvain partitions tree
    """

    node2com = {u: i for i, u in enumerate(G.nodes())}
    inner_partition = [{u} for u in G.nodes()]
    nbrs = {u: {v: data["weight"] for v, data in G[u].items() if v != u} for u in G}
    rand_nodes = list(G.nodes)
    random.shuffle(rand_nodes)
    nb_moves = 1
    improvement = False
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            best_mod = 0
            best_com = node2com[u]

            remove_cost = 0
            for v in inner_partition[best_com]:
                if v == u:  # should we skip diagonals?
                    continue
                remove_cost -= 2 * R[u, v]

            weights2com = _neighbor_weights(nbrs[u], node2com)
            for nbr_com in weights2com:  # keys of this stores community neighbors of u
                gain = 0
                for v in inner_partition[nbr_com]:  # add if u == v
                    if u == v:
                        continue
                    gain += 2 * R[u, v]
                gain = remove_cost + gain
                if gain > best_mod:
                    best_mod = gain
                    best_com = nbr_com

            if best_com != node2com[u]:
                com = G.nodes[u].get("nodes", {u})
                # print(f'Merging {com} and {partition[best_com]}')
                partition[node2com[u]].difference_update(com)
                inner_partition[node2com[u]].remove(u)
                partition[best_com].update(com)
                inner_partition[best_com].add(u)
                improvement = True
                nb_moves += 1
                node2com[u] = best_com
    # print('------------')

    partition = list(filter(len, partition))
    inner_partition = list(filter(len, inner_partition))
    return partition, inner_partition, improvement


def corr_leakage_partitions(G, R, weight="weight", threshold=0.0000001):
    """
    Yields partitions for each level of the Louvain Community Detection Algorithm

    Louvain Community Detection Algorithm is a simple method to extract the community
    structure of a network. This is a heuristic method based on modularity optimization. [1]_

    The partitions at each level (step of the algorithm) form a dendogram of communities.
    A dendrogram is a diagram representing a tree and each level represents
    a partition of the G graph. The top level contains the smallest communities
    and as you traverse to the bottom of the tree the communities get bigger
    and the overal modularity increases making the partition better.

    Each level is generated by executing the two phases of the Louvain Community
    Detection Algorithm.
    """

    partition = [{u} for u in sorted(G.nodes)]  # should also be in sorted order

    graph = G.__class__()
    graph.add_nodes_from(G)
    graph.add_weighted_edges_from(G.edges(data=weight, default=1))
    graph = _gen_graph(graph, partition)

    mod = quality_score(partition=partition, R=R)
    partition, inner_partition, improvement = _one_level(G=graph, partition=partition, R=R)

    improvement = True
    while improvement:
        yield partition

        new_mod = quality_score(partition=inner_partition, R=R)
        if new_mod - mod <= threshold:
            return
        mod = new_mod
        R = shrink_R_matrix(comm=inner_partition, R=R)  # shrink the R matrix based on the partition
        graph = _gen_graph(graph, inner_partition)
        partition, inner_partition, improvement = _one_level(G=graph, partition=partition, R=R)

    return


def _neighbor_weights(nbrs, node2com):
    """Calculate weights between node and its neighbor communities.

    Parameters
    ----------
    nbrs : dictionary
           Dictionary with nodes' neighbours as keys and their edge weight as value.
    node2com : dictionary
           Dictionary with all graph's nodes as keys and their community index as value.

    """
    weights = defaultdict(float)
    for nbr, wt in nbrs.items():
        weights[node2com[nbr]] += wt
    return weights


def _gen_graph(G, partition):
    """Generate a new graph based on the partitions of a given graph"""
    H = G.__class__()
    node2com = {}
    for i, part in enumerate(partition):
        nodes = set()
        for node in part:
            node2com[node] = i
            nodes.update(G.nodes[node].get("nodes", {node}))
        H.add_node(i, nodes=nodes)

    for node1, node2, wt in G.edges(data=True):
        wt = wt["weight"]
        com1 = node2com[node1]
        com2 = node2com[node2]
        temp = H.get_edge_data(com1, com2, {"weight": 0})["weight"]
        H.add_edge(com1, com2, **{"weight": wt + temp})
    return H


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
    # TODO: get description length from infomap object

    paths = Paths(basedir='../')
    avg_k = 20
    mu = 0.1
    lfr = LFRGraph(paths=paths, n=100, mu=mu, avg_k=avg_k)
    g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=1)
    # sizes = [20, 20, 20]
    # probs = [[0.25, 0.02, 0.02], [0.02, 0.35, 0.02], [0.02, 0.02, 0.40]]

    # g = nx.stochastic_block_model(sizes, probs, seed=1)
    # g = nx.ring_of_cliques(51, clique_size=4)
    # g, g.name = nx.barbell_graph(4, 5), 'barbell'
    # ground_truth_clustering = NodeClustering(communities=[[0, 1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12]], graph=g)

    louvain_comms = nx.community.louvain_communities(g)

    method = 'GW'

    corr_cluster = CorrelationClustering(paths=paths, input_graph=g, continuous=True, method=method,
                                         ground_truth_clustering=ground_truth_clustering, exact=True)

    clusterings, best_clusterings = corr_cluster.run(num_trials=5, write_stats=True, overwrite=True)
    clusterings
