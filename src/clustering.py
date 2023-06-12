"""Container for clustering"""
import abc
import csv
import sys
import time
from collections import deque
from datetime import datetime
from typing import Dict

import cdlib
import networkx as nx
import numpy as np
import scipy.sparse
import scipy.sparse.linalg
from cdlib import NodeClustering, algorithms
from cdlib.evaluation import normalized_mutual_information, adjusted_mutual_information
from tqdm.auto import tqdm

sys.path.extend(['./', '../'])
from src.graph_io import SBMGraph
from src.utils import Paths, write_clustering, clusim_ec_score
from src.cluster_helper import corr_leakage_partitions, quality_score, leiden, infomap_runner


class BaseClustering:
    def __init__(self, paths: Paths, input_graph: nx.Graph,
                 ground_truth_clustering: NodeClustering, method: str, family: str = 'LFR'):
        self.paths = paths
        self.method = method
        self.input_graph = input_graph
        assert input_graph.name != '', f'Input graph name is blank.'
        self.name = input_graph.name
        self.ground_truth_clustering = ground_truth_clustering
        self.output_cluster_base = self.paths.results_dir / family / self.method
        return

    @abc.abstractmethod
    def _run_method(self, **params) -> NodeClustering:
        pass

    def write_stats(self, stats_dict, stats_path):
        file_exists = stats_path.exists()
        round_cols = ['mu', 'running_time', 'modularity', 'NMI', 'EC', 'AMI']

        class_name = self.__class__.__name__

        # if isinstance(self, CorrelationClustering) or isinstance(self, MarkovClustering):
        if class_name in ['CorrelationClustering', 'MarkovClustering', 'MultiLouvainClustering']:
            columns = ['timestamp', 'name', 'mu', 'avg_k', 'max_k',
                       'method', 'tau_max', 'tau_diff', 'tau_ix', 'tau', 'orig_clusters', 'running_time',
                       'metric', 'score',
                       'num_clusters', 'modularity', 'NMI', 'AMI', 'EC', 'output_cluster_path']
            round_cols += ['tau_max']
        else:
            columns = ['timestamp', 'name', 'mu', 'avg_k', 'max_k',
                       'method', 'orig_clusters', 'metric', 'score',
                       'running_time', 'num_clusters', 'modularity', 'NMI', 'AMI', 'EC', 'output_cluster_path']

        for col in round_cols:
            if stats_dict[col] is not None:
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
        if self.method in ['louvain', 'leiden']:
            metric = 'modularity'
        elif self.method == 'infomap':
            metric = 'codelength'
        else:
            metric = 'nmi'
        cluster_path = self.output_cluster_base / f'{self.name}_{self.method}_best_clusters.txt'
        best_clustering = None  # store the best clustering for each trial
        best_score = -1
        best_clustering_stats = None

        for trial in tqdm(range(1, num_trials + 1), desc=f'{self.method!r}'):
            if cluster_path.exists() and not overwrite:  # clustering already exists
                # print(f'Existing clustering found {self.method!r} {self.name!r} {trial}')
                continue

            tic = time.time()
            generated_clustering = self._run_method()
            time_taken = time.time() - tic

            if metric == 'modularity':
                score = generated_clustering.newman_girvan_modularity().score
            elif metric == 'codelength':
                score = generated_clustering.method_parameters[metric]
            else:
                score = normalized_mutual_information(self.ground_truth_clustering, generated_clustering).score

            mu = self.input_graph.graph.get('mu')
            avg_k = self.input_graph.graph.get('avg_k')
            max_k = self.input_graph.graph.get('max_k')
            timestamp = datetime.now().strftime('%c')

            clustering_stats = self.get_clustering_stats(generated_clustering=generated_clustering,
                                                         running_time=time_taken, mu=mu, avg_k=avg_k, max_k=max_k,
                                                         timestamp=timestamp, score=score, metric=metric,
                                                         output_cluster_path=cluster_path)

            if score > best_score:
                best_score = score
                best_clustering = generated_clustering
                best_clustering_stats = clustering_stats

            # self.write_stats(stats_dict=clustering_stats, stats_path=stats_path)
        # only write the best clusterings
        if best_clustering is not None:
            write_clustering(clustering=best_clustering, cluster_path=cluster_path)
            self.write_stats(stats_dict=best_clustering_stats, stats_path=stats_path)

        return best_clustering

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

    _METHODS = ['GW', 'RW', 'Diff', 'RW',
                'CO', 'MultiLouvain']  # GW=GraphWalk, RW=RandomWalk, Diff=Diffusion, CO=Consensus

    def __init__(self, method: str, continuous: bool, paths: Paths, input_graph: nx.Graph,
                 ground_truth_clustering: NodeClustering, erdos_renyi_null: bool = False,
                 family: str = 'LFR', inertia_factor: int = 0, inertia_type: str = 'v1'):

        super().__init__(paths=paths, input_graph=input_graph, ground_truth_clustering=ground_truth_clustering,
                         method=method, family=family)
        assert method in CorrelationClustering._METHODS, f'Incorrect {method!r} pick from {CorrelationClustering._METHODS}'
        self.method = method
        self.erdos_renyi_null = erdos_renyi_null
        assert isinstance(continuous, bool)
        self.is_continuous = continuous  # true if continuous
        self.inertia_factor = inertia_factor  # factor of things to diagonal
        self.inertia_type = inertia_type
        self.full_leakage_score = None
        self.mapping = self._init_matrices()

        return

    def generate_taus(self):
        """
        Return a list of taus
        """
        # taus_1 = np.geomspace(np.finfo(float).eps, self.tau_max, num=5)
        # taus_2 = np.geomspace(self.tau_max, self.tau_diff, num=5)[1:]  # prevent tau_max from getting double counted

        # taus_1 = np.linspace(np.finfo(float).eps, self.tau_max, num=5)
        taus_1 = np.linspace(np.finfo(float).eps, self.tau_max, num=10)  # twice as granular

        # taus_2 = np.linspace(self.tau_max, self.tau_diff, num=10)[1:]  # prevent tau_max from getting double counted
        taus_2 = np.geomspace(self.tau_max, self.tau_diff, num=10)[1:]  # prevent tau_max from getting double counted

        taus = np.concatenate([taus_1, taus_2])

        return taus

    def _init_matrices(self):
        # numpy is a lot faster than scipy sparse

        mapping = {u: i for i, u in enumerate(sorted(self.input_graph.nodes))}
        # n, m = self.input_graph.order(), self.input_graph.size()
        # self.A = nx.to_scipy_sparse_array(self.input_graph, dtype='d', weight=1,  # unweighted graphs only
        #                                   nodelist=sorted(self.input_graph.nodes()))
        # self.A = nx.to_numpy_array(self.input_graph, dtype='d', weight=1, nodelist=sorted(self.input_graph.nodes))

        self.A = nx.to_numpy_array(self.input_graph, dtype='d', weight='weight',
                                   nodelist=sorted(self.input_graph.nodes))

        n = self.A.shape[0]

        for isolated_node in nx.isolates(self.input_graph):
            self.A[mapping[isolated_node], mapping[isolated_node]] = 1

        if self.inertia_factor != 0:  # adjust the A matrix
            if self.method == 'Diff':
                # self.A = self.A + (self.inertia_factor * np.eye(n, n))
                d = self.A @ np.ones((n, 1))
                D = np.diag(d.flatten())
                self.A = self.A + (self.inertia_factor * D)
            elif self.method == 'RW':
                if self.inertia_type == 'v1':
                    d = self.A @ np.ones((n, 1))
                    D = np.diag(d.flatten())
                    self.A = self.A + (self.inertia_factor * D)
                elif self.inertia_type == 'v2':
                    self.A = self.A + (self.inertia_factor * np.eye(n, n))
            elif self.method == 'CO':
                raise NotImplementedError(f'Undefined for {self.method}!')

        m = np.sum(self.A) / 2 + np.sum(np.diag(self.A)) / 2  # diagonals otherwise get counted 1/2 as much

        self.d = self.A @ np.ones((n, 1))  # n x 1 degree vector
        self.D = np.diag(self.d.flatten())

        if self.erdos_renyi_null:
            self.A_prime = np.ones((n, n)) * (2 * m / (n * (n - 1)))
            self.d_prime = self.A_prime @ np.ones((n, 1))  # n x 1 degree vector
            self.D_prime = np.diag(self.d_prime.flatten())

        else:  # configuration model null model
            self.A_prime = (self.d @ np.transpose(self.d)) / (2 * m)  # null model
            self.d_prime = self.A_prime @ np.ones((n, 1))  # n x 1 degree vector
            self.D_prime = np.diag(self.d_prime.flatten())

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
                self.H_prime = self.D_prime - self.A_prime  # D_prime should equal D
            else:
                raise NotImplementedError()

        elif self.method == 'RW':
            if self.is_continuous:
                n = self.input_graph.order()
                # I = scipy.sparse.eye(n, n)
                I = np.eye(n, n)
                # D_inv = scipy.sparse.diags((1 / self.d).flatten())
                D_inv = np.diag((1 / self.d).flatten())
                D_inv_prime = np.diag((1 / self.d_prime).flatten())

                self.H = I - (self.A @ D_inv)
                self.H_prime = I - (self.A_prime @ D_inv_prime)  # I_prime and D_inv_prime should be the same

            else:
                # D_inv = scipy.sparse.diags((1 / self.d).flatten())
                D_inv = np.diag((1 / self.d).flatten())
                D_inv_prime = np.diag((1 / self.d_prime).flatten())

                self.H = self.A @ D_inv
                self.H_prime = self.A_prime @ D_inv_prime

        elif self.method == 'CO':  # consensus
            if self.is_continuous:
                n = self.input_graph.order()
                # I = scipy.sparse.eye(n, n)
                I = np.eye(n, n)
                # D_inv = scipy.sparse.diags((1 / self.d).flatten())
                D_inv = np.diag((1 / self.d).flatten())
                D_inv_prime = np.diag((1 / self.d_prime).flatten())

                self.H = I - (D_inv @ self.A)
                self.H_prime = I - (D_inv_prime @ self.A_prime)  # I_prime and D_inv_prime should be the same
            else:
                raise NotImplementedError()
        else:
            raise Exception()

        self.tau_max = 1 / abs(
            scipy.sparse.linalg.eigs(self.H, k=1, return_eigenvectors=False)[0])  # to make sure it's non-negative

        # tau_diff is inverse of the second smallest non-zero eigenvalue of H

        ### SLOW because it unnecessarily computes the whole spectrum
        sorted_eigenvals = np.sort(np.abs(scipy.linalg.eig(self.H, right=False)))
        second_smallest_eig = None
        for eig in sorted_eigenvals:
            if not np.isclose(eig, 0):
                second_smallest_eig = eig
                break
        assert second_smallest_eig is not None, f'second smallest eigenvalue is None!!'

        ##  new add - Jun 12 - dont compute whole spectrum - pick the 2nd largest value around 0
        # second_smallest_eig = np.real(scipy.sparse.linalg.eigs(self.H, k=2, sigma=0, return_eigenvectors=False))[0]
        # assert not np.isclose(second_smallest_eig, 0), f'Second smallest eigenvalue is 0'
        # assert second_smallest_eig > 0, f'Second smallest eigenvalue is negative'
        # assert np.isclose(second_smallest_eig, second_smallest_eig_old), f'Second smallest: numpy {second_smallest_eig:.5f} != scipy: {second_smallest_eig_2:.5f}'

        self.tau_diff = 1 / second_smallest_eig

        return mapping

    def generate_R_matrix(self, tau: float = None):
        H, H_prime = self.H, self.H_prime
        H_t, H_prime_t = self.H.transpose(), self.H_prime.transpose()

        if self.is_continuous:
            # U = scipy.sparse.linalg.expm(-tau * H)
            U = scipy.linalg.expm(-tau * H)
            U = U @ U.transpose()

            if np.isnan(U).any():
                print(f'U matrix has NaNs!!')
                exit(1)

            R = H @ U + U @ H_t - H_prime @ U - U @ H_prime_t

        else:  # discrete method
            R = -H @ H_t + H_prime @ H_prime_t

        if np.isnan(R).any():
            mu = self.input_graph.graph.get('mu', np.nan)
            print(f'R matrix has NaNs for {self.method=} {mu=} {tau=}!!')
            exit(1)

        R = R * -1  # to maximize inside
        self.full_leakage_score = np.sum(R)  # every node is in its own community
        # assert not np.isclose(self.full_leakage_score, 0), f'Leakage score is 0 for {self.method} {tau=:.4g}'
        return R

    def run(self, num_trials: int, overwrite: bool, write_stats: bool = True, try_merging: bool = False):
        # running the method num_trials many times
        class_name = self.__class__.__name__
        if class_name == 'CorrelationClustering':
            stats_path = self.paths.results_dir / 'stats_CL.csv'
        elif class_name == 'MarkovClustering':
            stats_path = self.paths.results_dir / 'stats_MC.csv'
        elif class_name == 'MultiLouvainClustering':
            stats_path = self.paths.results_dir / 'stats_ML.csv'
        else:
            raise NotImplementedError(f'Invalid stats_path error {self.__name__}')

        if not self.is_continuous:
            taus = [-1]
        else:
            taus = self.generate_taus()

        clusterings = {}
        best_clusterings = {}  # store the best clustering for each tau

        G_ = nx.relabel_nodes(self.input_graph, mapping=self.mapping)
        # if self.__class__.__name__ == 'CorrelationClustering' and self.inertia_factor != 0:
        #     taus = [0]

        for tau_ix, tau in enumerate(taus):
            R = self.generate_R_matrix(tau=tau)

            if tau not in clusterings:
                clusterings[tau] = []
            method_ = f'{self.method}-C' if self.is_continuous else f'{self.method}-D'
            if self.__class__.__name__ == "MarkovClustering":
                method = f'MC-{method_}'
            elif self.__class__.__name__ == 'MultiLouvainClustering':
                method = f'MultiLouvain'
            else:
                method = method_

            if self.__class__.__name__ == 'CorrelationClustering':
                if self.inertia_factor != 0:
                    if self.inertia_type == 'v2':
                        method_ = f'{method_}-{self.inertia_factor}-v2'
                    else:
                        method_ = f'{method_}-{self.inertia_factor}'
                    if self.method == 'Diff':
                        method_ = f'{method_}-new'  # for newly adjusted inertia

                if self.erdos_renyi_null:
                    method = f'{method_}-ER'
                else:
                    method = method_

            mu = self.input_graph.graph.get('mu', np.nan)
            self.input_graph.graph.get('avg_k', np.nan)
            desc = f'{self.__class__.__name__[: 5]} {method} {mu=} {tau_ix=} ..'
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
                                                             timestamp=timestamp,  # leakage_score=leakage_score,
                                                             metric='leakage_score', score=leakage_score,
                                                             output_cluster_path=cluster_path)
                clustering_stats['method'] = method  # methods with augmented 'D' or 'C"
                self.write_stats(stats_dict=clustering_stats, stats_path=stats_path)
                write_clustering(clustering=best_clustering, cluster_path=cluster_path)
        return clusterings, best_clusterings

    def _run_method(self, G, R):
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


class MarkovClustering(CorrelationClustering):
    _METHODS = ['Diff', 'RW', 'MultiLouvain']

    def __init__(self, method: str, continuous: bool, paths: Paths, input_graph: nx.Graph,
                 ground_truth_clustering: NodeClustering, family: str = 'LFR'):

        super().__init__(paths=paths, input_graph=input_graph, ground_truth_clustering=ground_truth_clustering,
                         method=method, continuous=continuous, family=family)
        assert method in MarkovClustering._METHODS, f'Incorrect {method!r} pick from {MarkovClustering._METHODS}'
        self.method = method

        assert isinstance(continuous, bool)
        self.is_continuous = continuous  # true if continuous
        self.mapping = self._init_matrices()
        return

    def _init_matrices(self):
        # numpy is a lot faster than scipy sparse
        mapping = {u: i for i, u in enumerate(sorted(self.input_graph.nodes))}
        # n, m = self.input_graph.order(), self.input_graph.size()

        # self.A = nx.to_scipy_sparse_array(self.input_graph, dtype='d', weight=1,  # unweighted graphs only
        #                                   nodelist=sorted(self.input_graph.nodes()))
        # self.A = nx.to_numpy_array(self.input_graph, dtype='d', weight=1, nodelist=sorted(self.input_graph.nodes))

        self.A = nx.to_numpy_array(self.input_graph, dtype='d', weight='weight',
                                   nodelist=sorted(self.input_graph.nodes))
        for isolated_node in nx.isolates(self.input_graph):
            self.A[mapping[isolated_node], mapping[isolated_node]] = 1

        n, m = self.A.shape[0], np.sum(self.A) / 2

        self.d = self.A @ np.ones((n, 1))  # n x 1 degree vector
        self.D = np.diag(self.d.flatten())
        # self.D = scipy.sparse.diags(self.d.flatten())

        self.A_prime = (self.d @ np.transpose(self.d)) / (2 * m)  # null model

        if self.method == 'Diff':
            if self.is_continuous:
                self.H = self.D - self.A
                self.Pi = np.eye(n) / n
                self.Pi_steady = np.ones((n, n)) / n ** 2
            else:
                raise NotImplementedError()

        elif self.method == 'RW':
            if self.is_continuous:
                I = np.eye(n, n)
                D_inv = np.diag((1 / self.d).flatten())
                self.H = I - (self.A @ D_inv)

                self.Pi = self.D / (2 * m)
                self.Pi_steady = self.A_prime / (2 * m)
        else:
            raise Exception()

        self.tau_max = 1 / abs(
            scipy.sparse.linalg.eigs(self.H, k=1, return_eigenvectors=False)[0])  # to make sure it's non-negative

        # tau_diff is inverse of the second smallest non-zero eigenvalue of H

        sorted_eigenvals = np.sort(np.abs(scipy.linalg.eig(self.H, right=False)))
        second_smallest_eig = None
        for eig in sorted_eigenvals:
            if not np.isclose(eig, 0):
                second_smallest_eig = eig
                break
        assert second_smallest_eig is not None, f'second smallest eigenvalue is None!!'
        # eigs = np.abs(scipy.linalg.eig(self.H, k=2, sigma=0, which='LM', return_eigenvectors=False))
        # second_smallest_eig = eigs[0]
        #
        # print('\nSorted eigenvalues....')
        # for i, eig in enumerate(sorted_eigenvals[: 5]):
        #     print(f'{i=} {eig=}')
        # print(f'Scipy {eigs=}')
        # print('----------\n')
        #
        # assert not np.isclose(second_smallest_eig, 0), f'Second smallest eigenvalue {second_smallest_eig} is 0. {self.method=} {self.input_graph.name}'
        # assert second_smallest_eig > 0, f'Second smallest eigenvalue is negative'
        self.tau_diff = 1 / second_smallest_eig

        return mapping

    def generate_R_matrix(self, tau: float = None):
        """
        Construct the R matrix for the different methods
        """

        if self.is_continuous:
            R = (scipy.linalg.expm(-tau * self.H) @ self.Pi) - self.Pi_steady
        self.full_leakage_score = np.sum(R)  # every node is in its own community
        return R


class MultiLouvainClustering(MarkovClustering):
    def __init__(self, paths: Paths, input_graph: nx.Graph,
                 ground_truth_clustering: NodeClustering, method=f'MultiLouvain', continuous: bool = True,
                 family: str = 'LFR'):

        super().__init__(paths=paths, input_graph=input_graph, ground_truth_clustering=ground_truth_clustering,
                         method=method, family=family, continuous=True)
        self.method = method

        assert isinstance(continuous, bool)
        self.is_continuous = continuous  # true if continuous
        self.mapping = self._init_matrices()
        return

    def _init_matrices(self):
        # numpy is a lot faster than scipy sparse
        mapping = {u: i for i, u in enumerate(sorted(self.input_graph.nodes))}
        # n, m = self.input_graph.order(), self.input_graph.size()

        # self.A = nx.to_scipy_sparse_array(self.input_graph, dtype='d', weight=1,  # unweighted graphs only
        #                                   nodelist=sorted(self.input_graph.nodes()))
        # self.A = nx.to_numpy_array(self.input_graph, dtype='d', weight=1, nodelist=sorted(self.input_graph.nodes))

        self.A = nx.to_numpy_array(self.input_graph, dtype='d', weight='weight',
                                   nodelist=sorted(self.input_graph.nodes))
        for isolated_node in nx.isolates(self.input_graph):
            self.A[mapping[isolated_node], mapping[isolated_node]] = 1

        n, m = self.A.shape[0], np.sum(self.A) / 2

        self.d = self.A @ np.ones((n, 1))  # n x 1 degree vector
        self.D = np.diag(self.d.flatten())
        # self.D = scipy.sparse.diags(self.d.flatten())

        self.A_prime = (self.d @ np.transpose(self.d)) / (2 * m)  # null model

        if self.is_continuous:
            I = np.eye(n, n)
            D_inv = np.diag((1 / self.d).flatten())
            self.H = I - (self.A @ D_inv)

            self.Pi = self.D / (2 * m)
            self.Pi_steady = self.A_prime / (2 * m)

        self.tau_max = 1 / abs(
            scipy.sparse.linalg.eigs(self.H, k=1, return_eigenvectors=False)[0])  # to make sure it's non-negative

        # tau_diff is inverse of the second smallest non-zero eigenvalue of H
        sorted_eigenvals = np.sort(np.abs(scipy.linalg.eig(self.H, right=False)))
        second_smallest_eig = None
        for eig in sorted_eigenvals:
            if not np.isclose(eig, 0):
                second_smallest_eig = eig
                break
        assert second_smallest_eig is not None, f'second smallest eigenvalue is None!!'

        # second_smallest_eig = np.real(scipy.sparse.linalg.eigs(self.H, k=2, sigma=0, return_eigenvectors=False))[0]
        # assert not np.isclose(second_smallest_eig, 0), f'Second smallest eigenvalue is 0'
        # # assert second_smallest_eig > 0, f'Second smallest eigenvalue is negative'
        self.tau_diff = 1 / second_smallest_eig

        return mapping

    def generate_R_matrix(self, tau: float = None):
        """
        Construct the R matrix for the different methods
        """
        # R = (self.Pi @ scipy.linalg.expm(-tau * self.H)) - self.Pi_steady
        if np.isclose(tau, 0):
            tau = np.finfo(float).eps

        R = self.A - (self.A_prime * 1 / tau)

        self.full_leakage_score = np.sum(R)  # every node is in its own community
        return R


class OtherClustering(BaseClustering):
    _METHODS = ['louvain', 'greedy_modularity', 'label_propagation', 'infomap', 'leiden', 'eigenvector',
                'sbm_dl', 'sbm_dl_nested']

    def __init__(self, paths: Paths, input_graph: nx.Graph, method: str, ground_truth_clustering: NodeClustering,
                 family: str = 'LFR'):
        super().__init__(paths=paths, input_graph=input_graph, method=method, family=family,
                         ground_truth_clustering=ground_truth_clustering)
        self.method = method
        assert method in OtherClustering._METHODS, f'Incorrect method: {method!r}, pick from {OtherClustering._METHODS}'
        return

    def _run_method(self, **params) -> NodeClustering:
        if self.method == 'leiden':
            clustering = leiden(g_original=self.input_graph)
        elif self.method == 'infomap':
            clustering = infomap_runner(g_original=self.input_graph)
        else:
            cdlib_alg = getattr(cdlib.algorithms, self.method)
            clustering = cdlib_alg(self.input_graph)
        return clustering


if __name__ == '__main__':
    paths = Paths(basedir='../')
    avg_k = 15
    mu = 0.05
    # for trial in range(1, 20):
    # lfr = LFRGraph(paths=paths, n=1_000, mu=mu, avg_k=avg_k)
    # g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)

    sbm = SBMGraph(paths=paths, n=1_000, mu=0.05, avg_k=10)
    g, ground_truth_clustering = sbm.get_graph_and_clustering(trial=2)
    print(nx.info(g))

    method = 'Diff'
    inertia_factor = 0.5
    er_null_model = False
    markov_clustering = MarkovClustering(paths=paths, input_graph=g, continuous=True, method='RW',
                                         ground_truth_clustering=ground_truth_clustering)
    markov_clustering.run(num_trials=10, overwrite=True)
    # corr_cluster = CorrelationClustering(paths=paths, input_graph=g, continuous=True, method=method,
    #                                      ground_truth_clustering=ground_truth_clustering,
    #                                      inertia_factor=inertia_factor,
    #                                      erdos_renyi_null=er_null_model, inertia_type='v1')
    # break

    # clusterings, best_clusterings = corr_cluster.run(num_trials=3, write_stats=True, overwrite=True)
    # clusterings
    #
    # # g, g.name = nx.barbell_graph(4, 5), 'barbell'
    # # ground_truth_clustering = NodeClustering(communities=[[0, 1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12]], graph=g)
    #
    # # method = 'label_propagation'
    # # other_clustering = OtherClustering(paths=paths, input_graph=g, ground_truth_clustering=ground_truth_clustering,
    # #                                    method=method)
    # # other_clustering.run(num_trials=1, overwrite=True)
    # #
    # # cdlib.algorithms.greedy_modularity()
    # sizes = [30, 30, 30]
    # probs = [
    #     [0.35, 0.05, 0.05],
    #     [0.05, 0.35, 0.05],
    #     [0.05, 0.05, 0.35]
    # ]
    #
    # g = nx.stochastic_block_model(sizes, probs, seed=1)
    # ground_truth_clustering = NodeClustering(nx.community.louvain_communities(g), graph=g)
    #
    # # g, g.name = nx.ring_of_cliques(51, clique_size=4), 'clique-ring-51'
    # # g, g.name = nx.barbell_graph(4, 5), 'barbell'
    # #
    # # ground_truth_clustering = NodeClustering(communities=[[0, 1, 2, 3], [4, 5, 6, 7, 8], [9, 10, 11, 12]], graph=g)
    #
    # # ground_truth_clustering = NodeClustering(nx.community.louvain_communities(g), graph=g)
    #
    # method = 'RW'
    # inertia_factor = 0.5
    # er_null_model = False
    # corr_cluster = CorrelationClustering(paths=paths, input_graph=g, continuous=True, method=method,
    #                                      ground_truth_clustering=ground_truth_clustering, inertia_factor=inertia_factor,
    #                                      erdos_renyi_null=er_null_model, inertia_type='v1')
    #
    # clusterings, best_clusterings = corr_cluster.run(num_trials=1, write_stats=True, overwrite=True)
    # clusterings
    #
    # # method = 'RW'
    # # markov_cluster = MarkovClustering(paths=paths, input_graph=g, continuous=True, method=method,
    # #                                   ground_truth_clustering=ground_truth_clustering)
    # # clusterings, best_clusterings = markov_cluster.run(num_trials=5, overwrite=True)
    # # clusterings
    #
    # # louvain_cluster = MultiLouvainClustering(paths=paths, input_graph=g,
    # #                                          ground_truth_clustering=ground_truth_clustering)
    # # clusterings, best_clusterings = louvain_cluster.run(num_trials=5, overwrite=True)
    # # best_clusterings
