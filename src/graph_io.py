"""
File for general graph io and synthetic graph generation
"""
import subprocess
from pathlib import Path
from typing import Tuple

import networkx as nx
from cdlib import NodeClustering
from cdlib.evaluation import normalized_mutual_information, adjusted_mutual_information
from matplotlib import pyplot as plt
from tqdm.auto import tqdm

from src.utils import remove_files, plot_network_clusters, Paths


class LFRGraph:
    """
    Container for LFR graphs - store both the graph object and the node community labels
    """

    def __init__(self, paths: Paths, n: int, mu: float, avg_k: int, family='LFR'):
        """
        n: number of nodes
        mu: mixing parameter
        avg_k: average degree
        """
        self.paths = paths
        self.n = n
        self.mu = round(mu, 2)
        self.avg_k = avg_k
        if n == 1_000:
            max_k = 50
        elif n == 10_000:
            max_k = 500
        self.max_k = max_k
        self.family = family
        self.name = f'{family}-{self.n // 1000}k-{max_k}-{avg_k}-{int(self.mu * 100)}'

        self.graph_basepath = self.paths.data_dir / family / f'{self.n // 1000}k-{max_k}/{avg_k}'
        # print(f'{self.graph_basepath=}')

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

        if self.family == 'LFR':
            if self.n == 1000:
                c_min, c_max = 10, 100  # hard
            elif self.n == 10_000:
                c_min, c_max = 100, 1000  # hard
        else:
            c_min, c_max = 25, 50  # easy

        self.paths.lfr_dir = Path('./LFR')
        self.graph_basepath.mkdir(exist_ok=True, parents=True)
        print(f'{self.graph_basepath=}')
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


def generate_lfr_like_sbm(n, cluster_size, mu, avg_k):
    n_c = cluster_size
    alpha = (mu / (1 - mu)) * (n_c / (n - n_c))
    p_in = avg_k / (n_c * (1 - alpha) + alpha * n)
    p_out = alpha * p_in

    num_blocks = n // cluster_size
    sizes = [cluster_size] * num_blocks

    # mu = p_out / (p_out + p_in)
    G = nx.random_partition_graph(sizes=sizes, p_in=p_in, p_out=p_out)
    G.name = f'SBM-{n}-{avg_k}-{cluster_size}-{mu}'
    ground_truth_comm_labels = nx.get_node_attributes(G, name='block')
    # ground_truth_clustering = NodeClustering(dict_to_list_of_lists(ground_truth_comm_labels), graph=G)

    return G, ground_truth_comm_labels


class SBMGraph:
    """
    Container for SBM graphs
    """

    def __init__(self, paths: Paths, n: int, mu: float, avg_k: int, cluster_size: int = 50, family='sbm'):
        """
        n: number of nodes
        mu: mixing parameter
        avg_k: average degree
        """
        self.paths = paths
        self.n = n
        self.mu = round(mu, 2)
        self.avg_k = avg_k
        self.cluster_size = cluster_size
        self.family = family
        self.name = f'{family}-{self.n // 1000}k-{cluster_size}-{avg_k}-{int(self.mu * 100)}'

        self.graph_basepath = self.paths.data_dir / family / f'{self.n // 1000}k-{cluster_size}/{avg_k}'
        return

    def generate_graphs(self, num_graphs: int, overwrite: bool = False):
        """
         Generate num_graphs graphs with the same parameters
        """
        avg_degree = self.avg_k

        self.graph_basepath.mkdir(exist_ok=True, parents=True)
        # print(f'{self.graph_basepath=}')
        for trial in tqdm(range(1, num_graphs + 1)):
            output_graph = self.graph_basepath / f'{self.name}_{trial}.gml'
            if output_graph.exists() and not overwrite:
                size = output_graph.stat().st_size / 1024  # size in KB
                if size > 100:
                    continue

            G, ground_truth_comm_labels = generate_lfr_like_sbm(n=self.n, cluster_size=self.cluster_size, mu=self.mu,
                                                                avg_k=avg_degree)
            del G.graph['partition']  # delete graph level attribute
            # nx.set_node_attributes(G, name='ground_truth', values=ground_truth_comm_labels)
            try:
                nx.write_gml(G, path=output_graph)
            except nx.NetworkXError as e:
                print(e)

        return

    def get_graph_and_clustering(self, trial: int):
        output_graph_path = self.graph_basepath / f'{self.name}_{trial}.gml'
        try:
            sbm_g = nx.read_gml(output_graph_path, destringizer=int)
        except Exception as e:
            print(f'{self.mu=} {trial=} {self.avg_k=} {e=}')
            exit(1)

        # set graph level attributes
        sbm_g.name = self.name + f'_{trial}'  # add trial number for output
        sbm_g.graph['mu'] = self.mu
        sbm_g.graph['avg_k'] = self.avg_k
        sbm_g.graph['cluster_size'] = self.cluster_size

        ground_truth_labels = nx.get_node_attributes(sbm_g, name='block')
        clusters_dict = {}  # key: community label, value: list of member nodes

        for u, c_u in ground_truth_labels.items():
            if c_u not in clusters_dict:
                clusters_dict[c_u] = []
            clusters_dict[c_u].append(u)

        clustering = NodeClustering(communities=list(clusters_dict.values()), graph=sbm_g,
                                    method_name='ground_truth')
        return sbm_g, clustering
