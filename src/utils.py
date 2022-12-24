"""
General utilities file
"""
import multiprocessing
import os
import platform
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Union

import clusim.sim as sim
import matplotlib
import networkx as nx
import numpy as np
from cdlib import NodeClustering
from cdlib.utils import convert_graph_formats
from cdlib.viz.networks import __filter, COLOR
from clusim.clustering import Clustering
from matplotlib import pyplot as plt


class Paths:
    def __init__(self, basedir='./'):
        if platform.system() == 'Darwin':
            self.basedir: Path = Path.home() / 'data-dump/corr-cluster'
        else:
            self.basedir: Path = Path('/N/slate/ssikdar/data-dump/corr-cluster')

        self.data_dir: Path = self.basedir / 'input'
        self.results_dir: Path = self.basedir / 'results_new_4'
        self.lfr_dir: Path = self.basedir / 'LFR'

        self.results_dir.mkdir(parents=True, exist_ok=True)
        return


def clusim_ec_score(clustering_1: NodeClustering, clustering_2: NodeClustering):
    """
    Use the Clusim package to calculate EC score
    """
    return clusim_similarity_score(clustering_1=clustering_1, clustering_2=clustering_2, metric='EC')


def clusim_similarity_score(clustering_1: NodeClustering, clustering_2: NodeClustering, metric: str):
    if isinstance(clustering_1, NodeClustering) and isinstance(clustering_2, NodeClustering):
        clu_dict1 = {i: clusters for i, clusters in enumerate(clustering_1.communities)}
        clu_dict2 = {i: clusters for i, clusters in enumerate(clustering_2.communities)}
        clu1, clu2 = Clustering(clu2elm_dict=clu_dict1), Clustering(clu2elm_dict=clu_dict2)

    elif isinstance(clustering_1, dict) and isinstance(clustering_2, dict):
        elm2dict_1 = {k: [v] for k, v in clustering_1.items()}
        elm2dict_2 = {k: [v] for k, v in clustering_2.items()}
        clu1, clu2 = Clustering(elm2clu_dict=elm2dict_1), Clustering(elm2clu_dict=elm2dict_2)

    metric = metric.upper()

    if metric == 'EC':
        fn = sim.element_sim
    elif metric == 'NMI':
        fn = sim.nmi
    elif metric == 'AMI':
        fn = sim.adj_mi
    else:
        raise NotImplementedError()

    return fn(clu1, clu2)


def parallel_async(func, args, num_workers=10, timeout=None):
    def update_result(result):
        return result

    results = []
    async_promises = []
    with Pool(num_workers) as pool:
        for arg in args:
            r = pool.apply_async(func, arg, callback=update_result)
            async_promises.append(r)
        for r in async_promises:
            try:
                r.wait(timeout=timeout)
                results.append(r.get(timeout=timeout))
            except (TimeoutError, multiprocessing.context.TimeoutError) as e:
                print(f'Timeout after {timeout}s {e=}')
            except Exception as e:
                try:
                    results.append(r.get(timeout=timeout))
                except (TimeoutError, multiprocessing.context.TimeoutError) as e:
                    print(f'Timeout after {timeout}s {e=}')

    return results


def remove_files(files):
    for f in files:
        if Path(f).exists():
            os.remove(f)
    return


def plot_network_clusters(
        graph: object,
        partition: NodeClustering,
        ax: plt.Axes = None,
        title: str = None,
        position: dict = None,
        figsize: tuple = (5, 5),
        node_size: int = 150,
        plot_overlaps: bool = False,
        plot_labels: bool = False,
        cmap: object = 'tab20',
        top_k: int = None,
        min_size: int = None,
        plot_edges: bool = True,
        alpha: float = 0.5,
) -> object:
    if not isinstance(cmap, (type(None), str, matplotlib.colors.Colormap)):
        raise TypeError(
            "The 'cmap' argument must be NoneType, str or matplotlib.colors.Colormap, "
            "not %s." % (type(cmap).__name__)
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='tight')

    # ax.axis('off')
    if partition.method_parameters is None:
        partition.method_parameters = {}

    if title is None:
        modularity = partition.newman_girvan_modularity().score
        if 'method' in partition.method_parameters:
            method = partition.method_parameters['method']
        else:
            method = partition.method_name
        title = f'{graph.name!r} \t |V| = {graph.order():,} \t |E| = {graph.size():,}\n' \
                f'method = {method!r} \t comms = {len(partition.communities):,} \t modularity = {modularity:.4g}'

        if 'tau' in partition.method_parameters:
            title += rf'   $\tau$: {partition.method_parameters["tau"]:.3g}'

        if 'leakage_diff' in partition.method_parameters:
            leakage_diff = partition.method_parameters['leakage_diff']
            full_leakage = partition.method_parameters['full_leakage_score']
            leakage_score = partition.method_parameters['leakage_score']
            title += f'\nLeakage:\tfull: {full_leakage:.4f}\tscore: {leakage_score:.4g}\tdiff:{leakage_diff:.4g}'

        title = title.replace('\t', '    ')
        ax.set_title(title, fontsize=10)

    partition = __filter(partition.communities, top_k, min_size)
    graph = convert_graph_formats(graph, nx.Graph)
    if position is None:
        position = nx.spring_layout(graph)

    n_communities = len(partition)
    if n_communities == 0:
        warnings.warn("There are no communities that match the filter criteria.")
        return None

    if cmap is None:
        n_communities = min(n_communities, len(COLOR))
        cmap = matplotlib.colors.ListedColormap(COLOR[:n_communities])
    else:
        cmap = plt.cm.get_cmap(cmap, n_communities)
    # _norm = matplotlib.colors.Normalize(vmin=0, vmax=n_communities - 1)
    _norm = lambda x: x
    fontcolors = list(
        map(
            lambda rgb: ".15" if np.dot(rgb, [0.2126, 0.7152, 0.0722]) > 0.408 else "w",
            [cmap(_norm(i))[:3] for i in range(n_communities)],
        )
    )

    filtered_nodelist = list(np.concatenate(partition))
    filtered_edgelist = list(
        filter(
            lambda edge: len(np.intersect1d(edge, filtered_nodelist)) == 2,
            graph.edges(),
        )
    )
    # fig = nx.draw_networkx_nodes(
    #     graph, position, node_size=node_size, node_color="w", nodelist=filtered_nodelist
    # )

    # draw inter and intra comm edges differently
    cluster_labels = {}
    for i, comm in enumerate(partition):
        for u in comm:
            cluster_labels[u] = i

    # draw intra comm edges first

    intra_comm_edges = {(u, v) for u, v in graph.edges() if cluster_labels[u] == cluster_labels[v]}
    inter_comm_edges = set(graph.edges()) - intra_comm_edges

    if plot_edges:
        nx.draw_networkx_edges(graph, position, alpha=alpha, edgelist=inter_comm_edges, edge_color='gray', ax=ax)
        nx.draw_networkx_edges(graph, position, alpha=alpha + 0.1, edgelist=intra_comm_edges, edge_color='black', ax=ax)

    if plot_labels:
        nx.draw_networkx_labels(
            graph,
            position,
            ax=ax,
            font_color=".8",
            labels={node: str(node) for node in filtered_nodelist},
        )
    for i in range(n_communities):
        if len(partition[i]) > 0:
            if plot_overlaps:
                size = (n_communities - i) * node_size
            else:
                size = node_size
            nx.draw_networkx_nodes(
                graph,
                position,
                ax=ax,
                node_size=size,
                nodelist=partition[i],
                node_color=[cmap(_norm(i))],
                margins=0,
                edgecolors='white',
                linewidths=2,
            )
            # fig.set_edgecolor("k")
    if plot_labels:
        for i in range(n_communities):
            if len(partition[i]) > 0:
                nx.draw_networkx_labels(
                    graph,
                    position,
                    font_color=fontcolors[i],
                    ax=ax,
                    labels={node: str(node) for node in partition[i]},
                )
    ax.set_xlim([1.1 * x for x in ax.get_xlim()])
    ax.set_ylim([1.1 * y for y in ax.get_ylim()])
    return ax


def write_clustering(clustering: NodeClustering, cluster_path: Union[str, Path]):
    """
    Write NodeClustering object to file: each line will have <node_label community_label>
    :return:
    """
    cluster_path.parent.mkdir(exist_ok=True, parents=True)

    with open(cluster_path, 'w') as writer:
        for i, comm in enumerate(clustering.communities):
            for u in comm:
                # node u has community label i
                writer.write(f'{u} {i}\n')
    return
