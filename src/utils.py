"""
General utilities file
"""
import os
import platform
import warnings
from multiprocessing import Pool
from pathlib import Path
from typing import Union

import matplotlib
import networkx as nx
import numpy as np
from cdlib import NodeClustering
from cdlib.utils import convert_graph_formats
from cdlib.viz.networks import __filter, COLOR
from clusim.clusimelement import element_sim
from clusim.clustering import Clustering
from matplotlib import pyplot as plt


class Paths:
    def __init__(self, basedir='./'):
        if platform.system() == 'Darwin':
            self.basedir: Path = Path.home() / 'data-dump/corr-cluster'
        else:
            self.basedir: Path = Path('/N/slate/ssikdar/data-dump/corr-cluster')

        self.data_dir: Path = self.basedir / 'input'
        self.results_dir: Path = self.basedir / 'results_new'
        self.lfr_dir: Path = self.basedir / 'LFR'

        self.results_dir.mkdir(parents=True, exist_ok=True)
        return


def clusim_ec_score(clustering_1: NodeClustering, clustering_2: NodeClustering):
    """
    Use the Clusim package to calculate EC score
    """
    clu_dict1 = {i: clusters for i, clusters in enumerate(clustering_1.communities)}
    clu_dict2 = {i: clusters for i, clusters in enumerate(clustering_2.communities)}

    return element_sim(Clustering(clu2elm_dict=clu_dict1), Clustering(clu2elm_dict=clu_dict2))


def parallel_async(func, args, num_workers=10):
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
                r.wait()
                results.append(r.get())
            except Exception as e:
                results.append(r.get())

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
) -> object:
    """
    Plot a graph with node color coding for communities.

    :param graph: NetworkX/igraph graph
    :param partition: NodeClustering object
    :param position: A dictionary with nodes as keys and positions as values. Example: networkx.fruchterman_reingold_layout(G). By default, uses nx.spring_layout(g)
    :param figsize: the figure size; it is a pair of float, default (8, 8)
    :param node_size: int, default 200
    :param plot_overlaps: bool, default False. Flag to control if multiple algorithms memberships are plotted.
    :param plot_labels: bool, default False. Flag to control if node labels are plotted.
    :param cmap: str or Matplotlib colormap, Colormap(Matplotlib colormap) for mapping intensities of nodes. If set to None, original colormap is used.
    :param top_k: int, Show the top K influential communities. If set to zero or negative value indicates all.
    :param min_size: int, Exclude communities below the specified minimum size.

    Example:

    >>> from cdlib import algorithms, viz
    >>> import networkx as nx
    >>> g = nx.karate_club_graph()
    >>> coms = algorithms.louvain(g)
    >>> pos = nx.spring_layout(g)
    >>> viz.plot_network_clusters(g, coms, pos)
    """
    if not isinstance(cmap, (type(None), str, matplotlib.colors.Colormap)):
        raise TypeError(
            "The 'cmap' argument must be NoneType, str or matplotlib.colors.Colormap, "
            "not %s." % (type(cmap).__name__)
        )

    if ax is None:
        fig, ax = plt.subplots(figsize=figsize, layout='tight')

    # ax.axis('off')

    if title is None:
        modularity = partition.newman_girvan_modularity().score
        if 'method' in partition.method_parameters:
            method = partition.method_parameters['method']
        else:
            method = partition.method_name
        title = f'{graph.name!r} \t |V| = {graph.order():,} \t |E| = {graph.size():,}\n' \
                f'method = {method!r} \t clusters = {len(partition.communities):,} \t modularity = {modularity:.4g}'

        if 'leakage_diff' in partition.method_parameters:
            leakage_diff = partition.method_parameters['leakage_diff']
            full_leakage = partition.method_parameters['full_leakage_score']
            leakage_score = partition.method_parameters['leakage_score']

            title += f'\nLeakage:\tfull: {full_leakage:.4g}\tscore: {leakage_score:.4g}\tdiff:{leakage_diff:.4g}'

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
    _norm = matplotlib.colors.Normalize(vmin=0, vmax=n_communities - 1)
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

    nx.draw_networkx_edges(graph, position, alpha=0.5, edgelist=inter_comm_edges, edge_color='gray', ax=ax)
    nx.draw_networkx_edges(graph, position, alpha=0.6, edgelist=intra_comm_edges, edge_color='black', ax=ax)

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
