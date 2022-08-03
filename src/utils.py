"""
General utilities file
"""
import os
import warnings
from pathlib import Path
from typing import Union

import matplotlib
import networkx as nx
import numpy as np
from cdlib import NodeClustering
from cdlib.utils import convert_graph_formats
from cdlib.viz.networks import __filter, COLOR
from matplotlib import pyplot as plt


class Paths:
    def __init__(self, basedir='../'):
        self.basedir: Path = Path(basedir)
        self.data_dir: Path = self.basedir / 'data'
        self.results_dir: Path = self.basedir / 'results'
        return


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
        figsize: tuple = (8, 8),
        node_size: int = 200,
        plot_overlaps: bool = False,
        plot_labels: bool = False,
        cmap: object = None,
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
        fig, ax = plt.subplots(figsize=figsize)

    ax.axis('off')

    if title is None:
        modularity = partition.newman_girvan_modularity().score
        title = f'{graph.name!r} \t |V| = {graph.order():,} \t |E| = {graph.size():,}\n' \
                f'method = {partition.method_name!r} \t clusters = {len(partition.communities):,} \t modularity = {modularity:.4g}'

    title = title.replace('\t', '    ')
    ax.set_title(title, fontsize=40)

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

    return


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
