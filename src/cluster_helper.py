import itertools
import random
from collections import defaultdict

import igraph as ig
import infomap
import networkx as nx
import numpy as np
from cdlib import NodeClustering
from cdlib.utils import convert_graph_formats


def infomap_runner(g_original: object, flags: str = "") -> NodeClustering:
    im = infomap.Infomap(silent=True, two_level=False)
    mapping = im.add_networkx_graph(g_original)
    im.run()
    coms = {}

    for node in im.nodes:
        com_label = node.module_id
        if com_label not in coms:
            coms[com_label] = []
        coms[com_label].append(mapping[node.node_id])

    return NodeClustering(
        list(coms.values()), g_original, "Infomap", method_parameters={"flags": flags, 'codelength': im.getCodelength()}
    )


def _infomap(g_original: object, flags: str = "") -> NodeClustering:
    """
    Infomap is based on ideas of information theory.
    The algorithm uses the probability flow of random walks on a network as a proxy for information flows in the real system and it decomposes the network into modules by compressing a description of the probability flow.
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


def leiden(
        g_original: object, initial_membership: list = None, weights: list = None
) -> NodeClustering:
    """
    Modified implementation that uses igraph
    """
    ig_g = ig.Graph.from_networkx(g_original)
    leiden_comm = ig_g.community_leiden(objective_function='modularity')

    comms = {}
    for i, com_label in enumerate(leiden_comm.membership):
        if com_label not in comms:
            comms[com_label] = []
        comms[com_label].append(ig_g.vs[i]['_nx_name'])

    clustering = NodeClustering(communities=list(comms.values()), graph=g_original, method_name='Leiden_Q')
    return clustering


def quality_score(partition, R):
    """
    Given a graph, a partition, and leakage matrix R, compute the quality function
    partition: list of sets of nodes
    """
    leakage_score = 0
    for comm in partition:
        for i, j in itertools.combinations_with_replacement(comm, 2):  # to account for diagonals
            # leakage_score += 2 * R[i, j]
            leakage_score += (R[i, j] + R[j, i])
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
            val_1, val_2 = 0, 0
            for x in comm_i:
                for y in comm_j:
                    val_1 += R[x, y]
                    val_2 += R[y, x]

            new_R[i, j] = val_1
            new_R[j, i] = val_2

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
    # TODO: add stopping condition based on frac of nodes changing clusters in the current graph OR nodes in the original graph
    # TODO: nodes in the original graph is stored as "com"
    while nb_moves > 0:
        nb_moves = 0
        for u in rand_nodes:
            best_mod = 0
            best_com = node2com[u]

            remove_cost = 0
            for v in inner_partition[best_com]:
                if v == u:  # should we skip diagonals?
                    continue
                # remove_cost -= 2 * R[u, v]
                remove_cost -= (R[u, v] + R[v, u])  # R can be asymmetric

            weights2com = _neighbor_weights(nbrs[u], node2com)
            for nbr_com in weights2com:  # keys of this stores community neighbors of u
                gain = 0
                for v in inner_partition[nbr_com]:  # add if u == v
                    if u == v:
                        continue
                    # gain += 2 * R[u, v]
                    gain += (R[u, v] + R[v, u])  # R can be asymmetric
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
