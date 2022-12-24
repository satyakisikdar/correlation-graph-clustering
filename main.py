"""Main driver"""
import random
import subprocess
import sys
from pathlib import Path

import graph_tool.all as gt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.extend(['./', '../'])
from src.clustering import CorrelationClustering, OtherClustering, MarkovClustering, MultiLouvainClustering
from src.utils import Paths, remove_files, parallel_async
from src.graph_io import LFRGraph


def generate_LFR(paths: Paths, LFR_source: str = './LFR', n: int = 1000, mu: float = 0.25, overwrite: bool = False):
    tau1, tau2 = 2, 3
    avg_degree = 20
    max_degree = 50 if n == 1000 else 200
    c_min = 10
    c_max = 100 if n == 1000 else 1_000

    name = f'LFR-{n}-{int(mu * 100)}'
    (paths.data_dir / 'LFR').mkdir(exist_ok=True)  # ensure the dir exists

    input_graph_file, output_graph_file = Path('./network.dat'), paths.data_dir / 'LFR' / f'{name}_edgelist.txt'
    input_cluster_file, output_cluster_file = Path('./community.dat'), paths.data_dir / 'LFR' / f'{name}_clusters.txt'

    if output_graph_file.exists() and output_cluster_file.exists() and not overwrite:
        print(f'Graph already exists! {str(output_graph_file)}')
        return

    args = f'./{LFR_source}/benchmark -N {n} -k {avg_degree} -maxk {max_degree} -mu {mu} -t1 {tau1} -t2 {tau2} ' \
           f'-minc {c_min} -maxc {c_max}'
    subprocess.run(args=args, shell=True)

    # preprocess the grpah - reindex from 0, adjust cluster labels accordingly
    g = nx.read_edgelist(input_graph_file, nodetype=int)

    clusters = {}
    with open(input_cluster_file) as reader:
        for line in reader:
            u, c_u = map(int, line.split())
            clusters[u] = c_u
    nx.set_node_attributes(g, name='community', values=clusters)

    relabeled_g = nx.convert_node_labels_to_integers(g, first_label=0, ordering='sorted')

    nx.write_edgelist(relabeled_g, output_graph_file, data=False)
    with open(output_cluster_file, 'w') as writer:
        clusters = nx.get_node_attributes(relabeled_g, name='community')
        for u, c_u in sorted(clusters.items()):
            writer.write(f'{u} {c_u}\n')

    print(f'{output_graph_file=}')
    # remove the temporary files
    remove_files([input_graph_file, input_cluster_file, './statistics.dat', './time_seed.dat'])
    return


def LFR_runner():
    """
    Generate LFR graphs
    TODO: doesnt run on servers, runs on my computer
    :return:
    """
    paths = Paths(basedir='./')
    n = 1_000
    mus = np.linspace(0, 1, 21)
    for mu in mus:
        for avg_k in [15, 20, 25]:
            lfr = LFRGraph(paths=paths, n=n, mu=mu, avg_k=avg_k)
            lfr.generate_graphs(num_graphs=10, overwrite=False)
        # break
    return


def other_methods_runner(mu, avg_k, trial, method):
    paths = Paths(basedir='./')
    lfr = LFRGraph(mu=mu, paths=paths, n=1000, avg_k=avg_k)
    input_g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)
    other_clustering = OtherClustering(paths=paths, input_graph=input_g,
                                       ground_truth_clustering=ground_truth_clustering,
                                       method=method)
    try:
        other_clustering.run(num_trials=10, overwrite=False)
    except Exception as e:
        print(method, e)
    return


def other_methods():
    num_workers = 7
    methods = ['louvain', 'greedy_modularity', 'label_propagation', 'leiden', 'infomap', 'sbm_dl', 'sbm_dl_nested']
    # methods = ['leiden', 'louvain']
    args = []

    mus = np.linspace(0, 1, 21)
    for mu in mus:
        for avg_k in (15, 20, 25):
            for trial in range(1, 21):
                for method in methods:
                    args.append((mu, avg_k, trial, method))

    random.shuffle(args)
    parallel_async(func=other_methods_runner, args=args, num_workers=num_workers)
    return


def corr_cluster_runner(mu, avg_k, trial, method, continuous, erdos_renyi_null_model=False, family='LFR', inertia=0):
    paths = Paths(basedir='./')
    lfr = LFRGraph(mu=mu, paths=paths, n=1000, avg_k=avg_k, family=family)

    input_g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)
    clustering_method = CorrelationClustering(input_graph=input_g, family=family, inertia_factor=inertia,
                                              ground_truth_clustering=ground_truth_clustering,
                                              method=method, continuous=continuous, paths=paths,
                                              erdos_renyi_null=erdos_renyi_null_model)
    clustering_method.run(num_trials=10, overwrite=False)
    return


def corr_cluster(mus=np.linspace(0, 1, 21), family='LFR', inertia_range=range(1, 4)):
    num_workers = 15
    args = []

    num_trials = 10 if family == 'LFR-easy' else 20

    methods = ['GW', 'RW', 'Diff', 'CO'][1:]
    continuous = True

    for inertia in inertia_range:
        for erdos_renyi_null_model in [False, True]:
            for mu in mus:  # 21
                for avg_k in (15, 20, 25):  # 3
                    for trial in range(1, num_trials + 1):  # 20
                        for method in methods:  # 3
                            if inertia > 0 and method == 'CO':  # no inertia for CO yet
                                continue
                            args.append(
                                (mu, avg_k, trial, method, continuous, erdos_renyi_null_model, family, inertia))  # 10

    random.shuffle(args)
    parallel_async(func=corr_cluster_runner, args=args, num_workers=num_workers)
    return


def markov_cluster_runner(mu, avg_k, trial, method, continuous, family):
    paths = Paths(basedir='./')
    lfr = LFRGraph(mu=mu, paths=paths, n=1000, avg_k=avg_k)

    input_g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)
    clustering_method = MarkovClustering(input_graph=input_g, family=family,
                                         ground_truth_clustering=ground_truth_clustering,
                                         method=method, continuous=continuous, paths=paths)
    clustering_method.run(num_trials=10, overwrite=False)
    return


def markov_cluster(mus=np.linspace(0, 1, 21), family='LFR'):
    print(f'Markov {mus=}')
    num_workers = 8
    args = []
    # mus = np.linspace(0, 1, 21)
    continuous = True
    methods = ['RW', 'Diff']  # [1:]
    num_trials = 10 if family == 'LFR-easy' else 20

    for mu in mus:  # 21
        for avg_k in (15, 20, 25):  # 3
            for trial in range(1, num_trials + 1):  # 20
                for method in methods:  # 3
                    args.append((mu, avg_k, trial, method, continuous, family))  # 10

    # random.shuffle(args)
    parallel_async(func=markov_cluster_runner, args=args, num_workers=num_workers, timeout=120)
    return


def multilouvain_cluster_runner(mu, avg_k, trial, family):
    paths = Paths(basedir='./')
    lfr = LFRGraph(mu=mu, paths=paths, n=1000, avg_k=avg_k)

    input_g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)
    clustering_method = MultiLouvainClustering(input_graph=input_g, family=family,
                                               ground_truth_clustering=ground_truth_clustering,
                                               paths=paths)
    clustering_method.run(num_trials=10, overwrite=False)
    return


def multilouvain(mus=np.linspace(0, 1, 21), family='LFR'):
    print(f'MultiLouvain {mus=}')

    num_workers = 4
    args = []
    # mus = np.linspace(0, 1, 21)
    continuous = True
    num_trials = 10 if family == 'LFR-easy' else 20

    for mu in mus:  # 21
        for avg_k in (15, 20, 25):  # 3
            for trial in range(1, num_trials + 1):  # 20
                args.append((mu, avg_k, trial, family))  # 10

    # random.shuffle(args)
    parallel_async(func=multilouvain_cluster_runner, args=args, num_workers=num_workers, timeout=120)
    return


def lfr_level_stats_runner(mu, avg_k, trial, family, stats_dir):
    """
    Use nested SBMs to find number of levels in LFR graphs
    """
    paths = Paths(basedir='./')
    lfr = LFRGraph(mu=mu, paths=paths, n=1000, avg_k=avg_k)

    input_g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)

    G, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)
    name = G.name
    ground_truth_comm_labels = nx.get_node_attributes(G, name='ground_truth')

    lfr_gml_path = lfr.graph_basepath / f'{lfr.name}_{trial}.gml'
    gt_g = gt.load_graph(str(lfr_gml_path))

    state_dc = gt.minimize_nested_blockmodel_dl(gt_g,
                                                state_args=dict(b_max=gt_g.vp['ground\x00truth'],
                                                                deg_corr=True))

    state_ndc = gt.minimize_nested_blockmodel_dl(gt_g,
                                                 state_args=dict(b_max=gt_g.vp['ground\x00truth'],
                                                                 deg_corr=False))

    qual = state_dc.entropy() - state_ndc.entropy()

    if qual > 0:  # pick degree corrected
        state = state_dc
        kind = 'DC'
    else:
        state = state_ndc
        kind = 'NDC'

    levels = state.get_levels()
    for i, s in enumerate(levels):
        # print(f'{i=} {s.get_nonempty_B()=:,} {s.get_N()=:,}')
        if s.get_nonempty_B() == 1:
            break
    num_levels = i
    # info_row = dict(name=name, trial=trial, mu=mu, avg_k=avg_k, levels=num_levels, kind=kind)
    with open(stats_dir, 'a') as fp:
        fp.write(f'{name},{mu},{avg_k},{trial},{num_levels},{kind}\n')

    return


def lfr_level_stats(mus, family='LFR', overwrite=False):
    print(f'LFR stats {mus=}')
    paths = Paths(basedir='./')
    stats_dir = paths.results_dir / 'LFR_stats.csv'

    if not stats_dir.exists() or overwrite:
        with open(stats_dir, 'w') as fp:
            fp.write(f'name,mu,avg_k,trial,levels,kind\n')

    if stats_dir.exists():
        existing_df = pd.read_csv(stats_dir)

    num_workers = 8
    args = []
    # mus = np.linspace(0, 1, 21)
    num_trials = 10 if family == 'LFR-easy' else 20

    for mu in mus:  # 21
        for avg_k in (15, 20, 25):  # 3
            for trial in range(1, num_trials + 1):  # 20
                if len(existing_df.query('mu==@mu & avg_k==@avg_k & trial==@trial', engine='python')) > 0:
                    continue
                args.append((mu, avg_k, trial, family, stats_dir))  # 10

    # random.shuffle(args)
    print(f'{len(args)=}')
    parallel_async(func=lfr_level_stats_runner, args=args, num_workers=num_workers, timeout=120)
    return


if __name__ == '__main__':
    print('hello')
    mus = np.linspace(0, 1, 21)

    # lfr_level_stats(mus=mus)

    corr_cluster(mus=mus, family='LFR')
    # other_methods()
    # markov_cluster(family='LFR', mus=mus[: 5])
    # multilouvain(family='LFR-easy')

    # corr_cluster_runner(mu=0.1, avg_k=15, trial=1, method='GW', continuous=True, exact=True)
    #
    # main()
    # for avg_k in [15, 20, 25]:
    #     for mu in np.linspace(0, 1, 21):
    #         lfr = LFRGraph(paths=paths, n=1000, mu=mu, avg_k=avg_k)
    #         lfr.generate_graphs(num_graphs=20)
    # paths = Paths(basedir='./')
    # avg_k = 15
    # mu = 0.4
    # lfr = LFRGraph(paths=paths, n=1000, mu=mu, avg_k=avg_k)
    # g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=15)
    # print(nx.info(g), len(ground_truth_clustering.communities))
    # # print(list(g.nodes(data=True))[: 2])
    # #
    # method = 'GW'
    # is_continuous = True
    # corr_cluster = CorrelationClustering(paths=paths, input_graph=g, continuous=is_continuous, method=method,
    #                                      ground_truth_clustering=ground_truth_clustering, exact=True)
    #
    # corr_cluster.run(num_trials=1, overwrite=False)

    # other_clustering = OtherClustering(paths=paths, input_graph=g, ground_truth_clustering=ground_truth_clustering,
    #                                    method='leiden')
    # other_clustering.run(num_trials=1, overwrite=True)
    # corr_tester()
    # other_methods_runner()
    # LFR_runner()
