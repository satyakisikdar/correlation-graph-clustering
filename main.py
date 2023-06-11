"""Main driver"""
import random
import sys

import graph_tool.all as gt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.extend(['./', '../'])
from src.clustering import CorrelationClustering, OtherClustering, MarkovClustering, MultiLouvainClustering
from src.utils import Paths, parallel_async
from src.graph_io import LFRGraph, SBMGraph


def LFR_runner(n: int):
    """
    Generate LFR graphs
    TODO: doesnt run on servers, runs on my computer
    :return:
    """
    paths = Paths(basedir='./')
    mus = np.linspace(0, 1, 21)
    for mu in mus:
        for avg_k in [10, 15, 20]:
            lfr = LFRGraph(paths=paths, n=n, mu=mu, avg_k=avg_k)
            lfr.generate_graphs(num_graphs=10, overwrite=False)
        # break
    return


def sbm_graph_runner(mus):
    """
    SBM graph generator
    """
    paths = Paths(basedir='./')
    n = 1_000
    avg_ks = [5, 10, 15, 20, 25]
    for mu in mus:
        for avg_k in avg_ks:
            sbm = SBMGraph(paths=paths, n=n, mu=mu, avg_k=avg_k)
            sbm.generate_graphs(num_graphs=10, overwrite=False)

    # test
    # mu = 0.1
    # avg_k = 10
    # sbm = SBMGraph(paths=paths, n=n, mu=mu, avg_k=avg_k)
    # for trial in range(1, 11):
    #     g, clustering = sbm.get_graph_and_clustering(trial=trial)
    #     assert g.graph['avg_k'] == avg_k and g.graph['mu'] == mu
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


def corr_cluster_runner(mu, avg_k, trial, method, continuous, erdos_renyi_null_model=False, family='LFR', inertia=0,
                        inertia_type='v1', n=1000):
    paths = Paths(basedir='./')

    if family in ('LFR', 'LFR-easy'):
        lfr = LFRGraph(mu=mu, paths=paths, n=n, avg_k=avg_k, family=family)
        input_g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)
    elif family == 'sbm':
        sbm = SBMGraph(mu=mu, paths=paths, n=n, avg_k=avg_k, family=family)
        input_g, ground_truth_clustering = sbm.get_graph_and_clustering(trial=trial)
    else:
        raise NotImplementedError(f'{family=}')

    clustering_method = CorrelationClustering(input_graph=input_g, family=family, inertia_factor=inertia,
                                              ground_truth_clustering=ground_truth_clustering,
                                              method=method, continuous=continuous, paths=paths,
                                              erdos_renyi_null=erdos_renyi_null_model, inertia_type=inertia_type)
    clustering_method.run(num_trials=10, overwrite=False)
    return


def corr_cluster(methods, mus=np.linspace(0, 1, 21), family='LFR', inertia_range=[0], inertia_type='v1',
                 n=1_000):
    # inertia_range = np.arange(0.5, 5.5, 0.5)
    num_workers = 15
    args = []
    print(f'{mus=} {inertia_range=}')
    num_trials = 20
    if family in ('LFR-easy', 'sbm'):
        num_trials = 10

    # methods = ['GW', 'RW', 'Diff', 'CO']
    # methods = ['RW', 'Diff']
    continuous = True

    for inertia in inertia_range:
        for erdos_renyi_null_model in [False, True][: 1]:
            for mu in mus:  # 21
                for avg_k in (5, 10, 15, 20, 25):  # 3
                    if family == 'LFR' and avg_k == 5:
                        continue
                    if avg_k == 10 or n == 10_000:
                        num_trials = 10
                    for trial in range(1, num_trials + 1):  # 20
                        for method in methods:  # 3
                            if method != 'RW-C':
                                inertia_type = 'v1'
                            if inertia > 0 and method == 'CO':  # no inertia for CO yet
                                continue
                            args.append(
                                (mu, avg_k, trial, method, continuous, erdos_renyi_null_model, family, inertia,
                                 inertia_type, n))  # 10

    random.shuffle(args)
    parallel_async(func=corr_cluster_runner, args=args, num_workers=num_workers)
    return


def markov_cluster_runner(mu, avg_k, trial, method, continuous, family):
    paths = Paths(basedir='./')
    if family in ('LFR', 'LFR-easy'):
        lfr = LFRGraph(mu=mu, paths=paths, n=1000, avg_k=avg_k, family=family)
        input_g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)
    elif family == 'sbm':
        sbm = SBMGraph(mu=mu, paths=paths, n=1000, avg_k=avg_k, family=family)
        input_g, ground_truth_clustering = sbm.get_graph_and_clustering(trial=trial)
    else:
        raise NotImplementedError(f'{family=}')

    clustering_method = MarkovClustering(input_graph=input_g, family=family,
                                         ground_truth_clustering=ground_truth_clustering,
                                         method=method, continuous=continuous, paths=paths)
    clustering_method.run(num_trials=10, overwrite=False)
    return


def markov_cluster(mus=np.linspace(0, 1, 21), family='LFR'):
    print(f'Markov {mus=}')
    num_workers = 5
    args = []
    # mus = np.linspace(0, 1, 21)
    continuous = True
    methods = ['RW', 'Diff'][: 1]  # [1:]
    if family in ('LFR-easy', 'sbm'):
        num_trials = 10
    else:
        num_trials = 20

    for mu in mus:  # 21
        for avg_k in (10, 15, 20):  # 3
            if avg_k == 10:
                num_trials = 10
            for trial in range(1, num_trials + 1):  # 20
                for method in methods:  # 3
                    args.append((mu, avg_k, trial, method, continuous, family))  # 10

    # random.shuffle(args)
    parallel_async(func=markov_cluster_runner, args=args, num_workers=num_workers, timeout=120)
    return


def multilouvain_cluster_runner(mu, avg_k, trial, family):
    paths = Paths(basedir='./')
    if family in ('LFR', 'LFR-easy'):
        lfr = LFRGraph(mu=mu, paths=paths, n=1000, avg_k=avg_k, family=family)
        input_g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)
    elif family == 'sbm':
        sbm = SBMGraph(mu=mu, paths=paths, n=1000, avg_k=avg_k, family=family)
        input_g, ground_truth_clustering = sbm.get_graph_and_clustering(trial=trial)
    else:
        raise NotImplementedError(f'{family=}')

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

    if family in ('LFR-easy', 'sbm'):
        num_trials = 10
    else:
        num_trials = 20

    for mu in mus:  # 21
        for avg_k in (10, 15, 20, 25)[: -1]:  # 3
            if avg_k == 10:
                num_trials = 10
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
    # LFR_runner(n=10_000)

    print('Inertia experiments')
    # mus = np.linspace(0, 1, 21)
    family = 'LFR'
    n = 10_000
    # family = 'sbm'
    mus = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6]
    #
    inertia_values = list(range(0, 6)) + list(np.geomspace(1, 300, 5).round(2)) + [500, 750, 1000, 2000, 5000]
    inertia_values = sorted(set(inertia_values))  # prevent duplicates
    #
    # # # inertia_values = [0]  # run the vanilla RW
    # # inertia_type = 'v1'
    #
    corr_cluster(mus=mus, inertia_range=inertia_values, family=family, inertia_type='v2', methods=['Diff', 'RW'],
                 n=n)
    # sbm_graph_runner(mus)

    # lfr_level_stats(mus=mus)
    # other_methods()

    # markov_cluster(family=family, mus=mus)
    # multilouvain(family=family, mus=mus)
