"""Main driver"""
import random
import sys

import graph_tool.all as gt
import networkx as nx
import numpy as np
import pandas as pd

sys.path.extend(['./', '../'])
from src.clustering import CorrelationClustering, OtherClustering, MarkovClustering, MultiLouvainClustering
from src.utils import Paths, parallel_async, get_cluster_paths, get_long_method_name
from src.graph_io import LFRGraph, SBMGraph


def LFR_runner(n: int):
    """
    Generate LFR graphs
    :return:
    """
    paths = Paths(basedir='./')
    # mus = np.linspace(0, 1, 21)
    mus = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6]
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


def corr_cluster_runner(paths, mu, avg_k, trial, method, continuous, erdos_renyi_null_model=False, family='LFR',
                        inertia=0, inertia_type='v1', n=1000, run_method=True):
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
    if run_method:
        clustering_method.run(num_trials=10, overwrite=False)
    return


def corr_cluster(paths, methods, mus=np.linspace(0, 1, 21), family='LFR', inertia_range=[0],
                 n=1_000, num_workers=10, run_method=True):
    # inertia_range = np.arange(0.5, 5.5, 0.5)
    args = []
    print(f'{mus=} {inertia_range=}')

    if family in ('LFR-easy', 'sbm'):
        num_trials = 10
    else:
        num_trials = 20

    if n == 10_000:
        max_k = 500
    elif n == 5_000:
        max_k = 250
    else:
        max_k = 50

    if run_method:
        finished_paths = set(
            pd.read_csv(paths.results_dir / 'stats_CL.csv', engine='c', usecols=['output_cluster_path'])
            .output_cluster_path
            .values
        )
        print(f'{len(finished_paths):,} finished runs')
    else:
        spectrum_path = paths.results_dir / 'spectrum_CL.csv'
        if spectrum_path.exists():
            spectrum_df = pd.read_csv(spectrum_path, engine='c')
        else:
            spectrum_df = pd.DataFrame()

        finished_paths = set()

    continuous, erdos_renyi_null_model = True, False
    num_found = 0
    for inertia in inertia_range:
        for mu in mus:  # 21
            for avg_k in (10, 15, 20):  # 3
                if avg_k == 10 or n > 1_000:
                    num_trials = 10
                for trial in range(1, num_trials + 1):  # 20
                    for method in methods:  # 3
                        if inertia > 0 and method == 'CO':  # no inertia for CO yet
                            continue
                        if run_method:
                            cluster_paths = set(
                                get_cluster_paths(paths=paths, kind='CL', family=family, n=n, max_k=max_k, avg_k=avg_k,
                                                  mu=mu, trial=trial, is_continuous=continuous, method=method,
                                                  erdos_renyi_null=erdos_renyi_null_model, inertia_factor=inertia,
                                                  inertia_type='v1')
                            )
                            if method == 'RW':
                                cluster_paths.update(
                                    get_cluster_paths(paths=paths, kind='CL', family=family, n=n, max_k=max_k,
                                                      avg_k=avg_k,
                                                      mu=mu, trial=trial, is_continuous=continuous, method=method,
                                                      erdos_renyi_null=erdos_renyi_null_model, inertia_factor=inertia,
                                                      inertia_type='v2')
                                )

                            for cluster_path in cluster_paths:
                                if cluster_path in finished_paths:
                                    num_found += 1
                                    continue
                                if 'v2' in cluster_path:
                                    inertia_type = 'v2'
                                else:
                                    inertia_type = 'v1'
                                args.append(
                                    (paths, mu, avg_k, trial, method, continuous, erdos_renyi_null_model, family,
                                     inertia, inertia_type, n, True))  # 10
                        else:  ## this is for spectrum
                            found = False
                            for inertia_type in ['v1', 'v2']:
                                if method != 'RW' and inertia_type == 'v2':
                                    continue

                                method_long = get_long_method_name(kind='CL', method=method, inertia_type=inertia_type,
                                                                   inertia_factor=inertia, is_continuous=continuous,
                                                                   erdos_renyi_null=erdos_renyi_null_model)
                                if len(spectrum_df) > 0:
                                    rows = (
                                        spectrum_df
                                        .query(
                                            'method==@method_long & avg_k==@avg_k & family==@family & mu==@mu & trial==@trial')
                                    )
                                    if len(rows) == 1:
                                        found = True
                                        num_found += 1
                                    elif len(rows) > 1:
                                        raise NotImplementedError(f'{len(rows)=:,} found. expecting 0 or 1.')

                                if not found:
                                    args.append(
                                        (paths, mu, avg_k, trial, method, continuous, erdos_renyi_null_model, family,
                                         inertia, inertia_type, n, False))  # 10

    print(f'Found: {num_found:,} \t Missing: {len(args):,}, {num_workers=:,} {run_method=}')
    if len(args) > 0:
        random.shuffle(args)
        parallel_async(func=corr_cluster_runner, args=args, num_workers=num_workers)
    return


def markov_cluster_runner(paths, mu, avg_k, trial, method, continuous, family, n):
    if family in ('LFR', 'LFR-easy'):
        lfr = LFRGraph(mu=mu, paths=paths, n=n, avg_k=avg_k, family=family)
        input_g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)
    elif family == 'sbm':
        sbm = SBMGraph(mu=mu, paths=paths, n=n, avg_k=avg_k, family=family)
        input_g, ground_truth_clustering = sbm.get_graph_and_clustering(trial=trial)
    else:
        raise NotImplementedError(f'{family=}')

    clustering_method = MarkovClustering(input_graph=input_g, family=family,
                                         ground_truth_clustering=ground_truth_clustering,
                                         method=method, continuous=continuous, paths=paths)
    clustering_method.run(num_trials=10, overwrite=False)
    return


def markov_cluster(paths, mus=np.linspace(0, 1, 21), family='LFR', n=1000, num_workers=5):
    print(f'Markov {mus=}')
    args = []
    # mus = np.linspace(0, 1, 21)
    continuous = True
    methods = ['RW', 'Diff']  # [: 1]  # [1:]
    if family in ('LFR-easy', 'sbm'):
        num_trials = 10
    else:
        num_trials = 20

    if n > 1000:
        num_trials = 10

    if n == 10_000:
        max_k = 500
    else:
        max_k = 50

    finished_paths = set(
        pd.read_csv(paths.results_dir / 'stats_MC.csv', engine='c', usecols=['output_cluster_path'])
        .output_cluster_path
        .values
    )
    print(f'{len(finished_paths):,} finished runs')

    for mu in mus:  # 21
        for avg_k in (10, 15, 20):  # 3
            if avg_k == 10:
                num_trials = 10
            for trial in range(1, num_trials + 1):  # 20
                for method in methods:  # 3
                    cluster_paths = set(
                        get_cluster_paths(paths=paths, kind='MC', family=family, n=n, max_k=max_k, avg_k=avg_k,
                                          mu=mu, trial=trial, is_continuous=continuous, method=method,
                                          erdos_renyi_null=False, inertia_factor=0,
                                          inertia_type='v1')
                    )
                    for cluster_path in cluster_paths:
                        if cluster_path in finished_paths:
                            continue
                        args.append((paths, mu, avg_k, trial, method, continuous, family, n))  # 10

    print(f'Missing runs: {len(args):,} ({args[: 2]})')
    if len(args) > 0:
        random.shuffle(args)
        parallel_async(func=markov_cluster_runner, args=args, num_workers=num_workers, timeout=120)
    return


def multilouvain_cluster_runner(paths, mu, avg_k, trial, family, n):
    if family in ('LFR', 'LFR-easy'):
        lfr = LFRGraph(mu=mu, paths=paths, n=n, avg_k=avg_k, family=family)
        input_g, ground_truth_clustering = lfr.get_graph_and_clustering(trial=trial)
    elif family == 'sbm':
        sbm = SBMGraph(mu=mu, paths=paths, n=n, avg_k=avg_k, family=family)
        input_g, ground_truth_clustering = sbm.get_graph_and_clustering(trial=trial)
    else:
        raise NotImplementedError(f'{family=}')

    clustering_method = MultiLouvainClustering(input_graph=input_g, family=family,
                                               ground_truth_clustering=ground_truth_clustering,
                                               paths=paths)
    clustering_method.run(num_trials=10, overwrite=False)
    return


def multilouvain(paths, mus=np.linspace(0, 1, 21), family='LFR', n=1_000):
    print(f'MultiLouvain {mus=}')
    finished_paths = set(
        pd.read_csv(paths.results_dir / 'stats_ML.csv', engine='c', usecols=['output_cluster_path'])
        .output_cluster_path
        .values
    )
    print(f'{len(finished_paths):,} finished runs')

    num_workers = 4
    continuous = True

    if family in ('LFR-easy', 'sbm'):
        num_trials = 10
    else:
        num_trials = 20

    if n == 10_000:
        max_k = 500
    else:
        max_k = 50

    args = []
    for mu in mus:  # 21
        for avg_k in (10, 15, 20):  # 3
            if avg_k == 10:
                num_trials = 10
            for trial in range(1, num_trials + 1):  # 20

                cluster_paths = set(
                    get_cluster_paths(paths=paths, kind='ML', family=family, n=n, max_k=max_k, avg_k=avg_k,
                                      mu=mu, trial=trial, is_continuous=continuous, method='ML',
                                      erdos_renyi_null=False, inertia_factor=0,
                                      inertia_type='v1')
                )
                for cluster_path in cluster_paths:
                    if cluster_path in finished_paths:
                        continue

                    args.append((paths, mu, avg_k, trial, family, n))  # 10

    print(f'Missing runs: {len(args):,} ({args[: 2]})')
    if len(args) > 0:
        random.shuffle(args)
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
    # LFR_runner(n=5_000)

    # sbm = SBMGraph(paths=paths, n=1_000, mu=0.05, avg_k=10)
    # g, ground_truth_clustering = sbm.get_graph_and_clustering(trial=2)
    # print(nx.info(g))
    #
    # method = 'Diff'
    # inertia_factor = 0.5
    # er_null_model = False
    # markov_clustering = MarkovClustering(paths=paths, input_graph=g, continuous=True, method='RW',
    #                                      ground_truth_clustering=ground_truth_clustering)
    # markov_clustering.run(num_trials=10, overwrite=True)
    #
    paths = Paths(basedir='./')
    # # family, n, num_workers = 'sbm', 1_000, 6
    family, n, num_workers = 'LFR', 5_000, 12
    # # family, n, num_workers = 'LFR', 1_000, 8
    # family, n, num_workers = 'LFR', 10_000, 6

    mus = [0.05, 0.1, 0.2, 0.4, 0.5, 0.6]  # mus = np.linspace(0, 1, 21)
    method = 'CL'
    # # method = 'Markov'
    # # method = 'MultiLouvain'
    print(f'{method=!r} {family!r} {n=:,} {mus=} experiments')

    # # inertia_values = sorted(set(list(range(0, 6)) + list(np.geomspace(1, 300, 5).round(2)) + [500, 750, 1000, 2000, 5000]))
    inertia_values = [0, 1, 2, 3, 4, 5, 10, 20, 50, 100, 200, 500, 750, 1000, 2000, 5000][: 5]

    if method == 'CL':
        corr_cluster(paths=paths, mus=mus, inertia_range=inertia_values, family=family,
                     methods=['Diff', 'RW'], n=n, num_workers=num_workers, run_method=False)
    elif method == 'Markov':
        markov_cluster(paths=paths, family=family, mus=mus, n=n)
    elif method == 'MultiLouvain':
        multilouvain(family=family, mus=mus, n=n, paths=paths)
    else:
        raise NotImplementedError(f'{method=}')

    # lfr_level_stats(mus=mus)
    # other_methods()
