"""Main driver"""

import subprocess
from pathlib import Path

import networkx as nx
import numpy as np
from tqdm.auto import tqdm

from src.clustering import LFRGraph
from src.utils import Paths, remove_files


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
    :return:
    """
    paths = Paths(basedir='./')
    n = 1000
    mus = np.linspace(0, 1, 21)
    for mu in mus:
        generate_LFR(paths=paths, n=n, mu=mu)
    return


def main():
    paths = Paths(basedir='./')
    n = 1000
    mus = np.linspace(0, 1, 21)
    for mu in reversed(mus):
        lfr_graph = LFRGraph(paths=paths, n=1000, mu=mu)
        # methods = ['leiden', 'louvain', 'infomap', 'label_propagation', 'greedy_modularity'][-2: ]
        methods = ['infomap']
        with tqdm(total=len(methods), desc='Running algorithms...', leave=True) as pbar:
            for method in methods:
                pbar.set_description(f'Mu: {mu:.2f} Method: {method!r}')
                try:
                    lfr_graph.run_method(method=method, num_trials=10)
                except Exception as e:
                    print(e)
                pbar.update(1)

    return


if __name__ == '__main__':
    main()
