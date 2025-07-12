import argparse
import itertools
import os
import warnings
import pandas as pd
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster
from .generators import *
from .algorithms import *
from .utils import *
from .config import GLOBAL_CONFIG

warnings.simplefilter('ignore')

algorithm_table = {'glasso': graphical_lasso,
                   'gl_via_Kendall': graphical_lasso_via_Kendall,
                   'gl_via_Fechner': graphical_lasso_via_Fechner}
generator_table = {'DD': generate_dominant_diagonal, 'DS': generate_diagonal_shift, 'CH': generate_cholesky}

def generate_models(n_models, dim, density, generator):
    models = [generator(dim, density) for _ in range(n_models)]
    return models

def run_model(precision, algorithm, dim, n_samples, n_eps, n_repl, t_dof):
    result_metrics = list()

    covariance = np.linalg.inv(precision)
    upper_diagonal = np.triu_indices_from(precision, k=1)
    true_edges = (precision[upper_diagonal] != 0.).astype(int)

    for eps in np.linspace(0, 1, n_eps):
        for repl in range(n_repl):
            data = sample_from_mixed(n_samples, covariance, eps, t_dof)
            try:
                emp_prec = algorithm(data)
            except:
                continue

            pred_edges = (emp_prec[upper_diagonal] != 0.).astype(int)
            metrics = pd.DataFrame(evaluate(true_edges, pred_edges), index=[0,])
            metrics['eps'] = eps
            result_metrics.append(metrics)

    if len(result_metrics) > 0:
        result_df = pd.concat(result_metrics, ignore_index=True)
        return result_df
    else:
        return pd.DataFrame()
    

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('algorithm', type=str)
    parser.add_argument('generator', type=str)
    parser.add_argument('-pc', type=str, default='local', dest='parallel_context')
    args = parser.parse_args()

    if args.parallel_context == 'local':
        cluster = LocalCluster()
    elif args.parallel_context == 'slurm':
        cluster = SLURMCluster(cores=1, memory='16GB', walltime='05:00:00', header_skip=['--mem'], env_extra=[
            "export PYTHONPATH=/home/ikostylev/robustEGMS/experiments:$PYTHONPATH"
        ])
        cluster.scale(32)

    client = cluster.get_client()

    generator = generator_table[args.generator]
    algorithm = algorithm_table[args.algorithm]
    
    all_configs = [
        dict(zip(GLOBAL_CONFIG.keys(), values))
        for values in itertools.product(*GLOBAL_CONFIG.values())
    ]

    for config in all_configs:
        dim = config['N']
        n_samples = config['n']
        density = config['d']
        n_eps = config['n_eps_points']
        n_models = config['S_sg']
        n_repl = config['S_obs']
        t_dof = config['student_dof']
        
        results_dir = os.path.join('data', args.generator, args.algorithm)
        os.makedirs(results_dir, exist_ok=True)
        results_path = os.path.join(results_dir, f'N{dim}_n{n_samples}_d{density}_dof{t_dof}.csv')


        models = generate_models(n_models, dim, density, generator)

        futures = [client.submit(run_model, model, algorithm, dim, n_samples, n_eps, n_repl, t_dof, pure=False) for model in models]
        results_df = pd.concat(client.gather(futures))

        print(results_path)
        results_df.to_csv(results_path)

    client.close()
    cluster.close()

if __name__ == '__main__':
    main()