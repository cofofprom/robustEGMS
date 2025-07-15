import argparse
import itertools
import os
import warnings
import uuid
import pandas as pd
from .generators import *
from .algorithms import *
from .utils import *
from .config import GLOBAL_CONFIG

warnings.simplefilter('ignore')

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

            pred_edges = (np.abs(emp_prec[upper_diagonal] - 0.) >= 1e-6).astype(int)
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
    args = parser.parse_args()

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
        
        results_dir = os.path.join('data', args.generator, args.algorithm, f'N{dim}_n{n_samples}_d{density}_dof{t_dof}')
        results_path = os.path.join(results_dir, f'{uuid.uuid4()}.csv')
        os.makedirs(results_dir, exist_ok=True)

        models = generate_models(n_models, dim, density, generator)

        futures = [run_model(model, algorithm, dim, n_samples, n_eps, n_repl, t_dof) for model in models]
        results_df = pd.concat(futures)

        print(results_path)
        results_df.to_csv(results_path)

if __name__ == '__main__':
    main()