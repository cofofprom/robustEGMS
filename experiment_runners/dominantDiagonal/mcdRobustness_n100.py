import random
import numpy as np
import pandas as pd
from sklearn.covariance import MinCovDet
import logging
from time import perf_counter
import warnings
from dask.distributed import LocalCluster
from dask_jobqueue import SLURMCluster
from utils import *

warnings.simplefilter('ignore')
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger()

# np.random.seed(24142)
# random.seed(24142)

N = 30
n = 100
S_sg = 1000
S_obs = 200
eps_n_points = 10
d = 0.1
reg_param = 0.1
t_dof = 3
experiment_name = 'MCDRobustness_n100'
MAX_WORKERS=32

def run_model(eps, n_model):
    result_metrics = list()

    precision = generate_dominant_diagonal(N, d)
    covar = np.linalg.inv(precision)

    upper_diagonal = np.triu_indices_from(precision, k=1)
    true_edges = (precision[upper_diagonal] != 0.).astype(int)

    for n_repl in range(S_obs):
        data = sample_from_mixed(n, covar, eps, t_dof)
        
        try:
            mcd = MinCovDet(assume_centered=True).fit(data)
            emp_prec = mcd.get_precision()
        except:
            continue

        pred_edges = (emp_prec[upper_diagonal] != 0.).astype(int)
        metrics = pd.DataFrame(evaluate(true_edges, pred_edges), index=[0,])
        result_metrics.append(metrics)

    if len(result_metrics) > 0:
        df = pd.concat(result_metrics, ignore_index=True)
        df['model_id'] = n_model

        return df
    else:
        return pd.DataFrame()

def main():
    #cluster = LocalCluster()
    cluster = SLURMCluster(cores=1, memory='16GB', header_skip=['--mem'], env_extra=[
        "export PYTHONPATH=/home/ikostylev/robustEGMS/experiment_runners:$PYTHONPATH"
    ])
    cluster.scale(MAX_WORKERS)
    client = cluster.get_client()
    results = pd.DataFrame()

    for eps in np.linspace(0, 1, eps_n_points):
        logging.info(f"Epsilon: {eps:.2f}...")
        start_time = perf_counter()

        futures = list()
        for n_model in range(S_sg):
            futures.append(client.submit(run_model, eps, n_model, pure=False))
        metrics = client.gather(futures)
        eps_result = pd.concat(metrics, ignore_index=True)
        logging.info(eps_result.shape)
        
        end_time = perf_counter()
        dur = end_time - start_time
        logging.info(f"Finished in {dur:.3f}s")
        logging.info(f"~{dur / S_sg:.3f}s per model")
        logging.info(f"~{dur / S_sg / S_obs:.3f}s per replication")

        eps_result['epsilon'] = eps 
        results = pd.concat([results, eps_result], ignore_index=True)
    
    results.to_csv(f'data/{experiment_name}.csv', index=False)

    draw_plots(experiment_name, results)

if __name__ == '__main__':
    main()
