import numpy as np
import networkx as nx
from scipy.stats import multivariate_normal, multivariate_t, hmean, kendalltau
from sklearn.metrics import confusion_matrix
import matplotlib.pyplot as plt

def sample_from_mixed(n, cov, eps, dof):
    dim = cov.shape[0]
    t_cov_scaler = (dof - 2) / dof
    norm_samples = multivariate_normal.rvs(np.zeros(dim), cov, size=n)
    t_samples = multivariate_t.rvs(np.zeros(dim), t_cov_scaler * cov, df=dof, size=n)
    selector = (np.random.uniform(0, 1, size=n) < eps).astype(int)
    result = norm_samples.T * (1 - selector) + t_samples.T * selector
    return result.T

def generate_dominant_diagonal(dim, density):
    graph = nx.gnp_random_graph(dim, density)
    adj = nx.adjacency_matrix(graph).toarray()

    A = np.random.uniform(0.5, 1, size=(dim, dim))
    B = np.random.choice([-1, 1], size=(dim, dim))

    prec = adj * A * B
    rowsums = np.sum(np.abs(prec), axis=1)
    rowsums[rowsums == 0] = 1e-4

    prec = prec / (1.5 * rowsums[:, None])
    prec = (prec + prec.T) / 2 + np.eye(dim)

    return prec

def generate_diagonal_shift(dim, density, norm_diag=True):
    graph = nx.gnp_random_graph(dim, density)
    adj = nx.adjacency_matrix(graph).toarray()

    A = np.random.uniform(0.5, 1, size=(dim, dim))
    B = np.random.choice([-1, 1], size=(dim, dim))

    prec = adj * A * B
    prec = (prec + prec.T) / 2 + np.eye(dim)
    min_eig = np.min(np.linalg.eigvals(prec))

    prec[np.diag_indices_from(prec)] += np.abs(min_eig) + 0.1

    if norm_diag:
        D = np.diag(1 / np.sqrt(np.diag(prec)))
        prec = D @ prec @ D

    return prec

def evaluate(true_edges, pred_edges):
    conf = confusion_matrix(true_edges, pred_edges)
    tn = conf[0, 0]
    fp = conf[0, 1]
    fn = conf[1, 0]
    tp = conf[1, 1]
    
    fdr = np.nan_to_num(fp / (tp + fp), nan=0)
    fomr = np.nan_to_num(fn / (tn + fn), nan=0)
    tpr = np.nan_to_num(tp / (tp + fn), nan=1)
    tnr = np.nan_to_num(tn / (tn + fp), nan=1)

    ba = (tpr + tnr) / 2
    f1 = hmean([1 - fdr, tpr])
    mcc_first = tpr * tnr * (1 - fdr) * (1 - fomr)
    mcc_second = (1 - tpr) * (1 - tnr) * fomr * fdr
    mcc = np.sqrt(mcc_first) - np.sqrt(mcc_second)

    metrics = {'tn': tn, 'fp': fp,
               'fn': fn, 'tp': tp,
               'fdr': fdr, 'for': fomr,
               'tpr': tpr, 'tnr': tnr,
               'ba': ba, 'f1': f1, 'mcc': mcc}
    for k in metrics: metrics[k] = metrics[k].item()
    
    return metrics

def draw_plots(experiment_name, results_df):
    fig, axes = plt.subplots(1, 3)
    gb_df = results_df.groupby('epsilon')
    agg_df = gb_df.mean()
    q25_df = gb_df.quantile(0.25)
    q75_df = gb_df.quantile(0.75)
    eps = agg_df.index
    
    axes[0].plot(eps, agg_df['ba'])
    axes[0].fill_between(eps, q25_df['ba'], q75_df['ba'], alpha=0.2)
    axes[0].set_title('BA')
    axes[0].set_ylim(0, 1)

    axes[1].plot(eps, agg_df['f1'])
    axes[1].fill_between(eps, q25_df['f1'], q75_df['f1'], alpha=0.2)
    axes[1].set_title('F1')
    axes[1].set_ylim(0, 1)

    axes[2].plot(eps, agg_df['mcc'])
    axes[2].fill_between(eps, q25_df['mcc'], q75_df['mcc'], alpha=0.2)
    axes[2].set_title('MCC')
    axes[2].set_ylim(0, 1)

    fig.set_size_inches(10, 5)
    
    fig.savefig(f'plots/{experiment_name}.png', bbox_inches='tight')

def kendall_corr_mat(data):
    dim = data.shape[1]
    matrix = np.array([[kendalltau(data[:, i], data[:, j]).statistic
                        for j in range(dim)] for i in range(dim)])

    return matrix

def pearson_corr_via_kendall(data):
    kcorr_mat = kendall_corr_mat(data)
    pearson_corr = np.sin(np.pi / 2 * kcorr_mat)

    return pearson_corr