import numpy as np
import networkx as nx
from sklearn.datasets import make_sparse_spd_matrix

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

def generate_cholesky(dim, density):
    def density2alpha(d): return 0.95

    return make_sparse_spd_matrix(dim, alpha=density2alpha(density), norm_diag=True)