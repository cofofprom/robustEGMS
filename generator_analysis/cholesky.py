import numpy as np
import networkx as nx
import matplotlib.pyplot as plt
from sklearn.datasets import make_sparse_spd_matrix
from dask.distributed import LocalCluster

def calculate_spectrum(mat):
    eigs = np.linalg.eigvals(mat)
    eigs = np.sort(eigs)

    return np.real(eigs)

def calculate_condnum(mat):
    return np.linalg.cond(mat)

N = 30
d = 0.95
N_REP = 200
gen_name = 'CH'

def main():
    test_mats = [make_sparse_spd_matrix(N, alpha=d, norm_diag=True) for nmat in range(N_REP)]
    nonzero_off_diag_data = list()

    for mat in test_mats:
        off_diag_idx = np.triu_indices_from(mat, k=1)
        nonzero_off_diag = mat[off_diag_idx]
        nonzero_off_diag_data.extend(nonzero_off_diag[nonzero_off_diag != 0.])

    nonzero_off_diag_data = np.array(nonzero_off_diag_data)

    plt.hist(nonzero_off_diag_data, bins=int(np.sqrt(len(nonzero_off_diag_data))))
    plt.title('Off-diagonal nonzero values')
    plt.savefig(f'plots/{gen_name}_values.png', bbox_inches='tight')
    plt.gca().clear()

    cluster = LocalCluster()
    client = cluster.get_client()

    waiters = [client.submit(calculate_spectrum, test_mat) for test_mat in test_mats]

    eig_data = np.stack(client.gather(waiters))
    avg_eig = np.mean(eig_data, axis=0)
    q25_eig = np.quantile(eig_data, 0.25, axis=0)
    q75_eig = np.quantile(eig_data, 0.75, axis=0)

    ind = np.arange(1, N+1)
    plt.scatter(ind, avg_eig)
    plt.fill_between(ind, q25_eig, q75_eig, alpha=0.2, color='red')
    plt.title('Average spectrum')
    plt.savefig(f'plots/{gen_name}_spectrum.png', bbox_inches='tight')
    plt.gca().clear()

    waiters = [client.submit(calculate_condnum, test_mat) for test_mat in test_mats]
    cond_data = np.stack(client.gather(waiters))

    plt.boxplot(cond_data, showfliers=False)
    plt.title('Conditional numbers')
    plt.savefig(f'plots/{gen_name}_condnumbers.png', bbox_inches='tight')
    plt.gca().clear()


if __name__ == '__main__':
    main()