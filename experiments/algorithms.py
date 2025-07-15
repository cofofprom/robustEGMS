import numpy as np
from .config import GLOBAL_CONFIG
from .utils import pearson_corr_via_fechner, pearson_corr_via_kendall
from sklearn.covariance import graphical_lasso as sklearn_glasso, GraphicalLassoCV, MinCovDet

lasso_param = GLOBAL_CONFIG['glasso_reg_param'][0]

def graphical_lasso(data):
    emp_cov = np.cov(data, rowvar=False)
    _, emp_prec = sklearn_glasso(emp_cov, lasso_param)

    return emp_prec

def graphical_lasso_via_Kendall(data):
    emp_cov = pearson_corr_via_kendall(data)
    _, emp_prec = sklearn_glasso(emp_cov, lasso_param)
    return emp_prec

def graphical_lasso_via_Fechner(data):
    emp_cov = pearson_corr_via_fechner(data)
    _, emp_prec = sklearn_glasso(emp_cov, lasso_param)
    return emp_prec

def graphical_lasso_CV(data):
    gl = GraphicalLassoCV().fit(data)
    return gl.get_precision()

def mincovdet(data):
    mcd = MinCovDet().fit(data)
    return mcd.get_precision()

algorithm_table = {'glasso': graphical_lasso,
                   'glasso_via_Kendall': graphical_lasso_via_Kendall,
                   'glasso_via_Fechner': graphical_lasso_via_Fechner,
                   'glasso_cv': graphical_lasso_CV,
                   'mcd': mincovdet}