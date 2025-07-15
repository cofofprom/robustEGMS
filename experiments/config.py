GLOBAL_CONFIG = {
    'N': 30,
    'n': [20, 100],
    'd': 0.1,
    'student_dof': 3,
    'n_eps_points': 10,
    'S_sg': 10,
    'S_obs': 10,
    'glasso_reg_param': 0.1,
}

GLOBAL_CONFIG = {key: value if isinstance(value, list) else [value] 
                for key, value in GLOBAL_CONFIG.items()}