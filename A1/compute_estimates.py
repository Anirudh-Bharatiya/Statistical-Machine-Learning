import numpy as np

def compute_mle_estimates(X, y, classes):
    mle_params = {}
    n_features = X.shape[1]
    for c in classes:
        X_c = X[y == c]
        N_c = X_c.shape[0]
        mu_c = np.mean(X_c, axis=0).reshape(-1,1)
        X_centered = X_c - mu_c.T
        sigma_c = (X_centered.T @ X_centered) / N_c
        epsilon = 1e-4
        sigma_c += epsilon * np.eye(n_features)
        prior_c = N_c / len(y)
        mle_params[c] = {'mu': mu_c, 'sigma': sigma_c, 'prior': prior_c}
    return mle_params

def get_pooled_covariance(mle_params):
    first = list(mle_params.keys())[0]
    pooled = np.zeros_like(mle_params[first]['sigma'])
    total_prior = sum(p['prior'] for p in mle_params.values())
    for c in mle_params:
        pooled += (mle_params[c]['prior'] / total_prior) * mle_params[c]['sigma']
    return pooled