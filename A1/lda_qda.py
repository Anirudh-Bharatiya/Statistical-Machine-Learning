import numpy as np

def lda_discriminant(x, mu, sigma_inv, log_prior):
    w = sigma_inv @ mu
    term1 = w.T @ x.reshape(-1,1)
    term2 = -0.5 * (mu.T @ sigma_inv @ mu)
    return (term1 + term2 + log_prior).item()

def qda_discriminant(x, mu, sigma_inv, log_det, log_prior):
    diff = x.reshape(-1,1) - mu
    term1 = -0.5 * (diff.T @ sigma_inv @ diff)
    term2 = -0.5 * log_det
    return (term1 + term2 + log_prior).item()

def classify(X, mle_params, method='QDA'):
    classes = sorted(mle_params.keys())
    y_pred = []
    all_scores = []

    if method == 'LDA':
        from compute_estimates import get_pooled_covariance
        sigma_shared = get_pooled_covariance(mle_params)
        sigma_inv = np.linalg.inv(sigma_shared)

    for x in X:
        scores = []
        for c in classes:
            mu = mle_params[c]['mu']
            prior = mle_params[c]['prior']
            log_prior = np.log(prior)
            if method == 'LDA':
                score = lda_discriminant(x, mu, sigma_inv, log_prior)
            else:
                sigma = mle_params[c]['sigma']
                s_inv = np.linalg.inv(sigma)
                _, log_det = np.linalg.slogdet(sigma)
                score = qda_discriminant(x, mu, s_inv, log_det, log_prior)
            scores.append(score)
        y_pred.append(classes[np.argmax(scores)])
        all_scores.append(scores)
    return np.array(y_pred), all_scores

def calculate_accuracy(y_true, y_pred):
    """Return the fraction of correctly classified samples."""
    return np.mean(y_true == y_pred)