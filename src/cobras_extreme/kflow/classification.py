# import jax.numpy as jnp
import numpy as np
from sklearn.svm import SVC
from sklearn.metrics import roc_auc_score, average_precision_score, confusion_matrix
from sklearn.preprocessing import StandardScaler

import pandas as pd

def window_labels(q, q0, T):
    """
    q: 1D array of shape (N+1,)
    q0: threshold
    T: integer time horizon (inclusive)
    
    Returns:
        labels: 1D array of shape (N+1-T,)
                labels[t] = 1 if max(q[t : t+T+1]) >= q0
    """
    q = np.asarray(q)
    N = q.shape[0] - 1

    # Build sliding windows using stride tricks via jax.lax
    # We create a (N+1-T, T+1) matrix where each row is q[t:t+T+1]
    idx = np.arange(N + 1 - T)[:, None] + np.arange(T + 1)[None, :]
    windows = q[idx]  # shape (N+1-T, T+1)

    # Compute max over each window
    max_vals = np.max(windows, axis=1)
    # Compare to threshold
    labels = (max_vals >= q0).astype(np.int32)
    return labels


def symmetric_window_labels(q, q0, T):
    """
    q: 1D array of shape (N,)
    q0: threshold
    T: integer radius of symmetric window
    
    Returns:
        labels: 1D array of shape (N,)
                labels[t] = 1 if max(q[t-T : t+T+1]) >= q0
                (with boundary handling)
    """
    q = np.asarray(q)
    N = q.shape[0]

    # Build index offsets for symmetric window
    offsets = np.arange(-T, T+1)  # shape (2T+1)
    # Build full index matrix: shape (N, 2T+1)
    idx = np.arange(N)[:, None] + offsets[None, :]

    # Clip out-of-bounds indices
    idx = np.clip(idx, 0, N-1)

    # Gather windows
    windows = q[idx]  # shape (N, 2T+1)

    # Compute max over each symmetric window
    max_vals = np.max(windows, axis=1)

    # Compare to threshold
    labels = (max_vals >= q0).astype(np.int32)
    return labels



def train_svm_classifier(z, labels,
                         train_idx,
                         test_idx,
                         return_svm=False, 
                         svm_kwargs = None):
    if svm_kwargs is None:
        svm_kwargs = dict(kernel='rbf',
                          C=1.0, 
                          gamma='scale', 
                          class_weight = 'balanced',
                          random_state = 0,
                          max_iter=50000,
                          probability=True)
    
    # scale data so that we have fair tolerance comparisons
    scaler = StandardScaler()
    z_train = scaler.fit_transform(z[train_idx])
    z_test = scaler.transform(z[test_idx])
    
    labels_train = labels[train_idx]
    labels_test = labels[test_idx]
    svm = SVC(**svm_kwargs)
    svm.fit(z_train, labels_train)
    
    decision_vals = svm.decision_function(z_test)
    
    auc = roc_auc_score(labels_test, decision_vals)
    ap = average_precision_score(labels_test, decision_vals)
    score = svm.score(z_test, labels_test)
    
    preds = svm.predict(scaler.transform(z))
    cm = confusion_matrix(labels_test, 
                          preds[test_idx], 
                          normalize = 'true')
    tn = cm[0,0]
    tp = cm[1,1]
    fp = cm[0,1]
    fn = cm[1,0]
    
    metrics = {
        'auc': auc,
        'ap': ap,
        'score': score,
        'labels_test_mean': labels_test.mean().item(),
        'labels_train_mean': labels_train.mean().item(),
        'labels_mean': labels.mean().item(),
        'tn': tn,
        'tp': tp,
        'fp': fp,
        'fn': fn,
        'n_iter': svm.n_iter_[0],
    }
    if return_svm: 
        return metrics, svm, scaler
    else:
        return metrics
    