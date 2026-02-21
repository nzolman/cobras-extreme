import os 
os.environ["JAX_PLATFORM_NAME"] = "cpu" 
import numpy as np
import pandas as pd
import argparse

from cobras_extreme import _kflow_data_dir
from cobras_extreme.kflow.load_utils import load_projs
from cobras_extreme.kflow.classification import train_svm_classifier, window_labels

def get_parser():
    parser = argparse.ArgumentParser('get linear cobras projections')
    parser.add_argument(
            '--Re',
            help='Reynolds number',
            default=40,
            type = float,
    )
    parser.add_argument('--Tf',
                        help='gradient length',
                        default=4,
                        type=int
                        )
    
    parser.add_argument('--res',
                        help='spatial resolution',
                        default=256,
                        type=int
                        )
    parser.add_argument('--ker',
                        help='which kernel to use',
                        default='rbf',
                        type=str
                        )
    return parser


def wrapper(data, z_key, T_pred = 1, n_modes=1, ker = 'rbf'):
    
    svm_kwargs = dict(
                    C=1.0, 
                    gamma='scale', 
                    class_weight = 'balanced',
                    random_state = 0,
                    probability=True,
                    max_iter=50000,
                )
    if ker == 'poly':
        svm_kwargs['kernel'] = 'poly'
        svm_kwargs['degree'] = 2
        svm_kwargs['coef0'] = 1
    elif ker == 'rbf':
        svm_kwargs['kernel'] = 'rbf'
    else:
        svm_kwargs = None
    
    # restrict to first n_modes
    z = np.array(data[z_key])[:, :n_modes]

    train_idx = np.arange(0, 4000)
    
    # create labels
    energy_key = 'e_disp'
    energy = np.array(data[energy_key])
    e_thresh = energy[train_idx].mean() + 2 * energy[train_idx].std()
    labels = window_labels(np.array(energy), q0=e_thresh, T=T_pred)    
    
    test_idx = np.arange(4000, len(labels)) # remove the last T_pred samples to align with labels
    
    metrics =  train_svm_classifier(z, labels, train_idx, test_idx, False, svm_kwargs=svm_kwargs)
    metrics['n_modes'] = n_modes
    metrics['T_pred'] = T_pred
    metrics['name'] = z_key
    metrics['ker'] = ker
    
    return pd.DataFrame([metrics])


if __name__ == '__main__': 
    import multiprocessing as mp
    import time
    from pprint import pprint
    
    
    tic = time.time()
    
    parser = get_parser()
    args = parser.parse_args()
    config = vars(args) # to dict
    
    pprint(config)
    
    res = int(config['res'])
    Re = int(config['Re'])
    Tf = int(config['Tf'])
    ker = config.get('ker', 'rbf')
    
    data = load_projs(Re=Re, Tf=Tf, res=res)
        
    
    n_modes_list = range(1,20)
    T_pred_list = range(32)
    z_keys = ['z_pod_all', 'z_pod_symm_all', 
              'z_cobras_all', 'z_cobras_symm_all', 
            'z_ker', 'z_ker_symm'
              ]
    n_pool = 20

    
    with mp.get_context("spawn").Pool(n_pool) as pool:

        all_res = pool.starmap(
            wrapper,
            [
                (
                data,
                z_key,
                T_pred,
                n_modes,
                ker)
                for n_modes in n_modes_list
                for T_pred in T_pred_list
                for z_key in z_keys
            ]
    )
        
    
    # setup save
    save_dir = os.path.join(_kflow_data_dir, 'classification', ker)
    os.makedirs(save_dir, exist_ok=True)
    
    # save results as csv
    df = pd.concat(all_res, ignore_index=True)
    df.to_csv(os.path.join(save_dir, f'res={res}_Re={Re}_Tf={Tf}.csv'), index=False)
    
    toc = time.time()
    print(f'Elapsed time: {toc - tic:.2f} seconds')
    print(len(all_res))
    
    pprint(df.head())