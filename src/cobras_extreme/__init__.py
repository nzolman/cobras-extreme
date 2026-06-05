from pathlib import Path
import os
import sys

_parent_dir = Path(__file__).parents[2]
_kflow_dir = os.path.join(_parent_dir, 'Controlling-Kolmogorov-Flow')
_data_dir = os.path.join(_parent_dir, 'data')

_kflow_data_dir = os.path.join(_data_dir, 'kflow')
_fhn_data_dir = os.path.join(_data_dir, 'fhn')
_mnls_data_dir = os.path.join(_data_dir, 'mnls')


sys.path.append(_kflow_dir)

try:
    import equations, solvers
except ImportError as e:
    print('Error importing Kflow `equations` or `solvers`. Make sure you have the Controlling-Kolmogorov-Flow repo in the correct location and that it has been properly installed.')
