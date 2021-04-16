import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from glob import glob
from pprint import pprint
from utilities import get_data, get_extended_L_and_effective
from os import path
import lzma


def run_model_comparison(fglob):
    """
    """
    # has {modelname: trace}
    traces_dict = dict()
    for fpath in glob(fglob):
        print('f: ', fpath)
        params = get_params_from_fpath(fpath)
        with lzma.open(fpath, 'rb') as f:
            trace = pickle.load(f)
        traces_dict[params['LoT']] = trace
    comparison_df = az.compare(traces_dict)
    return comparison_df


if __name__=='__main__':
    fglob = 
    comparison_df = run_model_comparison(fglob)
