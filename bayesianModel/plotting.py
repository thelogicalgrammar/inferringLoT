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


def plot_data_fitted(trace, LoT, category_i, outcome_i, ax=None):
    
    if type(trace) == az.data.inference_data.InferenceData:
        a0_trace = trace.posterior['a_0'].values.flatten()
        a1_trace = trace.posterior['a_1'].values.flatten()
        sigma_trace = trace.posterior['sigma'].values.flatten()
    else:
        a0_trace = trace['a_0'].flatten()
        a1_trace = trace['a_1'].flatten()
        sigma_trace = trace['sigma'].flatten()
    
    if ax is None:
        fig, ax = plt.subplots()
    
    ax.scatter(LoT[category_i.astype(int)], outcome_i)

    xs = np.linspace(0,10,2)
    for a0,a1,s in zip(a0_trace, a1_trace, sigma_trace):
        ax.plot(
            xs,
            a0+a1*xs,
            color='blue',
            alpha=0.05,
            linewidth=1
        )
    
    return fig, ax
    
    
def plot_all_in_folder(fglob, path_L, path_learningdata):
    
    L, category_i, cost_i = get_data(path_L, path_learningdata)
    L_extended, effective_LoT_indices = get_extended_L_and_effective(L)
    print('Starting to plot')
    for fpath in glob(fglob):
        print('f: ', fpath)
        fname = path.splitext(path.basename(fpath))[0] 
        params = dict(s.split('-') for s in fname.split('_'))
        with lzma.open(fpath, 'rb') as f:
            trace = pickle.load(f)
        real_index = effective_LoT_indices[int(params['LoT'])]
        fig, ax = plot_data_fitted(trace, L_extended[real_index], category_i, cost_i)
        fig.savefig(f'./realLoTIndex-{real_index}.png')
        

if __name__=='__main__':
    # These paths are for the server, 
    # i.e. from the point of view of serverJobs
    plot_all_in_folder(
        fglob='*.xz',
        path_L='../../data/lengths_data.npy', 
        path_learningdata='../../data/learning_costs.pkl'
    )