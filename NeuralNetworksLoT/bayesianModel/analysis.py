import pickle
import numpy as np
import matplotlib.pyplot as plt
import seaborn as sns
import arviz as az
from glob import glob
from pprint import pprint
import sys
sys.path.append("../../")
from global_utilities import (
    get_data, 
    get_extended_L_and_effective, 
    get_params_from_fpath,
    log_mean_exp,
    log_normalize,
    log_sum_exp
)
from os import path
import lzma


def run_model_comparison(fglob):
    """
    Run loo based model comparison
    """
    # has {modelname: trace}
    traces_dict = dict()
    for fpath in glob(fglob):
        print('f: ', fpath)
        params = get_params_from_fpath(fpath)
        with open(fpath, 'rb') as f:
            trace = pickle.load(f)
        traces_dict[params['LoT']] = trace
    comparison_df = az.compare(traces_dict)
    return comparison_df


def run_single_loo(fname):
    print('Going to analyse: ', fname)
    params = get_params_from_fpath(fname)
    with open(fname, 'rb') as f:
        trace = pickle.load(f)
    print(az.loo(trace))
    
    
def run_frequentist_regression(save=False):
    from sklearn.linear_model import LinearRegression
    
    L, category_i, cost_i = get_data()
    L_extended, effective_LoT_indices = get_extended_L_and_effective(L)

    AICs = []
    regs = []
    for effective_index in range(len(effective_LoT_indices)):
        
        real_index = effective_LoT_indices[effective_index]
        lengths_LoT = L_extended[real_index]
        length_i=lengths_LoT[category_i.astype(int)].reshape(-1,1)

        model = LinearRegression()
        reg = model.fit(X=length_i,y=cost_i)
        regs.append(reg)
        y_hat = model.predict(length_i)
        resid = cost_i - y_hat
        sse = sum(resid**2)
        k = 2
        AIC = 2*k - 2 * np.log(sse)
        AICs.append(AIC)
    
    if save:
        with open('freq_reg.pkl', 'wb') as openf:
            pickle.dump([regs, AICs], openf)
    
    return regs, AICs


def get_SMC_logliks(fglob=None):
    if fglob is None:
        fglob = r'C:\Users\faust\Desktop\neuralNetsLoT\logliks\*.pkl'
    logliks = dict()
    for fpath in glob(fglob):
        params = get_params_from_fpath(fpath)
        with open(fpath,'rb') as openfile:
            data = pickle.load(openfile)
        LoT = int(params['LoT'])
        logliks[LoT] = data['log_marginal_likelihood']
    return logliks


def get_ELBOs(path_vi_glob, n_obs=819200, save=False):
    elbos = dict()
    for fpath in glob(path_vi_glob):
        print(f'Doing path {fpath}')
        # TODO: get actual params
        params = get_params_from_fpath(fpath)
        with lzma.open(fpath, 'rb') as openf:
            fit = pickle.load(openf)['fit']
        elbos[params['LoT']] = calculate_mean_elbo(fit,n_obs)
    if save:
        with open('./elbos.pkl', 'wb') as openf:
            pickle.dump(elbos, openf)
    return elbos
    
    
def calculate_mean_elbo(fit, n_obs):
    # Calculate the average ELBO
    # across minibatches
    # and multiply by the number of minibatches
    # minibatch_adjustment should be 409.6
    minibatch_adjustment = n_obs/2000
    mean_elbo = -np.mean(fit.hist[50000:]) * minibatch_adjustment
    return mean_elbo


def calculate_p_LoT(traces=None, logliks=None, barplot=False):
    """
    Get the probability of each LoT from SMC results
    Choose either tracted or logliks
    
    Parameters
    ----------
    traces: list
        List of traces from SMC pymc3
    logliks: list or array
        Shape (# LoTs, # loglik estimates)
    """
    assert (traces is not None) or (logliks is not None), 'Specify either traces or logliks!'
    if logliks is None:
        logliks = [a.report.log_marginal_likelihood for a in traces]
    marg_liks = np.exp(log_normalize(log_mean_exp(logliks, axis=1).flatten()))
    if barplot:
        plt.bar(np.arange(len(marg_liks)),marg_liks)
    return marg_liks


if __name__=='__main__':
    # fglob = 
    # comparison_df = run_model_comparison(fglob)
    
#     fname = './serverJobs/sampler-NUTS_LoT-0.pkl'
#     run_single_loo(fname)

    # get ELBOs (path is from the pow of the server)
    get_ELBOs(
        '../files_from_runs_VI/*.xz',
        save=True
    )
