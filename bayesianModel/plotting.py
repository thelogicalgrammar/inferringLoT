import pickle
import numpy as np
import matplotlib.pyplot as plt
import matplotlib.patches as mpatches
import seaborn as sns
import arviz as az
import pandas as pd
from glob import glob
from pprint import pprint
from utilities import (
    get_data, 
    get_extended_L_and_effective, 
    get_params_from_fpath, 
    LoT_indices_to_operators,
    log_mean_exp
)
from os import path
import lzma


def plot_reg_line(length_i, reg, ax, color):
    """
    Parameters
    ----------
    y_hat: array
        Array of predicted mean for each category
        obtained from analysis.run_frequentist_regression
    """
    a, b = reg.intercept_, reg.coef_
    xs = np.linspace(0,length_i.max(),10)
    ax.plot(
        xs, 
        a+xs*b,
        color=color
    )
    return ax


def plot_data(length_i, cost_i, ax=None, style='violin'):
    """
    Plots a violinplot for each category 
    length_i should be L_extended[real_index][category_i]
    """
    
    if ax is None:
        fig, ax = plt.subplots()
        
    df = pd.DataFrame({
        'x': length_i.flatten(), 
        'y': cost_i
    })
    if style=='violin':
        sns.violinplot(
            data=df,
            y='y',
            x='x',
            scale='count',
            color='lightblue',
            ax=ax
        )
    elif style=='scatter':
        ax.scatter(length_i, outcome_i)
    else:
        raise InputError('Style not recognized')
    return ax


def plot_data_fitted(trace, x_max, color, fig_ax=None):
    """
    Parameters
    ----------
    x_max: int
        It's meant to be the max cat length
    """
    
    if type(trace) == az.data.inference_data.InferenceData:
        a0_trace = trace.posterior['a_0'].values.flatten()
        a1_trace = trace.posterior['a_1'].values.flatten()
        sigma_trace = trace.posterior['sigma'].values.flatten()
    else:
        a0_trace = trace['a_0'].flatten()
        a1_trace = trace['a_1'].flatten()
        sigma_trace = trace['sigma'].flatten()
    
    fig, ax = fig_ax or plt.subplots()

    xs = np.linspace(0,x_max,2)
    for a0,a1,s in zip(a0_trace, a1_trace, sigma_trace):
        ax.plot(
            xs,
            a0+a1*xs,
            color=color,
            alpha=0.05,
            linewidth=1,
        )
    
    return fig_ax
    
    
def plot_all_in_folder(path_L, path_learningdata, folder_smc=None, folder_vi=None, path_freq_reg=None):
    """
    Parameters
    ----------
    folder_smc,folder_vi: None or str
        Folder containing the traces of SMC and VI runs
    freq_reg: None or list
        Output of analysis.run_frequentist_regression()
    """
    
    L, category_i, cost_i = get_data(path_L, path_learningdata)
    L_extended, effective_LoT_indices = get_extended_L_and_effective(L)
    ops_list = LoT_indices_to_operators(
        effective_LoT_indices,
        use_whole_effective_LoT_indices=True, 
        return_nested_list=True
    )
    print('Starting to plot')
    
    if path_freq_reg is not None:
        with open(path_freq_reg, 'rb') as openf:
            regs, AICs = pickle.load(openf)
    
    # i is the effective index here
    for i in range(len(L)):
        
        print('Starting i: ', i)
        legend_patches = []
        
        fig, ax = plt.subplots()
        
        real_index = effective_LoT_indices[i]
        ops = ops_list[i]        
        length_i = L_extended[real_index][category_i.astype(int)]
        plot_data(length_i, cost_i, ax=ax, style='violin')
        title = str(ops)
        
        if folder_smc is not None:
            color_smc = 'red'
            fpath_smc = folder_smc+f'/sampler-SMC_LoT-{i}.xz'
            with lzma.open(fpath_smc, 'rb') as f:
                trace_smc = pickle.load(f)
            plot_data_fitted(
                trace_smc,
                x_max=length_i.max(),
                fig_ax=(fig,ax),
                color=color_smc
            )
            fpath_smc_report = folder_smc+f'/report_sampler-SMC_LoT-{i}.pkl'
            with open(fpath_smc_report,'rb') as openfile:
                data = pickle.load(openfile)
            loglik_smc = data['log_marginal_likelihood']
            title += f'\n SMC loglik: {log_mean_exp(loglik_smc)}'
            legend_patches.append(mpatches.Patch(color=color_smc, label='SMC'))
                
        if folder_vi is not None:
            color_vi = 'green'
            fpath_vi = folder_vi+f'/sampler-VI_LoT-{i}.xz'
            with lzma.open(fpath_vi, 'rb') as f:
                fit_data = pickle.load(f)
            fit = fit_data['fit']
            trace_vi = fit.sample(1000)
            plot_data_fitted(
                trace_vi,
                x_max=length_i.max(),
                fig_ax=(fig,ax),
                color=color_vi
            )
            legend_patches.append(mpatches.Patch(color=color_vi, label='VI'))
            
            # Calculate the average ELBO
            # across minibatches
            # and multiply by the number of minibatches
            # minibatch_adjustment should be 409.6
            minibatch_adjustment = len(cost_i)/2000
            mean_elbo = -np.mean(fit.hist[50000:]) * minibatch_adjustment
            title += f'\n ELBO: {mean_elbo}'
        
        if path_freq_reg is not None:
            color_freq_reg = 'blue'
            plot_reg_line(
                length_i, 
                regs[i], 
                ax, 
                color=color_freq_reg
            )
            legend_patches.append(
                mpatches.Patch(color=color_freq_reg, label='FREQ_REG')
            )
            title+=f'\n AIC: {AICs[i]}'
            
        if legend_patches:
            plt.legend(handles=legend_patches)
        ax.set_title(title)
        plt.tight_layout()
        fig.savefig(f'./realLoTIndex-{real_index}.png')
        

if __name__=='__main__':
    
    # These paths are for the server, 
    # i.e. from the point of view of serverJobs
    plot_all_in_folder(
        path_L='../../data/lengths_data.npy', 
        path_learningdata='../../data/learning_costs.pkl',
        folder_smc='../files_from_runs_SMC', 
        folder_vi='../files_from_runs_VI', 
        path_freq_reg='../freq_reg.pkl'
    )
