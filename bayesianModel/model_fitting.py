import pickle
from os import path
import numpy as np
import pymc3 as pm
from pprint import pprint
import argparse
import pandas as pd
import arviz as az
import lzma


def get_data(path_L, path_learningdata):
    """
    Get the data, i.e. L and the learning data
    """

    # L has shape (LoT, cat)
    # where the LoT index encodes the LoT
    # in the way described by the 
    # encoding file
    L = np.load(path_L)

    # note that learning costs
    # are only calculated for half of the categories
    learning_data = pd.read_pickle(path_learningdata)
    category_i, _, cost_i = learning_data.values.T

    return L, category_i, cost_i


def define_model(LoT_lengths, category_i, outcome_i):
    """
    Parameters
    ----------
    LoT_lengths: array
        Has shape (# categories). Contains the formula length for each cat
    category_i: array
        Has shape (# observations). Contains category of each observation.
    outcome_i: array
        Has shape (# observations). Contains outcome for each observation.
    """
    
    length_i = LoT_lengths[category_i]
    
    with pm.Model() as model:

        # SAMPLE PRIOR
        sigma = pm.HalfNormal('sigma', sigma=100)
        a_0 = pm.Normal('a_0', mu=0, sigma=100)
        a_1 = pm.Normal('a_1', mu=0, sigma=100)
        
        pm.Normal(
            'outcome', 
            mu=a_0+a_1*length_i, 
            sigma=sigma, 
            observed=outcome_i
        )
        
    return model


def sample_NUTS(model, filename):
    with model:
        trace = pm.sample(
            1000, 
            cores=1, 
    #         init='advi+adapt_diag',
            return_inferencedata=True,
            target_accept=0.95
        )
    print('Saving the trace')
    with lzma.open(filename+'.xz', 'wb') as f:
        pickle.dump(trace_smc, f)
    return trace


def fit_variational(model, filename):
    with model:
        fit = pm.fit(
            n=50000,
            method='advi'
        )
    
    print('Saving the fit')
    with lzma.open(filename+'.xz', 'wb') as f:
        pickle.dump(trace_smc, f)
        
    return fit
        

def sample_smc(model, filename):
    with model:
        trace_smc = pm.sample_smc(
            n_steps=1000, 
            n_chains=8,
#             parallel=False
        )
    # trace = az.from_pymc3(trace_smc, model=model)
    
    print('Saving the trace')
    with lzma.open(filename+'.xz', 'wb') as f:
        pickle.dump(trace_smc, f)

    #### save everything
    # print('Saving trace to netcfd')
    # trace.to_netcdf(filename+'.nc')

    print('Saving loglik')
    with open('report_'+filename+'.pkl', 'wb') as openfile:
        loglik = trace_smc.report.__dict__
        pickle.dump(loglik, openfile)

    return trace_smc


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--indexLoT',
        type=int
    )
    parser.add_argument(
        '--useEffectiveIndex',
        help=(
            'The LoTs are stored in the file so that '
            'at many indices there are only -1s. '
            'In this case indexLoT range from 0 to 1023. '
            'When useEffectiveIndex is 1 (True), the indices '
            'with -1 are excluded, and indexLoT ranges from 0 to 837. '
            'Useful when there is a limit in the server on '
            'the number of jobs in a batchjob!'
        ),
        type=int,
        default=1
    )
    parser.add_argument(
        '--sampler',
        choices=['VI', 'NUTS', 'SMC'],
        default='SMC',
        type=str
    )
    parser.add_argument(
        '--path_L',
        default='../data/lengths_data.npy',
        type=str
    )
    parser.add_argument(
        '--path_learningdata',
        default='../data/learning_costs.pkl',
        type=str
    )
    args = parser.parse_args()

    L, category_i, cost_i = get_data(
        args.path_L,
        args.path_learningdata
    )

    # add interpretation of each category where 
    # in the input to the neural network 
    # 1 is interpreted as False and 0 as True
    # For instance, category 0000 in category_i
    # would correspond to category 1111 in fliplr(L)
    L_extended = np.concatenate((L,np.fliplr(L)))
    
    if bool(args.useEffectiveIndex):
        print('Using effective index for LoT')
        effective_LoTs_indices = np.argwhere(np.all(L_extended!=-1,axis=1)).flatten()
        try:
            indexLoT = effective_LoTs_indices[args.indexLoT]
        except IndexError:
            print('indexLoT is too high: use smaller index')
            raise
    else:
        indexLoT = args.indexLoT
        
    try:
        LoT_lengths = L_extended[indexLoT]
    except IndexError:
        print('Maybe you meant to use effective index? See help.')
        raise

    if np.all(LoT_lengths!=-1):

        model = define_model(
            LoT_lengths,
            category_i.astype(int),
            cost_i
        )

        sampler_func = {
            'NUTS': sample_NUTS,
            'VI': fit_variational,
            'SMC': sample_smc
        }[args.sampler]

        filename = f'sampler-{args.sampler}_LoT-{args.indexLoT}'
        trace = sampler_func(model, filename)
    else:
        print('All values for the specific LoT are -1')
