import pickle
from os import path
import numpy as np
import pymc3 as pm
from pprint import pprint
import argparse
import pandas as pd
import arviz as az
import theano 
import theano.tensor as T
import lzma
from utilities import get_data, get_extended_L_and_effective


def define_model_joint(L, category_i, outcome_i):
    """
    Parameters
    ----------
    L: array
        Shape (# LoTs, # categories)
    category_i, outcome_i: arrays
        Shape (# observations)
    """
    # TODO: CHANGE THIS!
    
    with pm.Model() as model:
        
        # sample one set of parameters for each LoT
        sigma = pm.HalfNormal('sigma', sigma=5, shape=len(L))
        a_0 = pm.Normal('a_0', mu=0, sigma=5, shape=len(L))
        a_1 = pm.Normal('a_1', mu=0, sigma=5, shape=len(L))
        # sample a true LoT
        z = pm.Categorical('z', np.ones(L.shape[0]))

        mu_i = a_0[z] + a_1[z] * theano.shared(L)[z][category_i]

        outcome_i = pm.Normal(
            'outcomes',
            mu=mu_i,
            sigma=sigma[z],
            observed=outcome_i,
        )

    return model


def define_model_singleLoT(LoT_lengths, category_i, outcome_i):
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


def sample_NUTS(model, filename, cores=4):
    with model:
        trace = pm.sample(
            1000, 
            cores=cores, 
    #         init='advi+adapt_diag',
            return_inferencedata=True,
            target_accept=0.95
        )
    if filename is not None:
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
    if filename is not None:
        print('Saving the fit')
        with lzma.open(filename+'.xz', 'wb') as f:
            pickle.dump(trace_smc, f)
        
    return fit
        

def sample_smc(model, filename):
    with model:
        trace_smc = pm.sample_smc(
#             n_steps=500, 
            chains=6,
            cores=6
        )
    # trace = az.from_pymc3(trace_smc, model=model)
    
    if filename is not None:
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
        '--path_L',
        default='../data/lengths_data.npy',
        type=str
    )
    parser.add_argument(
        '--path_learningdata',
        default='../data/learning_costs.pkl',
        type=str
    )
    
    subparsers = parser.add_subparsers(dest='modelType')
    parser_byLoT = subparsers.add_parser('byLoT')
    parser_joint = subparsers.add_parser('joint')

    ####### add arguments to parser ByLoT
    parser_byLoT.add_argument(
        '--indexLoT',
        type=int,
        required=True
    )
    parser_byLoT.add_argument(
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
    parser_byLoT.add_argument(
        '--sampler',
        choices=['VI', 'NUTS', 'SMC'],
        default='SMC',
        type=str
    )
    
    args = parser.parse_args()
    print(vars(args))

    L, category_i, cost_i = get_data(
        args.path_L,
        args.path_learningdata
    )

    L_extended, effective_LoTs_indices = get_extended_L_and_effective(L)
        
    if args.modelType=='byLoT':
                
        if bool(args.useEffectiveIndex):
            print('Using effective index for LoT')
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

            model = define_model_singleLoT(
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
    
    elif args.modelType=='joint':
        
        define_model_joint(L, category_i, outcome_i)
        trace = pm.sample(
            cores=4,
            return_inferencedata=True,
            nuts={'target_accept':0.99},
            draws=5000,
            chains=5
        )