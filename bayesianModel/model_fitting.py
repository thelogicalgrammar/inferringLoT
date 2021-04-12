import pickle
from os import path
import numpy as np
import pymc3 as pm
from pprint import pprint
import argparse
import pandas as pd


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
    coords = {
        'cat': np.arange(len(LoT_lengths)),
        'obs': np.arange(len(category_i))
    }
    
    with pm.Model(coords=coords) as model:

#         # SET DATA
#         LoT_lengths_data = pm.Data('LoT_lengths', LoT_lengths, dims='cat')
#         cat_i_data = pm.Data('cat_i', category_i, dims='obs')
        out_i_data = pm.Data('outcome_i', outcome_i, dims='obs')
        length_i_data = pm.Data('length_i', length_i, dims='obs')
        

        # SAMPLE PRIOR
        sigma = pm.HalfNormal('sigma', sigma=100)
        a_0 = pm.Normal('a_0', mu=0, sigma=100)
        a_1 = pm.Normal('a_1', mu=0, sigma=100)
        
        pm.Normal(
            'outcome', 
            mu=a_0+a_1*length_i_data, 
            sigma=sigma, 
            observed=out_i_data,
            dims='obs'
        )
        
    return model


def sample_NUTS(model):
    with model:
        trace = pm.sample(
            1000, 
            cores=1, 
    #         init='advi+adapt_diag',
            return_inferencedata=False,
            target_accept=0.95
        )
    return {'trace': trace}


def fit_variational(model):
    with model:
        fit = pm.fit(method='fullrank_advi')
    return {'fit': fit}
        

def sample_smc(model):
    with model:
        trace_smc = pm.sample_smc(
            500, 
            parallel=False
        )
    return {'trace': trace_smc}


if __name__=='__main__':

    parser = argparse.ArgumentParser()
    parser.add_argument(
        '--indexLoT',
        type=int
    )
    parser.add_argument(
        '--sampler',
        choices=['VI', 'NUTS', 'SMC'],
        default='SMC',
        type=str
    )
    parser.add_argument(
        '--path_L',
        default='lengths_data.npy',
        type=str
    )
    parser.add_argument(
        '--path_learningdata',
        default='../neuralNetsLearning/learning_costs.pkl',
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
    LoT_lengths = L_extended[args.indexLoT]

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
        returnvalue = sampler_func(model)
        storevalue = {
            'model': model,
            **returnvalue
        }

        filename = f'sampler-{args.sampler}_LoT-{args.indexLoT}.pkl'
        with open(filename, 'wb') as openfile:
            pickle.dump(storevalue, openfile)
        if args.sampler=='SMC':
            with open('loglik_'+filename, 'wb') as openfile:
                loglik = returnvalue['trace'].report.__dict__
                pickle.dump(loglik, openfile)
    else:
        print('All values are -1')
