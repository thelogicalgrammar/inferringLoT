import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import pymc3 as pm
import theano 
import theano.tensor as tt
import arviz as az
from pprint import pprint
from scipy import stats
import sqlite3 as sql


def create_sample_data():
    sigma = 2
    a_0, a_1 = 0, 1
    # true LoT
    z = 0 

    # shape (# LoTs, # categories)
    a = np.array([
        [5, 6.1, 4.1],
        [3, 2, 1],
        [2, 4, 5],
        [5, 6, 4]
    ])

    # category of the ith observation
    category_i = np.repeat(np.arange(a.shape[1]), 50)
    mu_i = a_0 + a_1 * a[z,category_i]
    outcome_i = np.random.normal(
        loc=mu_i,
        scale=[sigma]*len(mu_i)
    )

    real = {
        'sigma': sigma,
        'a_0': a_0,
        'a_1': a_1,
        'z': z,
        'a': a,
        'category_i': category_i,
        'mu_i': mu_i,
        'outcome_i': outcome_i
    }
    return real


def fit_sample_data(a, category_i, outcome_i,**kwargs):
    """
    Parameters to infer:
        sigma, a_0, a_1, z

    Known values:
        a, category_i, outcome_i
    """

    with pm.Model() as model:

        sigma = pm.HalfNormal('sigma', sigma=2)
        a_0 = pm.Normal('a_0', mu=0, sigma=1)
        a_1 = pm.Normal('a_1', mu=0, sigma=1)
        z = pm.Categorical('z', np.ones(a.shape[0]))

        mu_i = a_0 + a_1 * theano.shared(a)[z][category_i]

        outcome_i = pm.Normal(
            'outcomes',
            mu=mu_i,
            sigma=sigma,
            observed=outcome_i,
        )

        trace = pm.sample(
            cores=1,
            return_inferencedata=True
        )

    return {'model': model, 'trace': trace}


def create_fit_and_store_sample_data():
    data1 = create_sample_data()
    # data_nonmarg = fit_sample_data(**data1)
    data_marg = fit_sample_data_marginalized(**data1)

    # data_nonmarg['real'] = data1
    data_marg['real'] = data1
    # with open('model_and_trace.pickle', 'wb') as buff:
    #     pickle.dump(data_nonmarg, buff)
    with open('model_and_trace_marginalized.pickle', 'wb') as buff:
        pickle.dump(data_marg, buff)


def analyze_marginalized_data():
    with open('model_and_trace_marginalized.pickle', 'rb') as buff:
        data = pickle.load(buff)
    print(data['trace'].posterior)
    xs_cat = data['real']['category_i']
    xs_outcome = data['real']['outcome_i']
    # likelihoods = stats.norm.pdf(
    #     xs,
    #     loc=,
    #     scale=
    # )
    # to_average = np.prod(likelihoods, axis=1)
    az.plot_trace(data['trace'])
    plt.show()
    return to_average.mean()


def analyse_sample_data():
    with open('model_and_trace.pickle', 'rb') as buff:
        data = pickle.load(buff)
    pprint(data)
    az.plot_trace(data['trace'])
    plt.show()


if __name__=='__main__':
    create_fit_and_store_sample_data()
    # analyse_sample_data()
    # analyze_marginalized_data()

