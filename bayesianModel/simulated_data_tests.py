import pickle
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
import seaborn as sns
import arviz as az
from pprint import pprint
from scipy import stats
import matplotlib.cm as cm


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


def plot_simulated_data_raw(sample_data):
    """
    Parameters
    ----------
    sample_data: dict of arrays
        Out of the create_sample_data function
    """
    
    L = sample_data['a']
    category_i = sample_data['category_i']
    outcome_i = sample_data['outcome_i']
    
    df = pd.DataFrame({'cat':category_i,'outcome':outcome_i})
    
    fig, ax = plt.subplots()

    df.plot(
        kind='scatter',
        x='outcome',
        y='cat',
        yticks=np.arange(3),
        s=2,
        ax=ax
    )

    color=cm.rainbow(np.linspace(0,1,len(L)+1))

    for i, LoT_costs in enumerate(L):
        ax.scatter(
            x=LoT_costs,
            y=np.arange(len(LoT_costs)),
            c=[color[i]]*len(LoT_costs),
        )


if __name__=='__main__':
    pass
