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


def get_data(path_L='../data/lengths_data.npy', path_learningdata='../data/learning_costs.pkl'):
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


def get_extended_L_and_effective(L):
    # add interpretation of each category where 
    # in the input to the neural network 
    # 1 is interpreted as False and 0 as True
    # For instance, category 0000 in category_i
    # would correspond to category 1111 in fliplr(L)
    L_extended = np.concatenate((L,np.fliplr(L)))
    effective_LoTs_indices = np.argwhere(np.all(L_extended!=-1,axis=1)).flatten()
    return L_extended, effective_LoTs_indices


def bitslist_to_binary(list_bits):
    out = 0
    for bit in list_bits:
        out = (out<<1)|int(bit)
    return out


def from_minformula_db_to_usable_array():
    """
    Take the sqlite3 output of the booleanMinimization part of the project
    and convert it into an array with dimensions (LoTs, cats) that
    can be used for the bayesian model.
    """
    db_path = '/Users/faust/Desktop/neuralNetsLoT/db_numprop-4_nestlim-100.db'
    con = sql.connect(db_path)
    cur = con.cursor()
    p = 'SELECT O,A,N,C,B,X,NA,NOR,NC,category,length FROM data'
    cur.execute(p)
    df = cur.fetchall()
    # array containing binary category, category, and length
    bincat_cat_length = np.array([
        [bitslist_to_binary(a[:9]),*a[-2:]]
        for a in df
    ])
    max_lot, max_cat, _ = bincat_cat_length.max(axis=0)
    lengths = np.full((max_lot+1, max_cat+1), -1)
    lengths[bincat_cat_length[:,0],bincat_cat_length[:,1]] = bincat_cat_length[:,2]
    with open('lengths_data.npy', 'wb') as openfile:
        np.save(openfile, lengths)
        
        
def log_mean_exp(A,axis=None):
    A_max = np.max(A, axis=axis, keepdims=True)
    B = (
        np.log(np.mean(np.exp(A - A_max), axis=axis, keepdims=True)) +
        A_max
    )
    return B 


def log_sum_exp(x, axis=None):
    mx = np.max(x, axis=axis, keepdims=True)
    safe = x - mx
    return mx + np.log(np.sum(np.exp(safe), axis=axis, keepdims=True))


def log_normalize(x, axis=None):
    x = np.array(x)
    return x - log_sum_exp(x, axis)


def calculate_p_LoT(traces=None, logliks=None, barplot=False):
    """
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
    print("hello!")


