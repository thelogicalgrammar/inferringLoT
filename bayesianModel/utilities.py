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


if __name__=='__main__':
    print("hello!")


