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


def summarize_results(db_path):
    """
    Takes the database with the minimal formulas
    and creates from it a pickle file
    with a pandas dataframe with shape (LoT, category)
    as well as a 
    """
    con = sql.connect(db_path)
    cur = con.cursor()
    df = pd.read_sql(
        'SELECT * FROM data',
        con
    )
    print(df)

    df['invId'] = (
        df.loc[:,'O':'NC']
        .astype(str)
        .agg(''.join, axis=1)
        .apply(lambda x: bitslist_to_binary(list(x)))
    )

    L = (
        df[['category','length','invId']]
        .pivot(index='invId',columns='category',values='length')
    )
    L.to_pickle('L.db')

    df_ids = (
        df.iloc[:,np.r_[:9,-1]]
        .drop_duplicates()
    )
    df_ids.to_pickle('df_ids.db')
    

if __name__=='__main__':
    print("hello!")
    summarize_results(
        # '/Users/faust/Desktop/neuralNetsLoT/db_numprop-4_nestlim-100.db'
        '../../booleanMinimization/serverJobs/db_numprop-4_nestlim-100.db'
    )


