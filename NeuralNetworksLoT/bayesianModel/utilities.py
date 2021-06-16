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
from os import path


def get_saved_categories(cur):
    """
    Get the categories and how many times they appear,
    as stored in an sqlite3 database
    """
    instruction = (
        'SELECT category, count(category) '
        'FROM data '
        'group by category'
    )
    cur.execute(instruction)
    rows = cur.fetchall()
    df = pd.DataFrame(rows, columns=['cat', 'n'])
    return df


def check_status_individual_cat():
    """
    Change the status of category e.g. 11778 to 'w' so running the model calculates it automatically.
    Run this to change a single category back to 'w' (e.g. if it was unfinished, in this case category 11778):
    ```sql
        UPDATE status 
        SET status="w" 
        WHERE category=11778
    ```
    """
    instruction = (
        'SELECT * FROM status'
    #     'WHERE status != "d"'
    )
    cur.execute(instruction)
    rows = cur.fetchall()
    for row in rows:
        if row[1] != 'd':
            print(row)


def get_learning_info(print_n_rows=False, db_path=None):
    """
    Get information about the table encoding the learning times
    """
    
    if db_path is None:
        db_path='/Users/faust/Desktop/neuralNetsLoT/reps-25_batch_size-8_num_epochs-400.db'
    
    con = sql.connect(db_path)
    cur = con.cursor()

    # Get the names of the tables
    cur.execute("SELECT name FROM sqlite_master WHERE type='table';")
    print(cur.fetchall())

    # Get the column names
    instruction = 'PRAGMA table_info("data")'
    cur.execute(instruction)
    rows = cur.fetchall()
    print(rows)
    
    if print_n_rows:
        # Number of rows:
        res = cur.execute('SELECT COUNT(*) FROM data')
        print(list(res))


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
        