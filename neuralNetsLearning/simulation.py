from neural_network import CategoryLearner
from utilities import (
    create_database,
    get_first_unprocessed_category,
    change_status_in_db
)
from itertools import product
import sqlite3
import numpy as np

n_properties = 4
# eventually reps*batch_size*num_epochs datapoints are saved
parameters = {
    # how many times each category is taught
    'reps': 25,
    'batch_size': 8,
    # how many times the agents goes 
    # through the whole category
    'num_epochs': 400
}

# objects has shape (2**n_properties, n_properties)
objects = np.array(list(product([0,1],repeat=n_properties)))

# name = '_'.join(f'{key}-{value}' for key,value in parameters.items())
# db_path = f'./{name}.db'
db_path = '/Users/faust/Desktop/neuralNetsLoT/reps-25_batch_size-8_num_epochs-400.db'

# attempt to create the database if it's not there
create_database(n_properties, db_path)

with sqlite3.connect(db_path) as con:

    cur = con.cursor()

    while True:

        # get category as an int
        # to be converted to bin
        category_int = get_first_unprocessed_category(cur, con)

        # change status in database to 'r'(unning)
        change_status_in_db(cur,con,category_int,'r')

        # create the learner and train it on the category repeatedly
        learner = CategoryLearner(
            category_int,
            objects,
            **parameters
        )
        learner.train_on_category()

        # save the learning curve in the database
        learner.save_in_database(con, cur)

        # change the status of the category in the database to 'd'(one)
        change_status_in_db(cur,con,category_int,'d')
