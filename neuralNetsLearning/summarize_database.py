import sqlite3 as sql
import pandas as pd

"""
Once all the learning data is stored in the database for all categories,
run this function to summarize it and create a new database
with the data synthesized for use in the Bayesian model
"""

db_path = '/Users/faust/Desktop/neuralNetsLoT/reps-25_batch_size-8_num_epochs-400.db'
con = sql.connect(db_path)
cur = con.cursor()

inst = """
    SELECT category, rep, SUM(avgloss) as effort
    FROM 
    (
        SELECT category, rep, epoch, batch, AVG(loss) as avgloss
        FROM data
        GROUP BY category, rep, epoch
    ) as inner_query
    GROUP BY category, rep
"""

df = pd.read_sql_query(inst, con=con)
df.to_pickle('./learning_costs.pkl')
