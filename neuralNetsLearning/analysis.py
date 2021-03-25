import pandas as pd
import sqlite3

cn = sqlite3.connect(r'reps-10_batch_size-4_num_epochs-500.db')
df = pd.read_sql_query('SELECT * FROM data', cn)
print(df)
