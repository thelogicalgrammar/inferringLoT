import pandas as pd
import sqlite3 as lite


class DataBase:
    def __init__(self, db_path):
        self.path = db_path
        self.lengths, self.formulas, self.status = self.get_db()

    def get_db(self):
        with lite.connect(self.path) as con:
            lengths = pd.read_sql_query('SELECT * FROM lengths', con)
            formulas = pd.read_sql_query('SELECT * FROM formulas', con)
            status = pd.read_sql_query('SELECT * FROM status', con)
        return lengths, formulas, status

if __name__=='__main__':
    db = DataBase('db_numprop-3_nestlim-9.db')
    print(db.lengths)
    print(db.formulas)
    print(db.status)
