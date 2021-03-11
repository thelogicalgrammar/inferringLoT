import pandas as pd
import sqlite3 as lite


class DataBase:
    def __init__(self, db_path):
        self.path = db_path
        self.get_db()

    def get_db(self):
        with lite.connect(self.path) as con:
            self.data = pd.read_sql_query('SELECT * FROM data', con)
            self.status = pd.read_sql_query('SELECT * FROM status', con)

    def print_with_status(self, s):
        print(self.status[self.status['status']==s])

if __name__=='__main__':
    db = DataBase('db_numprop-4_nestlim-9.db')
    print(db.data)
    print(db.status)
    # db.print_with_status('r')
