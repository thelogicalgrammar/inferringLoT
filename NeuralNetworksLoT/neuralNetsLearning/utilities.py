import numpy as np
import sqlite3

def create_database(num_props, db_path):
    """
    Creates a database that is meant to contain
    for each category the information about the learning
    curve.     
    """
    try:
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        # create table to contain the learning data
        # for each category
        command = (
            'CREATE TABLE data(\n'
            'category INTEGER, \n'
            'rep INTEGER, \n'
            'epoch INTEGER, \n'
            'batch INTEGER, \n'
            'loss FLOAT \n'
            ');'
        )

        cur.execute(command)

        # create table to contain the status of
        # each category with default value 'w'(aiting)
        command = (
            'CREATE TABLE status(\n'
            'category INTEGER, \n'
            'status STRING DEFAULT "w"\n'
            ');'
        )
        cur.execute(command)

        # insert one row for each category
        # (they automatically get value 'w'
        # in the status columns
        command = (
            'INSERT INTO status (category)\n'
            'VALUES(?);'
        )
        # divide by two because I don't need to 
        # study congruent categories separately
        # e.g. no need to do both 0011 and 1100
        n_categories = (2**(2**num_props))//2
        cur.executemany(
            command,
            # takes a list of tuples
            [(i,) for i in range(n_categories)]
        )

        con.commit()
        con.close()

    except sqlite3.OperationalError:
        print('Database already exists!')


def get_first_unprocessed_category(cur,con):
    command_select_undone_categories = (
        'SELECT category FROM status '
        'WHERE status="w" '
        'LIMIT 1'
    )
    # get the category
    category_int = list(cur.execute(command_select_undone_categories))[0][0]
    print('Starting with category: ', category_int)
    return category_int


def change_status_in_db(cur,con,category_int,newstatus):

    command_update_status = (
        'UPDATE status \n'
        f'SET status="{newstatus}" '
        f'WHERE category={category_int}'
    )
    print(f'Updating category {category_int} to state {newstatus}')
    cur.execute(command_update_status)
    con.commit()


def binary_to_bitlist(n,total):
    return [int(a) for a in f'{n:0{total}b}']
