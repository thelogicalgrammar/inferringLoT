from language_minimizer import Inventory, Formula
from utilities import define_properties, create_database, bitslist_to_binary
from globalvars import NUM_PROPERTIES, NEST_LIMIT
from operators import OP_DICT, PROP_DICT, OP_TO_SYMBOL_DICT
import sqlite3
import signal
import sys

# Exits gracefully by setting the status of the current job back to 'w'(aiting)
# so that another job can pick it up 
# This is means to deal with the server's cancellation due to time limits
# NOTE: this relies on the fact that there is some time between
# SIGTERM and SIGKILL. As can be seen in the server with 'man sbatch'.
# Moreover, in LISA this time is 30 seconds, 
# as can be seen under 'KillWait' with 'scontrol show config | less'
# 30 seconds is plenty to change the status back.
def terminate_signal(signum, frame):

    print(f'Terminating gracefully. Changing {[o.__str__() for o in ops]} back to state "w" in database')

    with sqlite3.connect(db_name) as con:
        cur = con.cursor()
        inventory.change_status_in_db('w', con, cur)

    sys.exit(0)

# deals with cancellation from running over time in server
signal.signal(signal.SIGTERM, terminate_signal)
# deals with ctrl+c events
signal.signal(signal.SIGINT, terminate_signal)

db_name = f'db_numprop-{NUM_PROPERTIES}_nestlim-{NEST_LIMIT}.db'

# make the database if it's not already there
create_database(db_name, OP_DICT, PROP_DICT)

# connect to the database
with sqlite3.connect(db_name) as con:

    cur = con.cursor()

    while True:

        # find from the database which inventories are still not done
        # and select the first one
        select_unfinished_inventories = (
            'SELECT * ' + 
            'FROM status ' +
            'WHERE status="w" ' +
            'LIMIT 1'
        )
        inv_operators = list(cur.execute(select_unfinished_inventories))[0][:-1]
        # get column names (i.e. the operators)
        # (last column is the status column and is excluded)
        colnames = [a[1] for a in list(cur.execute('PRAGMA table_info("status")'))][:-1]
        # find formulas for all operators in inventory
        ops = [Formula(OP_TO_SYMBOL_DICT[a]) for a,b in zip(colnames, inv_operators) if b==1]
        print(*ops)

        # create inventory
        inventory = Inventory(*ops)

        # change the status of the inventory to 'running' in the database
        inventory.change_status_in_db('r', con, cur)
        con.commit()

        # find the minimal formulas
        inventory.calculate_minimal_formulas()

        # save the inventory in the database
        inventory.save_in_db(con=con)
