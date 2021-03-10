from globalvars import NUM_PROPERTIES
from itertools import product, combinations, chain
import sqlite3


def bitslist_to_binary(list_bits):
    out = 0
    for bit in list_bits:
        out = (out<<1)|bit
    return out


def calculate_powerset(iterable):
    s = list(iterable)
    return chain.from_iterable(combinations(s,r) for r in range(len(s)+1))


def calculate_all_inventories(ops, props):
    """
    Parameters
    ----------
    ops, props: list of Formulas
    """

    f_complete = [
        # one element
        {'NOR',},
        {'NA',},
        # two elements
        {'O', 'N'},
        {'A', 'N'},
        {'C', 'N'},
        {'IC', 'N'},
        # {'C', 'F'},
        # {'IC', 'F'},
        {'C', 'XOR'},
        {'IC', 'XOR'},
        {'C', 'NC'},
        {'C', 'NIC'},
        {'IC', 'NC'},
        {'IC', 'NIC'},
        {'NC', 'N'},
        {'NIC', 'N'},
        # {'NC', 'T'},
        # {'NIC', 'T'},
        {'NC', 'B'},
        {'NIC', 'B'},
        # three elements
        # {'O', 'B', 'F'},
        {'O', 'B', 'XOR'},
        # {'O', 'XOR', 'T'},
        # {'A', 'B', 'F'},
        {'A', 'B', 'XOR'},
        # {'A', 'XOR', 'T'}
    ]

    # calculate all combinations of operators
    powerset_ops = calculate_powerset(ops)

    # only include the elements of the powerset that are supersets 
    # of at least one element of f_complete
    all_inventories_formulas = [
        inv 
        for inv in powerset_ops
        if any([set(inv).issuperset(f_comp_inv) for f_comp_inv in f_complete])
    ]

    return all_inventories_formulas


def define_properties(num):
    """
    Return
    ------
    list of ints
        Each int is one propositional variable
    """
    a = list(map(list,zip(
        *[f'{i:0{num}b}' for i in range(2**NUM_PROPERTIES)]
    )))
    b = [[int(j) for j in subarray] for subarray in a]
    list_to_bool = lambda x: sum([b<<i for i, b in enumerate(x)])
    return [list_to_bool(j) for j in b]


def create_database(db_path, op_dict, prop_dict):
    """
    If table exists already, doesn't do anything
    """
    op_names, num_props = op_dict.keys(), NUM_PROPERTIES
    try:
        all_inventories = calculate_all_inventories(
            op_dict.keys(), prop_dict.keys()
        )
        con = sqlite3.connect(db_path)
        cur = con.cursor()

        # this command is the same for all tables
        # and creates one row for each inventory
        command_add_rows_func = lambda tablename: (
            f'INSERT INTO {tablename} ' +
            # add the columns to modify
            '(' +
            ', '.join([
                f'{op_name}'
                for op_name in op_names
            ]) +
            ')' +
            '\n VALUES \n' +
            # add the values for the columns
            # corresponding to the operators
            ', '.join([
                str(tuple([int(op in inv) for op in op_names]))
                for inv in all_inventories
            ])
            + ';'
        )
        
        ###################### CREATE LENGTHS TABLE
        # TODO: make more secure by not using
        # dynamically constructed string for SQL
        command_create_table_lengths = (
            'CREATE TABLE lengths(\n' +
            # create a column for each operator
            # which will hold boolean values
            ', \n'.join(
                f'{op_name} INTEGER'
                for op_name in op_names
            ) 
            + ', \n' +
            # create a column for each category
            # which will hold lengths
            # note that each category is effectively an int
            # that should be converted to bin
            ', \n'.join(
                f'b{cat_name} INTEGER'
                for cat_name in range(0,2**(2**num_props))
            )
            + '\n);'
        )
        cur.execute(command_create_table_lengths)
        # insert one row for each inventory,
        # i.e. one row for each functionally complete
        # combination of operators
        command_add_rows_lenghts = command_add_rows_func('lengths')
        cur.execute(command_add_rows_lenghts)

        ###################### CREATE FORMULAS TABLE

        command_create_table_formulas = (
            'CREATE TABLE formulas(\n' +
            # create a column for each operator
            # which will hold boolean values
            ', \n'.join(
                f'{op_name} INTEGER'
                for op_name in op_names
            ) 
            + ', \n' +
            # create a column for each category
            # which will hold the formulas
            ', \n'.join(
                f'b{cat_name} STRING'
                for cat_name in range(0,2**(2**num_props))
            )
            + '\n);'
        )
        cur.execute(command_create_table_formulas)
        # (see create of lengths table above for explanation)
        command_add_rows_formulas = command_add_rows_func('formulas')
        cur.execute(command_add_rows_formulas)
        ###################### CREATE STATUS TABLE

        command_create_table_status = (
            'CREATE TABLE status(\n' +
            ', \n'.join(
                f'{op_name} INTEGER'
                for op_name in op_names
            ) 
            + ', \n status STRING DEFAULT "w"\n);'
        )
        cur.execute(command_create_table_status)
        # (see create of lengths table above for explanation)
        command_add_rows_status = command_add_rows_func('status')
        cur.execute(command_add_rows_status)
        
        con.commit()
        con.close()

    except sqlite3.OperationalError:
        print('Database already exists!')

