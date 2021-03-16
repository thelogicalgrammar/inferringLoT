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
        {'oNOR',},
        {'oNA',},
        # two elements
        {'oO', 'oN'},
        {'oA', 'oN'},
        {'oC', 'oN'},
        {'oIC', 'oN'},
        # {'oC', 'oF'},
        # {'oIC', 'oF'},
        {'oC', 'oXOR'},
        {'oIC', 'oXOR'},
        {'oC', 'oNC'},
        {'oC', 'oNIC'},
        {'oIC', 'oNC'},
        {'oIC', 'oNIC'},
        {'oNC', 'oN'},
        {'oNIC', 'oN'},
        # {'oNC', 'oT'},
        # {'oNIC', 'oT'},
        {'oNC', 'oB'},
        {'oNIC', 'oB'},
        # three elements
        # {'oO', 'oB', 'oF'},
        {'oO', 'oB', 'oXOR'},
        # {'oO', 'oXOR', 'oT'},
        # {'oA', 'oB', 'oF'},
        {'oA', 'oB', 'oXOR'},
        # {'oA', 'oXOR', 'oT'}
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

        command_create_table_status = (
            'CREATE TABLE status(\n' +
            # create a column for each operator
            # which will hold boolean values
            ', \n'.join(
                f'{op_name} INTEGER'
                for op_name in op_names
            ) + ', \n' +
            'status STRING DEFAULT "w"\n);'
        )
        print(command_create_table_status)
        cur.execute(command_create_table_status)
        
        # add one row for each inventory
        # in the status table
        op_columns = f'({", ".join([f"{op_name}" for op_name in op_names])})' 
        command_add_inventories_to_status = (
            f'INSERT INTO status {op_columns} '
            f'VALUES ({", ".join("?" for _ in op_names)})'
        )
        arguments = [
            tuple([int(op in inv) for op in op_names]) 
            for inv in all_inventories
        ]
        print(command_add_inventories_to_status)
        cur.executemany(
            command_add_inventories_to_status,
            arguments
        )

        # create data table
        command_create_table_data = (
            'CREATE TABLE data(\n' +
            # create a column for each operator
            # which will hold boolean values
            ', \n'.join(
                f'{op_name} INTEGER'
                for op_name in op_names
            ) 
            + ', \n' +
            'category INTEGER,\n'+
            'length INTEGER,\n'+
            'formula STRING \n'
            + ');'
        )
        cur.execute(command_create_table_data)
        print(command_create_table_data)
        con.commit()
        con.close()

    except sqlite3.OperationalError:
        print('Database already exists!')

