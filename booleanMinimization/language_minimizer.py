from operators import OP_DICT, PROP_DICT, OP_TO_N_ARGS, OPS_SYMMETRIC
from globalvars import NUM_PROPERTIES, NEST_LIMIT
from itertools import product, chain, combinations
import re
import sqlite3
from pprint import pprint


# @profile
def formula_constructor(top_symbol, children=None):
    # top symbol is a property
    if top_symbol in PROP_DICT:
        meaning = PROP_DICT[top_symbol]
        symbol = top_symbol
    # top symbol is an operator
    else:
        meaning = OP_DICT[top_symbol](*[a['meaning'] for a in children])
        # concat with + is faster than f-strings
        symbol = top_symbol+'('+','.join([c['symbol'] for c in children])+')'
    formula = {
        'symbol': symbol,
        'length': symbol.count('('),
        'meaning': meaning
    }
    return formula


class Inventory():
    """
    """

    def __init__(self, operator_names):
        self.operator_names = operator_names
        self.properties = [formula_constructor(i) for i in PROP_DICT.keys()]
        # dict of formula objects (i.e. dicts)
        # keys are meanings, values are corresponding formulas
        self.minimal_formulas = {
            a['meaning']: a 
            for a in self.properties
        }
        self.has_all_minimal_formulas = False

    def try_add_minimal_formula(self, formula):
        meaning = formula['meaning']
        if m_new := (meaning not in self.minimal_formulas):
            # if meaning is already in minimal formulas,
            # do not add to minimal formulas
            # and also do not add to layer
            self.minimal_formulas[meaning] = formula
            returnvalue = True

        if len(self.minimal_formulas)==(2**(2**NUM_PROPERTIES)):
            self.has_all_minimal_formulas = True
        return m_new

    def calculate_minimal_formulas(self):
        """
        Algo:
        - Keep a list of layers, 'layers'
        - Keep a dict of minimal formulas with elements {meaning: formula}
        - layers[n] contains all and only the formulas of length n
          that are also minimal formulas
            - This is obtained progressively as the tree is built
        - Layers[0] contains just the bare propositions
        - At step n, for each operator O do the following:
            - if O only takes one arg:
                - apply all elements of layers[n-1]
            - if O takes two args and it's symmetric:
                - Loop through the *combinations* of numbers that sum to n-1. 
                - (this way the formulas in layer n will have length n)
                - For each combination (i,j), loop through the cartesian 
                  product layers[i]xlayers[j] and produce the formulas
                  by applying o to the resulting tuples.
                - As you loop, add to layer n any formula which isn't 
                  yet in the minimal_formulas dict.
            - if O takes two args and is not symmetric:
                - Same as with symmetric, but use permutations
                  instead of combinations.
        """
        layers = [self.properties]
        n = 1
        while (not self.has_all_minimal_formulas) and (n <= NEST_LIMIT):
            new_layer = []
            for op_name in self.operator_names:
                # this in theory should check if it has one argument,
                # but the only one with 1 arg I have is 'N'
                if op_name=='N':
                    # only combine with the previous layer
                    for old_form in layers[-1]:
                        # if the old_form isn't a negation at the top
                        if not old_form['symbol'] == 'N':
                            newform = formula_constructor(op_name, [old_form])
                            # here it's checking that newform expresses
                            # a new meaning and also adding to 
                            # self.minimal_formulas
                            if self.try_add_minimal_formula(newform):
                                new_layer.append(newform)
                else:
                    if op_name in OPS_SYMMETRIC:
                        # loop through all combinations of numbers
                        # the lengths of whose operators sum to level-1
                        # (e.g. if I have (3,4), don't have (4,3) )
                        for (a,b) in zip(range(n//2+1), range(n-1,n//2-1,-1)):
                            for args in product(layers[a], layers[b]):
                                newform = formula_constructor(op_name, args)
                                if self.try_add_minimal_formula(newform):
                                    new_layer.append(newform)
                    else:
                        # loop through all permutations of two numbers
                        # the lengths of whose operators sum to level-1
                        for (a,b) in zip(range(n), range(n-1,-1,-1)):
                            for args in product(layers[a], layers[b]):
                                newform = formula_constructor(op_name,args)
                                if self.try_add_minimal_formula(newform):
                                    new_layer.append(newform)
            n += 1
            layers.append(new_layer)
            print(n, len(new_layer), len(self.minimal_formulas))
            # for m, f in self.minimal_formulas.items():
            #     print(f'{m:04b}', f)
            # print('\n')

    def save_in_db(self, db_path='database.db', con=None):
        """
        Save info about the inventory to an sqlite database
        """
        # get names of the operators in the inventory
        # by stripping self.formulas
        
        if con is None:
            con = sqlite3.connect(db_path)

        cur = con.cursor()

        # self.all_ops contains the list of ops
        # self.has_op_bools contains a list of 
        # binary encoding whether has respective op
        # crucially, the two lists are in the same order
        self.all_ops, self.has_op_bools = zip(*[
            (f'"{op}"', str(int(op in self.operator_names)))
            for op in OP_DICT.keys()
        ])

        ########### add data to the data table
        # NOTE: it only adds the formulas that were calculated.
        # implicitly longer formulas can be dealt with later
        # in the analysis
        arguments = [
            (cat['meaning'], cat['length'], cat['symbol'])
            for cat in self.minimal_formulas.values()
        ]
        add_rows_command = (
            f'INSERT INTO data(' +
            f'{",".join(self.all_ops)}, "category", "length", "formula") \n' +
            f'VALUES({",".join(self.has_op_bools)}, ?,?,?);'
        )
        print(add_rows_command)
        cur.executemany(
            add_rows_command,
            arguments
        )

        ############ change status to "done"
        self.change_status_in_db('d', con, cur)
        con.commit()

    def change_status_in_db(self, newstatus, con, cur):
        command_update_status = (
            'UPDATE status \n' +
            f'SET status="{newstatus}"' +
            ' WHERE ' + 
            ' AND '.join(
                f'{op_name} = {int(op_name in self.operator_names)}'
                for op_name in OP_DICT.keys())
        )
        cur.execute(command_update_status)
        con.commit()

    def __str__(self):
        return ' | '.join([str(f) for f in self.operator_names+self.properties])

