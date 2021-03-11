from operators import OP_DICT, PROP_DICT
from globalvars import NUM_PROPERTIES, NEST_LIMIT
import re
import sqlite3

class Formula:
    """
    Formula class, implements loads of useful methods
    """
    def __init__(self, symbol):
        self.symbol = symbol
        self.num_arguments = self.symbol.count('_')
        self.saturated = self.num_arguments == 0
        self.length = self.symbol.count('(')
        if self.saturated:
            self.meaning = eval(
                self.symbol, 
                {**OP_DICT, **PROP_DICT}
            )

    def apply(self, formula):
        """
        Create a new formula from self and another formula
        by applying the new formula to the top level
        unsaturated argument
        """
        # TODO: double check if this works in all cases
        newsymbol = self.symbol.replace('),_', f'),{formula.symbol}', 1)
        if self.symbol == newsymbol:
            newsymbol = self.symbol.replace('_', formula.symbol, 1)
        return Formula(newsymbol)

    def check_synonym(self, formula):
        return self.meaning == formula.meaning
    
    def __str__(self):
        if self.saturated:
            return f'{self.symbol} ({self.meaning:04b})'
        return self.symbol


class SetOfFormulas:
    """
    This is the most general class for any set of formulas.
    Layers and inventories are subclasses.
    """
    def __init__(self, *args):
        """
        Parameters
        ----------
        args: list of Formulas
            A list of formula objects
        """
        self.formulas = list(args)

    def __str__(self):
        return ' | '.join([str(f) for f in self.formulas])


class Layer(SetOfFormulas):
    """
    Add a dict that keeps track for every symmetric operator
    of which arguments have been explored as first argument
    so I don't need to also add them as second numbers
    """

    def add_formula(self, formula):
        self.formulas.append(formula)

    def __len__(self):
        return len(self.formulas)


class Inventory(SetOfFormulas):
    """
    self.formulas contains all the primitives
    self.saturated is meant to contain all the saturated formulas
    (which it does once they are calculated)
    self.unsaturated contains all the unsaturated primitives
    """

    def __init__(self, *args):
        super().__init__(*args)

        # define property formulas
        for i in PROP_DICT.keys():
            print(i)
            self.formulas.append(Formula(i))

        self.saturated = [a for a in self.formulas if a.saturated]
        self.unsaturated = [a for a in self.formulas if not a.saturated]
        self.has_all_minimal_formulas = False
        self.operators = [
            re.sub('[_(),]', '', f.symbol)
            for f in self.formulas
        ]

    def try_add_minimal_formula(self, formula):
        # check if it is already in self.saturated
        if not any([formula.check_synonym(a) for a in self.saturated]):
            # add the formula to saturated
            self.saturated.append(formula)
            if len(self.saturated)==(2**(2**NUM_PROPERTIES)):
                self.has_all_minimal_formulas = True

    def calculate_minimal_formulas(self):
        terminal_nodes = Layer(*self.unsaturated)
        level = 1
        while (not self.has_all_minimal_formulas) and (level <= NEST_LIMIT):
            new_nodes = Layer()
            print(len(terminal_nodes))
            for current_node in terminal_nodes.formulas:
                for a in self.formulas:
                    new_formula = current_node.apply(a)
                    if new_formula.saturated:
                        # add to minimal formulas
                        # if it's not there already
                        self.try_add_minimal_formula(new_formula)
                    else:
                        # TODO: add conditions to deal
                        # with symmetric operators
                        if 'n(n' not in new_formula.symbol.lower():
                            new_nodes.add_formula(new_formula)
            terminal_nodes = new_nodes
            level += 1
    
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
            (f'"{op}"', str(int(op in self.operators)))
            for op in OP_DICT.keys()
        ])

        ########### add data to the data table
        # NOTE: it only adds the formulas that were calculated.
        # implicitly longer formulas can be dealt with later
        # in the analysis
        arguments = [
            (cat.meaning, cat.length, cat.symbol)
            for cat in self.saturated
        ]
        add_rows_command = (
            f'INSERT INTO data({",".join(self.all_ops)}, "category", "length", "formula") \n' +
            f'VALUES({",".join(self.has_op_bools)}, ?,?,?);'
        )
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
                f'{op_name} = {int(op_name in self.operators)}'
                for op_name in OP_DICT.keys())
        )
        cur.execute(command_update_status)
