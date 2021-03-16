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

        # Example:
        # A(A(_,_),B(_,_)) ->
        # A(A(p,_),B(_,_)) ->
        # A(A(p,q),B(_,_)) ->
        # A(A(p,q),B(A(_,_),_)) ->

        # to which operator does the argument of the next addition belong
        # e.g. A in 'O(p,A(q,_))'
        self.operator_next = ''
        # match all operators
        operators_in_sym = re.finditer(r'(?<=o)[a-zA-Z]*(?=\()', self.symbol)
        op_positions, op_symbols = zip(*[
            (m.group(0), m.start(0) for m in operators_in_sym)
        ])

        # whether the locus of next addition is a second argument
        # AND THE FIRST ARGUMENT IS AN UNSATURATED OPERATOR
        self.second_argument_next = False
        # NOTE: double check this
        # Store the index, in self.symbol, of the next _ to fill.
        # If there is an unfilled argument after a filled argument
        # fill the unfilled argument
        if (self.index_next:=self.symbol.find('),_') != -1:
            # the actual index of the _ is two chars after the )
            self.index_next += 2
            self.second_argument_next = True
        else:
            # if that's not found
            # fill the first unfilled argument
            # if formula is saturated,
            # there are no '_' and self.index_next is -1
            self.index_next = self.symbol.find('_')


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
        # # Unfortunately asserts slow down code
        # assert self.index_next != -1, 'Argh!'
        newsymbol = (
            self.symbol[:self.index_next] +
            formula.symbol +
            self.symbol[self.index_next+1:]
        )
        newformula = Formula(newsymbol)
        return newformula

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
                # find out whether I am adding to first or second argument
                # and find out to which op's argument I am adding
                first_to_apply_index = 0
                if (current_node.second_argument_next and 
                        (current_node.operator_next in symmetric_ops)):
                    # index of the first operator to apply 
                    # in this session
                    current_first = 
                    first_to_apply_index = self.formulas.index(current_first)
                # self.formulas includes both saturated and unsaturated
                for a in self.formulas[first_to_apply_index:]:
                    new_formula = current_node.apply(a)
                    if new_formula.saturated:
                        # add to minimal formulas
                        # if it's not there already
                        self.try_add_minimal_formula(new_formula)
                        # TODO: also add expressions which are
                        # equivalent for the neural nets,
                        # e.g. if new_formula has meaning 0101,
                        # add 1010. 
                    else:
                        if 'oN(oN' not in new_formula.symbol:
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
