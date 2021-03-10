from utilities import define_properties
from globalvars import NUM_PROPERTIES

##############################
# define the functions so that
# eval(formula) works in language_minimizer
# NOTE: restrictions on the names of operators.
#   1. They cannot correspond to python keywords
#   2. They cannot correspond to sqlite keywords
##############################

def O(a,b):
    # OR
    return a | b

def A(a,b):
    # AND
    return a & b

def N(a):
    # NOT
    return (1<<(2**NUM_PROPERTIES)) - 1 - a

def C(a,b):
    # CONDITIONAL
    return N(a)|b

def IC(a,b):
    # INVERTED CONDITIONAL
    return N(b)|a

def B(a,b):
    # BICONDITIONAL
    return N(a^b)

def X(a,b):
    # XOR
    return a^b

def NA(a,b):
    # NAND
    return N(a&b)

def NOR(a,b):
    # NOR
    return N(a|b)

def NC(a,b):
    # NEGATED CONDITIONAL
    # (ONLYA in Wataru's lingo)
    return a&N(b)

def NIC(a,b):
    # NEGATED INVERTED CONDITIONAL
    # (ONLYB in Wataru's lingo)
    return b&N(a)


OP_DICT = {
    'O': O,
    'A': A,
    'N': N,
    'C': C,
    'IC': IC,
    'B': B,
    'X': X,
    'NA': NA,
    'NOR': NOR,
    'NC': NC,
    'NIC': NIC,
}

OP_TO_SYMBOL_DICT = {
    'O': 'O(_,_)',
    'A': 'A(_,_)',
    'N': 'N(_)',
    'C': 'C(_,_)',
    'IC': 'IC(_,_)',
    'B': 'B(_,_)',
    'X': 'X(_,_)',
    'NA': 'NA(_,_)',
    'NOR': 'NOR(_,_)',
    'NC': 'NC(_,_)',
    'NIC': 'NIC(_,_)',
}


#######################
# define the properties
#######################

properties = define_properties(NUM_PROPERTIES)
PROP_DICT = dict()
# starts with p and then follows the alphabet
for i, property_int in enumerate(properties):
    PROP_DICT[chr(112+i)] = property_int
    print(bin(property_int))

