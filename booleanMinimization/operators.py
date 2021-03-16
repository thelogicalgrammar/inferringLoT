from utilities import define_properties
from globalvars import NUM_PROPERTIES

##############################
# define the functions so that
# eval(formula) works in language_minimizer
# NOTE: restrictions on the names of operators.
#   1. They cannot correspond to python keywords
#   2. They cannot correspond to sqlite keywords
##############################

def oO(a,b):
    # OR
    return a | b

def oA(a,b):
    # AND
    return a & b

def oN(a):
    # NOT
    return (1<<(2**NUM_PROPERTIES)) - 1 - a

def oC(a,b):
    # CONDITIONAL
    return N(a)|b

def oIC(a,b):
    # INVERTED CONDITIONAL
    return N(b)|a

def oB(a,b):
    # BICONDITIONAL
    return N(a^b)

def oX(a,b):
    # XOR
    return a^b

def oNA(a,b):
    # NAND
    return N(a&b)

def oNOR(a,b):
    # NOR
    return N(a|b)

def oNC(a,b):
    # NEGATED CONDITIONAL
    # (ONLYA in Wataru's lingo)
    return a&N(b)

def oNIC(a,b):
    # NEGATED INVERTED CONDITIONAL
    # (ONLYB in Wataru's lingo)
    return b&N(a)


OP_DICT = {
    'oO': oO,
    'oA': oA,
    'oN': oN,
    'oC': oC,
    'oIC': oIC,
    'oB': oB,
    'oX': oX,
    'oNA': oNA,
    'oNOR': oNOR,
    'oNC': oNC,
    'oNIC': oNIC,
}

OP_TO_SYMBOL_DICT = {
    'oO': 'oO(_,_)',
    'oA': 'oA(_,_)',
    'oN': 'oN(_)',
    'oC': 'oC(_,_)',
    'oIC': 'oIC(_,_)',
    'oB': 'oB(_,_)',
    'oX': 'oX(_,_)',
    'oNA': 'oNA(_,_)',
    'oNOR': 'oNOR(_,_)',
    'oNC': 'oNC(_,_)',
    'oNIC': 'oNIC(_,_)',
}


#######################
# define the properties
#######################

properties = define_properties(NUM_PROPERTIES)
PROP_DICT = dict()
# starts with p and then follows the alphabet
for i, property_int in enumerate(properties):
    PROP_DICT['o' + chr(112+i)] = property_int
    print(bin(property_int))

