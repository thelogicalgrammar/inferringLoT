from language_minimizer import Inventory
from utilities import define_properties, create_database, calculate_all_inventories
from globalvars import NUM_PROPERTIES
from operators import OP_DICT, PROP_DICT
from pprint import pprint

# create_database('database.db', OP_DICT, PROP_DICT)
# define operators formulas
# ops = [
#     'O', 
#     'A', 
#     'N', 
#     'C', 
#     'IC',
#     'B', 
#     'X', 
#     'NA',
#     'NOR',
#     'NC',
#     'NIC',
# ]

ops1 = [
    # 'O', 
    'N', 
    'NC',
    # 'C'
]

ops2 = [
    # 'A',
    'N', 
    'NA',
    'C',
    # 'IC'
]


# create_database('database.db', OP_DICT, PROP_DICT)

# # run model
inv1 = Inventory(ops1)
inv1.calculate_minimal_formulas()
pprint(inv1.minimal_formulas)
# pprint(sorted([a['length'] for a in inv1.minimal_formulas.values()]))

# inv2 = Inventory(ops2)
# inv2.calculate_minimal_formulas()
# # pprint(inv2.minimal_formulas)
# # pprint(sorted([a['length'] for a in inv2.minimal_formulas.values()]))

# for key, form in inv1.minimal_formulas.items():
#     print(f'{key:04b}', form['length'], inv2.minimal_formulas[key]['length'])

# inv.save_in_db()
# print(len(calculate_all_inventories(OP_DICT.keys(), PROP_DICT.keys())))
