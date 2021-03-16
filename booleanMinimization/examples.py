from language_minimizer import Inventory, Formula
from utilities import define_properties, create_database
from globalvars import NUM_PROPERTIES
from operators import OP_DICT, PROP_DICT

# print(not_op
#     .apply(p)
# )
# print(not_op
#     .apply(or_op)
# )
# print(not_op
#     .apply(or_op)
#     .apply(p)
#     .apply(q)
# )


create_database('database.db', OP_DICT, PROP_DICT)
# define operators formulas
or_op = Formula('oO(_,_)')
not_op = Formula('oN(_)')
ops = [not_op, or_op]

# run model
inv = Inventory(*ops)
print(inv)
inv.calculate_minimal_formulas()
print([m.meaning for m in inv.saturated])
print(*inv.saturated)
inv.save_in_db()

