# Test the function calculate_logp_LoT_given_behaviour_dynamic

import numpy as np
import matplotlib.pyplot as plt

import sys
sys.path.append("../")
from global_utilities import LoT_indices_to_operators
from functions import calculate_logp_LoT_given_behaviour_dynamic
import functions

try:
    # get minimal formulas length
    with open('../data/complete_lengths.npy', 'rb') as openfile:
        lengths = np.load(openfile)
except FileNotFoundError:
    # get minimal formulas length
    with open(
            '/mnt/c/Users/faust/Documents/LoTNeuralNets/ANN_complexity/data/complete_lengths.npy',
            'rb') as openfile:
        lengths_full = np.load(openfile)
        
        
NUM_PROPERTIES = 4
LoTs_full = LoT_indices_to_operators()
argsort_by_N = LoTs_full['N'].argsort()
# lengths_full has the lengths for ALL functionally complete LoTs.
# re-order so that all first part of the array is when N is False
# which means that in the reduced array all the cases where 
# there are two LoTs, one with and one without N,
# the one without N is the one that appears in the reduced version
lengths, LoTs = functions.prepare_arrays(
    lengths_full[argsort_by_N], 
    LoTs_full.iloc[argsort_by_N].values
)
print('Prepared the lengths and LoTs arrays')

categories = np.array([
    [int(a) for a in f'{n:0{2**NUM_PROPERTIES}b}']
    for n in range(0, 2**(2**NUM_PROPERTIES))
])

_, logp_LoT_given_behaviour = functions.calculate_logp_LoT_given_behaviour_dynamic(
    lengths=lengths, 
    LoTs=LoTs, 
    categories=categories, 
    n_participants=10, 
    temp=3., 
    true_LoT=100
)