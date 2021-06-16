import numpy as np
import argparse
import sys
sys.path.append("../")
sys.path.append("../../")
from global_utilities import LoT_indices_to_operators
from functions import *


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--datasize', 
        required=True,
        type=int, 
        help='Number of objects shown'
    )
    parser.add_argument(
        '--n_participants', 
        required=True,
        type=int, 
        help='Number of participants in each simulated experiment'
    )
    parser.add_argument(
        '--temp', 
        required=True,
        type=float, 
        help='Strength of simplicity preference'
    )
    parser.add_argument(
        '--n_experiments', 
        default=10,
        type=int, 
        help='Number of experiments'
    )

    args = parser.parse_args()
    datasize = args.datasize
    n_participants = args.n_participants
    temp = args.temp
    n_experiments = args.n_experiments

    # get minimal formulas length
    with open('../data/lengths_data.npy', 'rb') as openfile:
        lengths_full = np.load(openfile)
        
    NUM_PROPERTIES = 4
    LoTs_full = LoT_indices_to_operators().values

    categories = np.array([
        [int(a) for a in f'{n:0{2**NUM_PROPERTIES}b}']
        for n in range(0, 2**(2**NUM_PROPERTIES))
    ])
    
    lengths, LoTs = prepare_arrays(lengths_full, LoTs_full)

    # Store for every LoT, for each experiment within that LoT,
    # the posterior over LoTs given the data in that experiment
    results = np.zeros((len(LoTs), n_experiments, len(LoTs)))
    for j, true_LoT in enumerate(LoTs):
        for i in range(n_experiments):
            # logp_LoT_given_behaviour has shape (LoT)
            _, logp_LoT_given_behaviour = calculate_logp_LoT_given_behaviour(
                datasize=datasize, 
                lengths=lengths, 
                LoTs=LoTs, 
                categories=categories, 
                n_participants=n_participants,
                temp=temp, 
                true_LoT=true_LoT[None], 
            )
            results[j,i] = logp_LoT_given_behaviour
        print(j)
    
    filename = (
        f'datasize-{datasize}_'
        f'nparticipants-{n_participants}_'
        f'temp-{temp}_'
        f'nexperiments-{n_experiments}'
    )
    with open(f'./data/{filename}.npz', 'wb') as openfile:
        np.savez_compressed(
            file=openfile,
            LoTs=LoTs,
            results=results
        )
    