import numpy as np
import argparse
import sys
import os
sys.path.append("../")
sys.path.append("../../")
from global_utilities import LoT_indices_to_operators
from functions import *
from pprint import pprint
import time


if __name__=='__main__':
    
    parser = argparse.ArgumentParser(description='Process some integers.')
    parser.add_argument(
        '--datasize', 
        # required=True,
        type=int, 
        help='Number of objects shown. Only used for scattershot experiments.'
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
        default=1,
        type=int, 
        help='Number of experiments'
    )

    parser.add_argument(
        '--complete_lengths_path', 
        default='../../data/complete_lengths.npy',
        type=str, 
        help=(
            'Path of file containing minimal formula lengths '
            'for ALL functionally complete categories.'
            '(NOTE: including all dual LoTs)'
        )
    )
    
    parser.add_argument(
        '--type_experiment', 
        default='scattershot',
        type=str, 
        help=(
            'Type of experiment. Implemented: \n'
            'scattershot: Participants are asked to categorize all unseen datapoints \n'
            'dynamic: participants categorize and get feedback on each trial;'
            ' next stim is chosen with greedy Bayesian exp design \n' 
            'serial: '
        )
    )

    args = parser.parse_args()
    n_participants = args.n_participants
    temp = args.temp
    n_experiments = args.n_experiments
    type_experiment = args.type_experiment
    datasize = args.datasize
    
    pprint(vars(args))

    # get minimal formulas length
    with open(args.complete_lengths_path, 'rb') as openfile:
        lengths_full = np.load(openfile)
    
    print('Got lengths full')
        
    NUM_PROPERTIES = 4
    LoTs_full = LoT_indices_to_operators()
    argsort_by_N = LoTs_full['N'].argsort()
    
    # lengths_full has the lengths for ALL functionally complete LoTs.
    # re-order so that all first part of the array is when N is False
    # which means that in the reduced array all the cases where 
    # there are two LoTs, one with and one without N,
    # the one without N is the one that appears in the reduced version
    lengths, LoTs = prepare_arrays(
        lengths_full[argsort_by_N], 
        LoTs_full.iloc[argsort_by_N].values
    )
    
    print("LoTs shape: ", LoTs.shape)
    print('Prepared the lengths and LoTs arrays')
    
    categories = np.array([
        [int(a) for a in f'{n:0{2**NUM_PROPERTIES}b}']
        for n in range(0, 2**(2**NUM_PROPERTIES))
    ])
    
    start_time = time.time()
    
    for j, true_LoT in enumerate(LoTs):
        
        filename = (
            f'datasize-{datasize}_' if (datasize is not None) else '' +
            f'nparticipants-{n_participants}_'
            f'temp-{temp}_'
            f'nexperiments-{n_experiments}_'
            f'type-{type_experiment}_'
            f'LoT-{j}'
        )
        
        filepath = f'./data/{filename}.npz'
        
        # if filename already exists from e.g. a previous run
        # of the model, don't save
        if os.path.isfile(filepath):
            print(f"File {filename} exists")
            continue
        
        iteration_start_time = time.time()
        histories_LoT = []
        # Store for the current LoT, for each experiment within that LoT,
        # the posterior over LoTs given the data in that experiment
        results = []
        for i in range(n_experiments):
            
            if type_experiment == "scattershot":
                                
                # logp_LoT_given_behaviour has shape (LoT)
                _, logp_LoT_given_behaviour = calculate_logp_LoT_given_behaviour(
                    datasize=datasize, 
                    lengths=lengths, 
                    LoTs=LoTs, 
                    categories=categories,
                    n_participants=n_participants,
                    temp=temp, 
                    # index of the true LoT
                    true_LoT=true_LoT[None], 
                )
                
            elif type_experiment == "dynamic":
                
                # logp_LoT_given_behaviour has shape (LoT)
                _, logp_LoT_given_behaviour, history = calculate_logp_LoT_given_behaviour_dynamic(
                    lengths=lengths, 
                    LoTs=LoTs, 
                    categories=categories, 
                    n_participants=n_participants, 
                    temp=temp, 
                    index_true_LoT=j
                )
                histories_LoT.append(history)
        
            elif type_experiment == "serial":
                
                # logp_LoT_given_behaviour has shape (LoT)
                _, logp_LoT_given_behaviour, history = calculate_logp_LoT_given_behaviour_serial(
                    lengths=lengths, 
                    LoTs=LoTs, 
                    categories=categories, 
                    n_participants=n_participants, 
                    temp=temp, 
                    index_true_LoT=j
                )
                histories_LoT.append(history)
                
            else:
                raise NotImplementedError(
                    "Specified experiment type not recognized!"
                )
            
            results.append(logp_LoT_given_behaviour)
            
            print("Done with experiment: ", i)
                
        print("Done with LoT: ", j)
        print("For this iteration, seconds: ", time.time()-iteration_start_time)
        print("For all simulation, seconds: ", time.time()-start_time, '\n')

        content_to_save = {
            'LoTs': LoTs,
            'index_true_LoT': j,
            'results': results,
            'histories': histories_LoT
        }
    
        with open(filepath, 'wb') as openfile:
            np.savez_compressed(
                file=openfile,
                **content_to_save
            )
    