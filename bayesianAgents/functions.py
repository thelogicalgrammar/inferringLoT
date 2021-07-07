import numpy as np
import numexpr as ne

import sys
sys.path.append("../")
sys.path.append("../../")
from global_utilities import LoT_indices_to_operators


def expand_with_equivalent_LoTs(LoTs):
    OP_EQUIV_DICT = {
        'O': 'A',
        'A': 'O',
        'N': 'N',
        # the dual of C is NIC in theory
        # but NIC is equivalent to NC
        # so I am just using NC instead of NIC everywhere
        # to have fewer operators
        'C': 'NC',
    #     'IC': 'NC',
        'B': 'X',
        'X': 'B',
        'NA': 'NOR',
        'NOR': 'NA',
        # the dual of 'NC' in theory is 'IC'
        # but IC is equivalent to C
        # so I am just using C instead of IC everywhere
        # to have fewer operators
        'NC': 'C',
    #     'NIC': 'C'
    }
    new_cols = [OP_EQUIV_DICT[i] for i in LoTs.columns]
    # only return unique rows
    return (
        pd.DataFrame(LoTs.values, columns=new_cols)
        .drop_duplicates()
        .reindex(LoTs.columns, axis=1)
    )


def calculate_complete_lengths(lengths, LoTs):
    """
    To each element of array "lengths" with -1, add the 
    lengths from the corresponding dual LoT.
    The only arrays left with -1 are the ones
    corresponding to functionally incomplete LoTs
    """
    
    index_to_dual = number_to_dual_number(np.arange(65536))
    
    # for each LoTs, save the index of the dual LoT
    index_dual_LoT = []
    for row_index in np.arange(len(lengths)):
        equiv_LoT = expand_with_equivalent_LoTs(LoTs.iloc[row_index:row_index+1])
        index_equivalent = np.argwhere(np.all(LoTs.values==equiv_LoT.values, axis=1))[0][0]
        index_dual_LoT.append(index_equivalent)
    
    index_incomplete = lengths[:,0]==-1
    lengths[index_incomplete] = lengths[index_dual_LoT][:, index_to_dual][index_incomplete]
    return lengths


# def logsumexp(X, axis, keepdims=True):
#     alpha = np.max(X, axis=axis, keepdims=True)  # Find maximum value in X
#     return np.log(np.sum(np.exp(X-alpha), axis=axis, keepdims=keepdims)) + alpha


def logsumexp(X, axis):
    alpha = np.max(X, axis=axis, keepdims=True)  # Find maximum value in X
    a = ne.evaluate('exp(X-alpha)')
    a = np.sum(a, axis=axis, keepdims=True)
    a = ne.evaluate('log(a)+alpha')
    return a 


def log_softmax(array, axis, temp):
    log_unnorm = (temp*-array).astype(np.float64)
    log_norm = log_unnorm - logsumexp(log_unnorm, axis=axis)
    return log_norm


def calculate_logp_accept_object_marginal(lengths, categories, data, temp=3):
    """
    Get marginal probability that the participant will accept each object
    as belonging to the unobserved category given each possible LoT.
    For each LoT, sum across categories:
    (p_category * indicator function of object for that category)
    """
    
    # the prior probability of each category 
    # assuming a softmax simplicity prior
    logp_category_given_LoT = log_softmax(lengths.astype(np.int64), axis=1, temp=temp)
    
    if data.sum()==0:
        logp_data_given_category = np.zeros(len(categories))
    else:
        logp_data_given_category = -data.sum()*np.log(categories.sum(axis=1))

    # calculate the probability of categories (rows)
    # given each LoT and the observed data
    logp_category_given_data = logp_data_given_category + logp_category_given_LoT
    
    logp_category_given_data = (
        logp_category_given_data - 
        logsumexp(logp_category_given_data, axis=1)
    )
    
    # Array has shape (LoTs, categories compatible with data, objects)
    # For each LoT, category and object, array contains the probability of the
    # category to which the object belongs
    # array is -inf when the object is not in the category
    array = np.log(categories[None]) + logp_category_given_data[:,:,None]
    # sum across categories
    # which gives the marginal probability of each object
    # given each LoT
    logp_accept_object_marginal = logsumexp(array, axis=1)

    return logp_accept_object_marginal.squeeze()

    
def simulate_agent_choice(p_accept_object_marginal, index_true_LoT):
    # make the agent accept or reject each object
    # NOTE: some numbers were going *slightly* over 1,
    # E.g. 1.0000000000000024
    # So I am clipping to avoid the problem
    return np.random.binomial(
        n=1, 
        p=np.clip(p_accept_object_marginal[index_true_LoT], 0,1)
    ).flatten()


def calculate_logp_behaviour_given_LoT(logp_accept_object_marginal, agent_behaviour, lengths):
    
    # for the objects accepted by the agent, use the p_accept
    # for the objects rejected by the agent, use 1 - p_accept
    logp_individual_choices_given_LoT = np.where(
        agent_behaviour, 
        logp_accept_object_marginal, 
        np.log(-np.expm1(logp_accept_object_marginal))
    )
    
#     agent_rejected = agent_asked - agent_accepted
#     logp_reject_object_marginal = np.log(-np.expm1(logp_accept_object_marginal))

#     logp_behaviour_given_LoT = np.sum(
#         logp_accept_object_marginal*agent_accepted + 
#         logp_reject_object_marginal*agent_rejected,
#         axis=1
#     )
    
    logp_behaviour_given_LoT = np.sum(
        logp_individual_choices_given_LoT, 
        axis=1
    )
    
    return logp_behaviour_given_LoT


def prepare_arrays(lengths_full, LoTs_full, return_indices_ar=False, return_inverse_indices=False):
    """
    Eliminate the functionally incomplete arrays as well as the repetitions
    """
    
    # to reduce the size of the involved arrays
    # reduce the LoT dimensions of the arrays
    # to only have the functionally complete arrays
    functionally_complete_LoTs_indices = np.argwhere(lengths_full[:,1]!=-1).flatten()
    functionally_complete_lengths = lengths_full[functionally_complete_LoTs_indices]
    functionally_complete_LoTs = LoTs_full[functionally_complete_LoTs_indices]
    
    # eliminate the repeated rows in the functionally complete LoTs
    # to get the LoTs that can be in principle distinguished through data
    lengths, indices_ar, inverse_indices, counts = np.unique(
        functionally_complete_lengths, 
        axis=0, 
        return_index=True,
        return_counts=True,
        return_inverse=True
    )
    
    # get the array with the true LoTs
    LoTs = LoTs_full[indices_ar,:]
    
    returnvalues = lengths, LoTs
    if return_indices_ar:
        returnvalues+=(indices_ar,)
    if return_inverse_indices:
        returnvalues+=(inverse_indices,)
    
    return returnvalues


def calculate_logp_LoT_given_behaviour(datasize, lengths, LoTs, categories, n_participants,
                       temp=3, true_LoT=None, data=None):
    """
    Parameters
    ----------
    datasize: int
        The number of objects to show to each participant (different across participants)
        Does nothing if data is specified
    lengths: array
        Array with shape (LoTs, categories)
        Containing the length of the minimal formula for each category in each LoT
    LoTs: array
        Array with shape (LoT, operator)
        each row is an LoT and each row says whether each LoT containg the
        operator corresponding to the column
    categories: array
        Array with shape (category, object)
        that says which object is contained in each category
    n_participants: int
        The number of participants to get data from
    temp: float
        The strength of simplicity preference
    true_LoT: int
        The true LoT for the agents
    data: array
        The data to show to the agents
    """
    
    logp_LoT_given_behaviour = np.zeros(len(LoTs))

    # Doing it participant-wise rather than vectorized
    # because logsumexp, which is the computational bottleneck here,
    # is faster with smaller array.
    # To make arrays faster I remove the indices of the categories
    # incompatible with the data seen by each participant.
    # But since this has to be done *by participant*,
    # I cannot do it in a vectorized way
    # TODO: Think if there is a better way here
    for participant_index in range(n_participants):
        
        if data is None:
            # Boolean vector encoding the objects that
            # are shown to the agent as belonging to the category.
            # (the LoT is then inferred from which other objects
            # the participant thinks belong to the category)
            # There are 16 objects in total.
            data = np.zeros(16, dtype=int)
            data[:datasize] = 1
            np.random.shuffle(data)

        # the true agent's LoT
        # selected at random from the functionally complete LoTs
        if true_LoT is None:
            # index when excluding functionally incomplete
            true_LoT = LoTs[np.random.choice(len(LoTs))]

        # boolean mask of categories compatible with the observed data
        compatible_with_data = np.all(
            np.logical_not(data & np.logical_not(categories)), 
            axis=1
        )
        
        # only consider categories compatible with data
        logp_accept_object_marginal = calculate_logp_accept_object_marginal(
            lengths[:,compatible_with_data], 
            categories[compatible_with_data], 
            data, 
            temp=temp
        )

        agent_behaviour = simulate_agent_choice(
            np.exp(logp_accept_object_marginal), 
            np.argwhere(np.all(LoTs==true_LoT, axis=1))[0][0]
        )

        logp_behaviour_given_LoT = calculate_logp_behaviour_given_LoT(
            logp_accept_object_marginal, 
            agent_behaviour, 
            lengths[:,compatible_with_data]
        )
        
        # accumulated likelihood in unnormalized posterior
        logp_LoT_given_behaviour += logp_behaviour_given_LoT
    
    # find probability of each LoT given the agent's behaviour
    # i.e. which stimuli were accepted and rejected
    logp_LoT_given_behaviour -= logsumexp(logp_LoT_given_behaviour, axis=0)

#     # Re-expand the array to also include the LoTs that are not functionally complete
#     # NOTE: This doesn't work anymore
#     logp_LoT_given_behaviour_full = np.full(len(lengths_full), fill_value=-np.inf)
#     logp_LoT_given_behaviour_full[functionally_complete_LoTs_indices] = logp_LoT_given_behaviour

    return true_LoT, logp_LoT_given_behaviour


def run_simulation_log(datasize, lengths_full, LoTs_full, categories, n_participants,
                       temp=3, true_LoT=None, data=None):
    
    lengths, LoTs = prepare_arrays(lengths_full, LoTs_full)
    return LoTs, *calculate_logp_LoT_given_behaviour(
        datasize=datasize, 
        lengths=lengths, 
        LoTs=LoTs, 
        categories=categories, 
        n_participants=n_participants,
        temp=temp, 
        true_LoT=true_LoT, 
        data=data
    )
