import numpy as np
import pandas as pd

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


def number_to_category(number):
    return [int(n) for n in f'{number:0{16}b}']


def dual_category(categories_array):
    return 1-categories_array[:,::-1]


def category_to_number(categories_array):
    return categories_array.dot(1 << np.arange(categories_array.shape[-1] - 1, -1, -1))


def number_to_dual_number(numbers_list):
    return category_to_number(
        dual_category(
            np.array(list(map(number_to_category, numbers_list)))
        )
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


def logsumexp(X, axis, use_ne=True):
    if use_ne:
        import numexpr as ne
        alpha = np.max(X, axis=axis, keepdims=True)  # Find maximum value in X
        a = ne.evaluate('exp(X-alpha)')
        a = np.sum(a, axis=axis, keepdims=True)
        a = ne.evaluate('log(a)+alpha')
        return a 
    else:
        alpha = np.max(X, axis=axis, keepdims=True)  # Find maximum value in X
        return np.log(np.sum(np.exp(X-alpha), axis=axis, keepdims=True)) + alpha


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
    
    To reduce the size of the involved arrays,
    reduce the LoT dimensions of the arrays
    to only have the functionally complete arrays
    """
    
    # find the indices of the rows in lengths_full where the first element isn't -1
    functionally_complete_LoTs_indices = np.argwhere(lengths_full[:,1]!=-1).flatten()
    # get the corresponding rows from lengths_full
    functionally_complete_lengths = lengths_full[functionally_complete_LoTs_indices]
    # get the corresponding LoTs
    functionally_complete_LoTs = LoTs_full[functionally_complete_LoTs_indices]
    
    print(functionally_complete_LoTs_indices.shape)
    
    # eliminate the repeated rows in the functionally complete LoTs
    # to get the LoTs that can be in principle distinguished through data
    lengths, indices_ar, inverse_indices, counts = np.unique(
        functionally_complete_lengths, 
        axis=0, 
        return_index=True,
        return_counts=True,
        return_inverse=True
    )
    
    # get the array with the LoTs we want
    LoTs = functionally_complete_LoTs[indices_ar,:]
    
    returnvalues = lengths, LoTs
    if return_indices_ar:
        returnvalues += (indices_ar,)
    if return_inverse_indices:
        returnvalues += (inverse_indices,)
    
    return returnvalues


def calculate_logp_LoT_given_behaviour(datasize, lengths, LoTs, categories, n_participants,
                                       temp=3, index_true_LoT=None, data=None):
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
    index_true_LoT: int
        The true LoT for the agents
    data: array
        The data to show to the agents
    
    Returns
    -------
    tuple
        A tuple of two elements. The first element is the true LoT.
        The second element is an array with dimensions (LoT)
        which encodes the posterior probability of each LoT
        given the data from that experiment.
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

        # if no value was given to the true agent's LoT, it is
        # selected at random from the functionally complete LoTs
        if index_true_LoT is None:
            # index when excluding functionally incomplete
            index_true_LoT = np.random.choice(len(LoTs))

        # boolean mask of categories compatible with the observed data
        compatible_with_data = np.all(
            np.logical_not(data & np.logical_not(categories)), 
            axis=1
        )
        
        # Calculate the logprobability to accept 
        # each of the object for all LoTs
        # (only consider categories compatible with data)
        # Dimensions: (LoT, object)
        logp_accept_object_marginal = calculate_logp_accept_object_marginal(
            # for all LoTs, give the length of the categories
            # compatible with the data seen thus far
            lengths[:,compatible_with_data], 
            # give the categories compatible with the data
            categories[compatible_with_data], 
            # give the data this far
            data, 
            temp=temp
        )

        # generate simulated behaviour from the participant
        agent_behaviour = simulate_agent_choice(
            # marginal probability to accept each object
            np.exp(logp_accept_object_marginal), 
            # index of the participant's true LoT
            index_true_LoT
        )

        logp_behaviour_given_LoT = calculate_logp_behaviour_given_LoT(
            logp_accept_object_marginal, 
            agent_behaviour, 
            lengths[:,compatible_with_data]
        )
        
        # accumulated likelihood in unnormalized posterior
        logp_LoT_given_behaviour += logp_behaviour_given_LoT
    
    # normalize to find probability of each LoT given the agent's behaviour
    # i.e. which stimuli were accepted and rejected
    logp_LoT_given_behaviour -= logsumexp(logp_LoT_given_behaviour, axis=0)

#     # Re-expand the array to also include the LoTs that are not functionally complete
#     # NOTE: This doesn't work anymore
#     logp_LoT_given_behaviour_full = np.full(len(lengths_full), fill_value=-np.inf)
#     logp_LoT_given_behaviour_full[functionally_complete_LoTs_indices] = logp_LoT_given_behaviour

    return LoTs[index_true_LoT], logp_LoT_given_behaviour


def calculate_logp_LoT_given_behaviour_dynamic(lengths, LoTs, categories, 
                                               n_participants, temp, index_true_LoT):
    """
    This is the version of the experiment where for each participant 
    at each trial we calculate the expected information gain of each remaining 
    object that could be shown. 
    
    This is a version of greedy optimal Bayesian experimental design.
    
    Returns
    -------
    tuple
        Tuple (true_LoT, logp_LoT_given_behaviour_total)
        where true_LoT is the true LoT 
        and logp_LoT_given_behaviour_total is the posterior probability over LoT
        given the participant's behaviour in the dynamic experiment.
    """
    
    # the participant's prior probability of each category 
    # given each LoT, assuming a softmax simplicity prior
    # Dimensions: (LoT, category)
    logp_category_given_LoT = log_softmax(
        lengths.astype(np.int64), 
        axis=1, 
        temp=temp
    )
    
    # initialize distribution over LoTs given total 
    # participant behaviour
    logp_LoT_given_behaviour_total = (
        np.zeros(len(LoTs)) - np.log(len(LoTs))
    )
    
    for participant_index in range(n_participants):

        # initialize experimenter's distribution 
        # over LoTs for that specific participant
        # Dimensions: (LoT)
        logp_LoT_given_behaviour = (
            np.zeros(len(LoTs)) - np.log(len(LoTs))
        )
        
        # Observes is 1 at the indices that the
        # participant has observed.
        # Start without having observed anything
        observed = np.zeros(16)
        
        # choose random category to teach the participant
        category = categories[
            np.random.randint(0,len(categories))
        ]

        # loop over trials
        for trial in range(16):
            
            # boolean mask of categories compatible with the 
            # data observed so far. This is a Boolean array.
            # Dimensions: (category)
            compatible_with_data = np.all(
                # either not observed or same as category
                (category==categories) | np.logical_not(observed),
                axis=1
            )
            
            # print("Compatible with data: ", compatible_with_data)
            
            # Calculate the likelihood from the participant's point of view
            # of all the data seen thus far
            # for each possible participant's LoT.
            # Since we are doing weak sampling, to calculate
            # the probability of each category given each LoT given the observed
            # (given the participant has each of the possible LoTs),
            # we just normalize the prior probabilities of the categories 
            # compatible with the observed data (which depend on the LoT).
            
            # Consider from the participant's prior
            # only the categories compatible with the data
            # Dimensions (LoT, category compatible with data)
            logp_cat_given_LoT_unnorm = (
                logp_category_given_LoT
                [:,compatible_with_data]
            )
            
            # normalize so we have a distribution 
            # over categories for each LoT
            # Dimensions (LoT, category compatible with data)
            logp_cat_given_LoT = (
                logp_cat_given_LoT_unnorm
                - logsumexp(
                    logp_cat_given_LoT_unnorm,
                    1,
                    use_ne=False
                )
            )
            
            # print("Calculated array")
            
            # Do hypothesis averaging with logp_cat_given_LoT, 
            # which gives the marginal probability of accepting 
            # each object given each LoT.
            # This is the logprobability of answering 'yes' to each object
            # I.e., saying that object belongs to the category,
            # given each LoT, given current knowledge about LoTs. 
            # Dimensions (LoT, object)
            
            # Loop over categories, and for each LoT / object combo
            # add the probability of that category
            # if the category contains the object and nothing otherwise
            
            # Accumulator. Dimensions (LoT, object)
            logp_yes_given_LoT = np.full(
                (logp_cat_given_LoT.shape[0], categories.shape[1]), 
                -np.inf
            )
            # loop over categories compatible with the data
            # subarr has dims: (LoT)
            # And contains the probability of that category 
            # in each LoT (given the data so far).
            # Mask has dims: (object)
            # and contains for each object whether it is compatible with
            # the current category
            for subarr, mask in zip(logp_cat_given_LoT.T, categories[compatible_with_data]):
                
                # Dimensions (LoT, object)
                # Add for each LoT the probability of the category
                # in that LoT if the object belongs to the category,
                # and -inf otherwise
                logp_yes_given_LoT = np.logaddexp(
                    logp_yes_given_LoT,
                    # shape (LoT, object)
                    np.where(
                        # mask of categories compatible with data
                        # shape (category compatible with data, object)
                        mask,
                        # logprobability of LoT. Shape (LoT)
                        subarr[:,None],
                        # 0
                        -np.inf
                    )
                )   
            
            # print("Calculated logp_yes_given_LoT")
            
            # logp of each LoT given the hypothetical answer "yes"
            # in response to seeing each object.
            # Needed to calculate expected info gain
            # From the experimenter's point of view
            # Dimensions (LoT, object)
            logp_LoT_given_yes_unnorm = (
                # likelihood
                logp_yes_given_LoT 
                # current prior
                + logp_LoT_given_behaviour[:,None]
            )
            # normalize so there is a prob vector 
            # for each LoT 
            # Dimensions: (LoT, object)
            logp_LoT_given_yes = (
                logp_LoT_given_yes_unnorm 
                - logsumexp(
                    logp_LoT_given_yes_unnorm, 
                    0,
                    use_ne=False
                )
            )
            
            # print("Calculated logp_LoT_given_yes")
            
            # logprobability of answering no to each object
            # for each LoT, given current knowledge about LoTs. 
            # Dimensions (LoT, object)
            logp_no_given_LoT = np.log(
                -np.expm1(logp_yes_given_LoT)
            )
            
            # logp of each LoT given hypothetical answer "no"
            logp_LoT_given_no_unnorm = (
                # likelihood
                logp_no_given_LoT 
                # current prior
                + logp_LoT_given_behaviour[:,None]
            )
            # normalize to get distribution over LoTs
            # Given answer "no" for each stimulus
            # Dimensions: (LoT, object)
            logp_LoT_given_no = (
                logp_LoT_given_no_unnorm 
                - logsumexp(
                    logp_LoT_given_no_unnorm, 
                    0,
                    use_ne=False
                )
            )
            
            # for each object, information gain 
            # if participant answers yes
            # Dimension (object)
            infogain_yes = (
                np.exp(logp_LoT_given_yes) *
                (logp_LoT_given_yes - logp_LoT_given_behaviour[:,None])
            ).sum(0)
            
            # information gain if participant answers no
            # for each object. Dimension (object)
            infogain_no = (
                np.exp(logp_LoT_given_no) *
                (logp_LoT_given_no - logp_LoT_given_behaviour[:,None])
            ).sum(0)
            
            infogain = np.stack((infogain_no, infogain_yes))
            
            # Expected (over the two possible answers) information gain
            # of showing each object to the participant.
            # dimensions: (object)
            # QUESTION: Do I need to weight by overall probability
            # of answering yes / no?
            expected_info_gain = np.mean(
                infogain,
                axis=0
            )
            
            # print("Expected info gains: ", expected_info_gain)
            
            # for objects that were already asked about, infogain
            # is going to be nan. 
            # Set it to 0 since no info is gained by asking about that object
            expected_info_gain = np.nan_to_num(
                expected_info_gain, 
                nan=0.0
            )
            
            # index of object out of the still unobserved ones
            # that the participant is asked to categorize
            # asked_ps = expected_info_gain / expected_info_gain.sum()
            # asked_object = np.random.choice(
            #     len(expected_info_gain),
            #     p=asked_ps
            # )
            asked_object = np.argmax(expected_info_gain)
            
            # print("Asked object: ", asked_object)
            
            # Record the asked-about object in observed array
            observed[asked_object] = 1
            
            # generate simulated behaviour from the participant
            # by sampling an answer ('yes' or 'no') for that trial
            p_yes = np.exp(
                logp_yes_given_LoT[index_true_LoT,asked_object]
            )
            # print("Probability of yes: ", p_yes)
            
            agent_behaviour = np.random.choice(
                [0,1],
                p=[1-p_yes, p_yes]
            )
            
            # print("Agent behaviour: ", agent_behaviour)

            # logp of the answer given each possible LoT
            # Dimensions: (LoT)
            logp_behaviour_given_LoT = (
                logp_no_given_LoT[:,asked_object] 
                if (agent_behaviour == 0)
                else logp_yes_given_LoT[:,asked_object]
            )

            # accumulate likelihood of that trial in unnormalized posterior
            logp_LoT_given_behaviour += logp_behaviour_given_LoT

        # normalize to find probability of each LoT given the agent's behaviour
        # i.e. which stimuli were accepted and rejected
        # Dimensions: (LoT)
        logp_LoT_given_behaviour -= logsumexp(
            logp_LoT_given_behaviour, 
            axis=0,
            use_ne=False
        )
        # Dimensions: (LoT)
        logp_LoT_given_behaviour_total += logp_LoT_given_behaviour
        
        print("Done with participant ", participant_index)
    
    logp_LoT_given_behaviour_total -= logsumexp(
        logp_LoT_given_behaviour_total, 
        axis=0,
        use_ne=False
    )
    
    print("\n\nTrue LoT: ", index_true_LoT)
    print("P true LoT: ", np.exp(logp_LoT_given_behaviour_total[index_true_LoT]))
    argmax = np.argmax(logp_LoT_given_behaviour_total)
    print("P max, LoT max: ", argmax, np.exp(logp_LoT_given_behaviour_total[argmax]), "\n\n") 

    return LoTs[index_true_LoT], logp_LoT_given_behaviour_total



def calculate_logp_LoT_given_behaviour_serial(lengths, LoTs, categories, 
                                               n_participants, temp, index_true_LoT):
    """
    This is the version of the experiment where participants are given
    feedback trial-by-trial, but the order in which the stimuli are shown 
    is randomized.
    
    Returns
    -------
    tuple
        Tuple (true_LoT, logp_LoT_given_behaviour_total)
        where true_LoT is the true LoT 
        and logp_LoT_given_behaviour_total is the posterior probability over LoT
        given the participant's behaviour in the serial experiment.
    """
    
    # the participant's prior probability of each category 
    # given each LoT, assuming a softmax simplicity prior
    # Dimensions: (LoT, category)
    logp_category_given_LoT = log_softmax(
        lengths.astype(np.int64), 
        axis=1, 
        temp=temp
    )
    
    # initialize distribution over LoTs given total 
    # participant behaviour
    logp_LoT_given_behaviour_total = (
        np.zeros(len(LoTs)) - np.log(len(LoTs))
    )
    
    for participant_index in range(n_participants):

        # initialize experimenter's distribution 
        # over LoTs for that specific participant
        # Dimensions: (LoT)
        logp_LoT_given_behaviour = (
            np.zeros(len(LoTs)) - np.log(len(LoTs))
        )
        
        # Observes is 1 at the indices that the
        # participant has observed.
        # Start without having observed anything
        observed = np.zeros(16)
        
        # shuffle the order of the objects to show
        order_shown = np.arange(16)
        np.random.shuffle(order_shown)
        
        # choose random category to teach the participant
        category = categories[
            np.random.randint(0,len(categories))
        ]

        # loop over trials
        for trial, asked_object in enumerate(order_shown):
            
            # boolean mask of categories compatible with the 
            # data observed so far. This is a Boolean array.
            # Dimensions: (category)
            compatible_with_data = np.all(
                # either not observed or same as category
                (category==categories) | np.logical_not(observed),
                axis=1
            )
            
            # print("Compatible with data: ", compatible_with_data)
            
            # Calculate the likelihood from the participant's point of view
            # of all the data seen thus far
            # for each possible participant's LoT.
            # Since we are doing weak sampling, to calculate
            # the probability of each category given each LoT given the observed
            # (given the participant has each of the possible LoTs),
            # we just normalize the prior probabilities of the categories 
            # compatible with the observed data (which depend on the LoT).
            
            # Consider from the participant's prior
            # only the categories compatible with the data
            # Dimensions (LoT, category compatible with data)
            logp_cat_given_LoT_unnorm = (
                logp_category_given_LoT
                [:,compatible_with_data]
            )
            
            # normalize so we have a distribution 
            # over categories for each LoT
            # Dimensions (LoT, category compatible with data)
            logp_cat_given_LoT = (
                logp_cat_given_LoT_unnorm
                - logsumexp(
                    logp_cat_given_LoT_unnorm,
                    1,
                    use_ne=False
                )
            )
            
            # print("Calculated array")
            
            # Do hypothesis averaging with logp_cat_given_LoT, 
            # which gives the marginal probability of accepting 
            # each object given each LoT.
            # This is the logprobability of answering 'yes' to each object
            # I.e., saying that object belongs to the category,
            # given each LoT, given current knowledge about LoTs. 
            # Dimensions (LoT, object)
            
            # Loop over categories, and for each LoT / object combo
            # add the probability of that category
            # if the category contains the object and nothing otherwise
            
            # Accumulator. Dimensions (LoT, object)
            logp_yes_given_LoT = np.full(
                (logp_cat_given_LoT.shape[0], categories.shape[1]), 
                -np.inf
            )
            # loop over categories compatible with the data
            # subarr has dims: (LoT)
            # And contains the probability of that category 
            # in each LoT (given the data so far).
            # Mask has dims: (object)
            # and contains for each object whether it is compatible with
            # the current category
            for subarr, mask in zip(logp_cat_given_LoT.T, categories[compatible_with_data]):
                
                # Dimensions (LoT, object)
                # Add for each LoT the probability of the category
                # in that LoT if the object belongs to the category,
                # and -inf otherwise
                logp_yes_given_LoT = np.logaddexp(
                    logp_yes_given_LoT,
                    # shape (LoT, object)
                    np.where(
                        # mask of categories compatible with data
                        # shape (category compatible with data, object)
                        mask,
                        # logprobability of LoT. Shape (LoT)
                        subarr[:,None],
                        # 0
                        -np.inf
                    )
                )   
            
            # logprobability of answering no to each object
            # for each LoT, given current knowledge about LoTs. 
            # Dimensions (LoT, object)
            logp_no_given_LoT = np.log(
                -np.expm1(logp_yes_given_LoT)
            )
                        
            # print("Asked object: ", asked_object)
            
            # Record the asked-about object in observed array
            observed[asked_object] = 1
            
            # generate simulated behaviour from the participant
            # by sampling an answer ('yes' or 'no') for that trial
            p_yes = np.exp(
                logp_yes_given_LoT[index_true_LoT,asked_object]
            )
            # print("Probability of yes: ", p_yes)
            
            agent_behaviour = np.random.choice(
                [0,1],
                p=[1-p_yes, p_yes]
            )
            
            # print("Agent behaviour: ", agent_behaviour)

            # logp of the answer given each possible LoT
            # Dimensions: (LoT)
            logp_behaviour_given_LoT = (
                logp_no_given_LoT[:,asked_object] 
                if (agent_behaviour == 0)
                else logp_yes_given_LoT[:,asked_object]
            )

            # accumulate likelihood of that trial in unnormalized posterior
            logp_LoT_given_behaviour += logp_behaviour_given_LoT

        # normalize to find probability of each LoT given the agent's behaviour
        # i.e. which stimuli were accepted and rejected
        # Dimensions: (LoT)
        logp_LoT_given_behaviour -= logsumexp(
            logp_LoT_given_behaviour, 
            axis=0,
            use_ne=False
        )
        # Dimensions: (LoT)
        logp_LoT_given_behaviour_total += logp_LoT_given_behaviour
        
        print("Done with participant ", participant_index)
    
    logp_LoT_given_behaviour_total -= logsumexp(
        logp_LoT_given_behaviour_total, 
        axis=0,
        use_ne=False
    )
    
    print("\n\nTrue LoT: ", index_true_LoT)
    print("P true LoT: ", np.exp(logp_LoT_given_behaviour_total[index_true_LoT]))
    argmax = np.argmax(logp_LoT_given_behaviour_total)
    print("P max, LoT max: ", argmax, np.exp(logp_LoT_given_behaviour_total[argmax]), "\n\n") 

    return index_true_LoT, logp_LoT_given_behaviour_total


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
