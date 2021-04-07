import pymc3 as pm
import theano 


def get_data():

    # L has shape (LoT, cat)
    # where the LoT index encodes the LoT
    # in the way described by the 
    # encoding file
    L = pd.read_pickle('./L.pkl')

    # note that learning costs
    # are only calculated for half of the categories
    learning_costs_df = pd.read_pickle('../learning_costs.pkl')
    category_i = learning_costs_df['cat']
    outcome_i = learning_costs_df['outcome']

    return {
        'L': L,
        'category_i': category_i,
        'outcome_i': outcome_i
    }


def define_model(L, category_i, outcome_i):
    coords = {
        'LoT': np.arange(len(L)),
        'cat': np.arange(L.shape[1]),
        'obs': np.arange(len(category_i))
    }

    with pm.Model(coords=coords) as model:

    # #     SET DATA
    #     L_data = pm.Data('L', L)#, dims=('LoT', 'cat'))
    #     cat_i_data = pm.Data('cat_i', category_i)#, dims='obs')
    #     out_i_data = pm.Data('outcome_i', outcome_i)#, dims='obs')

        L_data, cat_i_data, out_i_data = L, category_i, outcome_i

        L_dims = L.shape
        n_lots, n_cats = L.shape

        # SAMPLE PRIOR
        # sample a different sigma for each category
        sigma = pm.HalfNormal('sigma', sigma=2, dims='cat')
        a_0 = pm.Normal('a_0', mu=0, sigma=2)
        a_1 = pm.Normal('a_1', mu=0, sigma=2)

        # BUILD LIKELIHOOD
        # z_logp will contain the total loglik
        # given each LoT.
        z_logp = tt.zeros((n_lots,), dtype='float')
        # loop over LoT index
        for z_i in range(n_lots):
            # zi_logp is the logprob of all observations
            # across all categories for LoT with index z_i
            zi_logp = 0.
            # loop over categories
            for cat in range(n_cats):
                # get indices of observations with category cat
                obs_idx = np.where(cat_i_data==cat)[0]
                # calculate mean of costs 
                # given LoT z_i and category cat
                # (muZ is a scalar)
                muZ = a_0 + a_1 * L[z_i, cat]
                # increment logp by the joint prob
                # of all observations for category cat
                # given LoT with index z_i
                zi_logp = zi_logp + tt.sum(
                    pm.Normal
                    .dist(mu=muZ, sd=sigma[cat])
                    .logp(out_i_data[obs_idx])
                )
            z_logp = tt.set_subtensor(z_logp[z_i], zi_logp)
        # vector with the loglik of the whole data
        # given each LoT
        lp3 = pm.Deterministic('z_logp', z_logp, dims='LoT')
    #     # total logp is log(pr[X|z_1] + pr[X|z_2] + ...)
        tot_logp = pm.math.logsumexp(z_logp)
        pot = pm.Potential('e', tot_logp)
        
    return model


def sample_from_model(model):
    with model:
        trace = pm.sample(
            1000, 
            cores=1, 
    #         init='advi+adapt_diag',
            return_inferencedata=True,
            target_accept=0.95
        )
    return trace


def lognormalize(x):
    a = np.logaddexp.reduce(x)
    return x - a


def logmean(x, axis=0):
    # mean in log space
    # x is a vector of logprobs
    return np.logaddexp.reduce(x, axis) - np.log(len(x))


def posterior_over_LoTs(trace):
    posterior = trace.posterior
    # each sample contains the total loglik
    # of the data given all other nuisance variables for that sample.
    LoT_posterior = np.exp(lognormalize(logmean(posterior.z_logp.values[0], axis=0)))
    return LoT_posterior


def fit_variational(model):
    with model:
        fit = pm.fit()


if __name__=='__main__':
    results = get_data()
    model = define_model(**results)
    trace = sample_from_model(model)
    lot_posterior = posterior_over_LoTs(trace)

        
