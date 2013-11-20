import sys
sys.path.append('../eq_stats')

import numpy as np
import pandas as pd
import eq_stats as eqs
import time
from joblib import Parallel, delayed

# problem setup

# read in fault data table
f = pd.read_csv('../data/lanf_stats.csv', index_col=0) 

# define some constants and parameters
n_cores = -4 # number of cores for parallel processing
n_eq_samp = 3e4 # number of earthquakes in time series
time_window = np.hstack( (1, np.arange(5, 105, step=5) ) ) # observation times
mc_iters = 2e3 # number of Monte Carlo iterations
mc_index = np.arange(mc_iters, dtype='int')
mc_cols = ['dip', 'Ddot'] + [t for t in time_window]
max_eq_slip = 15 #m

# define frequency-magnitude distribution  not needed in this instance
Mc = 7.64
#M_vec = np.linspace(5, Mc, num=1000)
#FM_vec = eqs.F(M=M_vec, Mc=Mc)

# load fault data and make dfs for each minimum search magnitude
min_M_list = [5, 5.5, 6, 6.5, 7, 7.5]

df_ind_tuples = [[i, M] for i in mc_index for M in min_M_list]
df_multi_ind = pd.MultiIndex.from_tuples(df_ind_tuples, names=['mc_iter','M'])


# define function to calculate probabilities for each iteration
# function is defined here so it can access all variables
def calc_iter_probs(iter):
    df_iter = fdf.loc[iter].copy()
    df_iter['dip'] = mc_d['dip_samp'][iter]
    df_iter['Ddot'] = mc_d['Ddot_samp'][iter]

    # Generate EQ sample/sequence from F_char(M) dist.
    m_vec = np.linspace(5, mc_d['max_M'][iter], num=1000)
    fm_vec = eqs.F_char(m_vec, Mc=Mc, char_M=6.25, char_amplitude_scale=5.25)
    M_samp = eqs.sample_from_pdf(m_vec, fm_vec, n_eq_samp)
    Mo_samp = eqs.calc_Mo_from_M(M_samp)
    
    # Make time series of earthquakes, including no eq years
    recur_int = eqs.calc_recurrence_interval(Mo=Mo_samp, 
                                             dip=mc_d['dip_samp'][iter],
                                             slip_rate=mc_d['Ddot_samp'][iter],
                                             L=params['L_km'],
                                             z=params['z_km'])

    cum_yrs = eqs.calc_cumulative_yrs(recur_int)
    eq_series = eqs.make_eq_time_series(M_samp, cum_yrs)
    
    # calculate probability of observing EQ in time_window
    for t in time_window:
        roll_max = pd.rolling_max(eq_series, t)
        df_iter[t] = (eqs.get_probability_above_value(roll_max, min_M_list)
                      * mc_d['dip_frac'] )

    return df_iter

# run for south lunggar trial
for fault in list(f.index):
    fdf = pd.DataFrame(index=df_multi_ind, columns=mc_cols, dtype='float')
    params = f.loc[fault]
    mc_d = {}
    mc_d['dip_samp'], mc_d['dip_frac'] = eqs.dip_rand_samp( params['dip_deg'], 
                                                         params['dip_err_deg'], 
                                                         mc_iters)

    mc_d['Ddot_samp'] = eqs.Ddot_rand_samp(params['slip_rate_mm_a'],
                                           params['sr_err_mm_a'], mc_iters)

    mc_d['max_Mo'] = eqs.calc_Mo_from_fault_params(L=params['L_km'], 
                                                   z=params['z_km'], 
                                                   dip=mc_d['dip_samp'], 
                                                   D=max_eq_slip)

    mc_d['max_M'] = eqs.calc_M_from_Mo(mc_d['max_Mo'])

    t0 = time.time()
    prob_list = Parallel(n_jobs=n_cores)( delayed( calc_iter_probs)(ii) 
                                    for ii in mc_index)
    print 'done with', fault, 'parallel calcs in {} s'.format((time.time()-t0))
    for ii in mc_index:
        fdf.loc[ii][:] = prob_list[ii]
    fdf.to_csv('../results/{}_F_char.csv'.format(fault))    
