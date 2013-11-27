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
n_eq_samp = 2.5e4
time_window = np.hstack( (1, np.arange(5, 105, step=5) ) ) # observation times
mc_iters = 2e3 # number of Monte Carlo iterations
mc_index = np.arange(mc_iters, dtype='int')
mc_cols = ['dip', 'Ddot'] + [t for t in time_window]
max_eq_slip = 15 #m
char_eq_slip= 1.5 #m
Mc = 7.64 # Hypothetical corner magnitude for Continental Rift Boundaries

# load fault data and make dataframes for each minimum search magnitude
min_M_list = [5, 5.5, 6, 6.5, 7, 7.5]

df_ind_tuples = [[i, M] for i in mc_index for M in min_M_list]
df_multi_ind = pd.MultiIndex.from_tuples(df_ind_tuples, names=['mc_iter','M'])

rec_int_bins = np.logspace(1, 5)  # bins for recurrence interval statistics

# define function to calculate probabilities for each iteration
# function is defined here so it can access all variables
def calc_iter_probs(ii):
    df_iter = fdf.loc[ii].copy()
    df_iter['dip'] = mc_d['dip_samp'][ii]
    df_iter['Ddot'] = mc_d['Ddot_samp'][ii]

    # Generate EQ sample/sequence from F(M) dist.
    m_vec = np.linspace(5, mc_d['max_M'][ii], num=1000)
    fm_vec = eqs.F_char(m_vec, Mc=Mc, char_M=mc_d['char_M'][ii])
    M_samp = eqs.sample_from_pdf(m_vec, fm_vec, n_eq_samp)
    Mo_samp = eqs.calc_Mo_from_M(M_samp)
    
    # Make time series of earthquakes, including no eq years
    recur_int = eqs.calc_recurrence_interval(Mo=Mo_samp, 
                                             dip=mc_d['dip_samp'][ii],
                                             slip_rate=mc_d['Ddot_samp'][ii],
                                             L=params['L_km'],
                                             z=params['z_km'])

    cum_yrs = eqs.calc_cumulative_yrs(recur_int)
    eq_series = eqs.make_eq_time_series(M_samp, cum_yrs)
    
    # calculate probability of observing EQ in time_window
    for t in time_window:
        roll_max = pd.rolling_max(eq_series, t)
        df_iter[t] = (eqs.get_probability_above_value(roll_max, min_M_list)
                      * mc_d['dip_frac'])
        
    # calculate histgrams of recurrence intervals
    rec_int_counts_df = rec_int_df.loc[ii].copy()
    for mm in np.array(min_M_list):
        ints = np.diff( np.where(eq_series >= mm) )
        rec_int_counts_df.loc[mm] = np.histogram(ints, bins=rec_int_bins)[0]
        

    return df_iter, rec_int_counts_df

t_init = time.time()

# do calculations, in parallel
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
    
    mc_d['char_Mo'] = eqs.calc_Mo_from_fault_params(L=params['L_km'], 
                                                    z=params['z_km'], 
                                                    dip=mc_d['dip_samp'], 
                                                    D=char_eq_slip)
    
    mc_d['char_M'] = eqs.calc_M_from_Mo(mc_d['char_Mo'])
    
    rec_int_df = pd.DataFrame(columns = rec_int_bins[1:],
                              index=df_multi_ind, dtype='float')
    t0 = time.time()
    out_list = Parallel(n_jobs=n_cores)( delayed( calc_iter_probs)(ii) 
                                    for ii in mc_index)
    print 'done with', fault, 'parallel calcs in {} s'.format((time.time()-t0))
    for ii in mc_index:
        fdf.loc[ii][:] = out_list[ii][0]
        rec_int_df.loc[ii][:] = out_list[ii][1]
        
    fdf.to_csv('../results/{}_char.csv'.format(fault))
    rec_int_df.to_csv('../results/{}_char_rec_ints.csv'.format(fault))

print 'done with all faults in {} s'.format((time.time()-t_init))
