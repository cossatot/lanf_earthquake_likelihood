import sys
sys.path.append('../eq_stats')

import numpy as np
import pandas as pd
import eq_stats as eqs

# problem setup

# read in fault data table
f = pd.read_csv('../data/lanf_stats.csv', index_col=0) 

# define some constants and parameters
n_eq_samp = 5e4 # number of earthquakes in time series
time_window = np.hstack( (1, np.arange(5, 105, step=5) ) ) # observation times
mc_iters = 1e3
mc_index = np.arange(mc_iters, dtype='int')
mc_cols = ['dip', 'Ddot'] + [t for t in time_window]
mag_min = 5

# define frequency-magnitude distribution
M_vec = np.linspace(5, 7.64, num=1000)
FM_vec = eqs.F(M_vec)


# First run on South Lunggar Detachment
slr = f.loc['s_lunggar'] # select S. Lunggar fault data from data table

#DF for SLR MC, >= M5 eqs
slr_5 = pd.DataFrame(index=mc_index, columns=mc_cols, dtype='float') 

slr_5.dip, slr_dip_frac = eqs.dip_rand_samp( slr['dip_deg'], 
                                             slr['dip_err_deg'], mc_iters)

slr_5.Ddot = eqs.Ddot_rand_samp( slr['slip_rate_mm_a'], 
                                      slr['sr_err_mm_a'], mc_iters)

# Loop through
for ii in mc_index:
    ss = slr_5.iloc[ii]
    M_samp = eqs.sample_from_pdf(M_vec, FM_vec, n_eq_samp)
    Mo_samp = eqs.calc_Mo_from_M(M_samp)

    recur_int = eqs.calc_recurrence_interval(Mo=Mo_samp, dip=ss['dip'],
                                             slip_rate=ss['Ddot'],
                                             L=slr['L_km'], z=slr['z_km'])

    cum_yrs = eqs.calc_cumulative_yrs(recur_int)

    eq_series = eqs.make_eq_time_series(M_samp, cum_yrs)  # maybe store in dict
    
    for t in time_window:
        ss[t] = ( eqs.get_prob_above_val_in_window(eq_series, mag_min, t) 
                * slr_dip_frac )

    print ii

print 'done'


