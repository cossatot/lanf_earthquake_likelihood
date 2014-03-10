import numpy as np
import pandas as pd

mu = 30e9
C = 6

def F(M=None, Mmin=2.5, Mc=7.64, B=0.65):
    """
    F(M) is the tapered Gutenberg-Richter distribution
    given by Shen et al., 2007 SRL with values for constants
    from the CRB values of Bird and Kagan 2004 BSSA.
    """
    term1 = 10**(-1.5 * B * (M - Mmin) )
    term2 = np.exp(10**(1.5 * (Mmin - Mc) - 1.5 * (M-Mc) ) )
    
    return term1 * term2


def _lognormal(x, mu = -0.5, sigma=0.5):
    """
    Calculates a lognormal value given location param mu
    and scale param sigma, for each point in x
    """
    term1 = 1 / sigma * x * np.sqrt(2 * np.pi)
    term2 = np.exp(-(np.log(x) - mu)**2 / (2 * sigma**2) )
    
    return term1 * term2


def F_char(M, Mmin=2.5, Mc=7.64, B=0.65, char_M=6.25,
           char_amplitude_scale=5.25):
    """
    F(M) is the tapered Gutenberg-Richter distribution
    given by Shen et al., 2007 SRL with values for constants
    from the CRB values of Bird and Kagan 2004 BSSA.
    """
    Fm = F(M, Mmin=Mmin, Mc=Mc, B=B)
    F_char_amp = F(char_amplitude_scale)
    
    M_shift = M - char_M + 1
    M_shift[M_shift < 0] = 0
    
    F_lognormal = _lognormal( (M_shift) )
    norm_constant = np.max(F_lognormal) / F_char_amp
    
    F_lognormal *= 1/ norm_constant
    
    F_char = Fm + F_lognormal
    
    F_char += - F_char.min()
    
    return F_char


def get_ecdf(counts, norm=True):
    """Calculates an array sort of like an ECDF, but no x reference"""
    if norm == True:
        ecdf = np.cumsum(counts) / np.float(np.sum(counts))
    else:
        ecdf = np.cumsum(counts)
    
    return ecdf


def get_ecdf_dict(vals=None, counts=None, norm=True):
    """
    Returns a dict object linking counts to values.
    Things better be sorted!
    """
    ecdf = get_ecdf(counts, norm=norm)
    ecdf_dict = {}
    
    for i, val in enumerate(vals):
        ecdf_dict[val] = ecdf[i]
    
    return ecdf_dict


def get_inverse_ecdf_dict(vals=None, counts=None, norm=True):
    
    ecdf_dict = get_ecdf_dict(vals, counts, norm)
    inv_ecdf_dict = {y:x for x,y in ecdf_dict.iteritems()}
    
    return inv_ecdf_dict
    

def find_nearest_dict_val(dict_, vals):
    key_array = np.array(dict_.keys())
    nearest_vals = np.empty(vals.shape)
    
    for i, val in enumerate(vals):
        idx = (np.abs(key_array - val)).argmin()
        nearest_vals[i] = dict_[ key_array[idx] ]
    
    return nearest_vals


def sample_from_pdf(vals, counts, num):
    inv_ecdf_d = get_inverse_ecdf_dict(vals, counts)
    rand_samp = np.random.rand(num)
    pdf_samp = find_nearest_dict_val(inv_ecdf_d, rand_samp)
    
    return pdf_samp


def dip_rand_samp(dip, err, num, threshold=30, output='degrees'):
    """
    Creates a uniform random sample between min(dip+err, threshold)
    and returns both the sample and the fraction (decimal) of the total
    dip range that is below the threshold (this fraction should be multiplied
    by any final probabilities of observing an earthquake on a dip below the
    threshold value).

    Assumes input dips are in degrees (it's dip...) but output can be in
    radians, if desired.

    Returns: tuple [samp, fraction]
    """
    if threshold != None:

        dip_min = dip - err
        dip_max = min((dip + err), threshold)
    
        dip_samp = np.random.rand(num) * (dip_max - dip_min) + dip_min

        frac = (dip_max-dip_min) / float(2 * err)

    else:
        dip_min = dip - err
        dip_max = dip + err
        
        dip_samp = np.random.rand(num) * (dip_max - dip_min) + dip_min

        frac = 1

    if output == 'radians':
        dip_samp = np.radian(dip_samp)

    return [dip_samp, frac]


def Ddot_rand_samp(Ddot, err, num, rand_type = 'uniform'):
    """
    Creates a random sample of fault slip rates (Ddot, where 
    D := fault slip).  Can either be uniform between (Ddot-err, Ddot+err)
    or Gaussian, with mu=Ddot and std=err.

    specify rand_type = 'Gaussian'
    """
    if rand_type == 'uniform':
        Ddot_samp = np.random.random_sample(num) * (err * 2) + (Ddot - err)

    elif rand_type in ('Gaussian', 'gauusian', 'normal'):
        Ddot_samp = np.random.randn(num) * err + Ddot

    return Ddot_samp


def calc_Mo_from_M(M, C=C):
    """
    Calculate seismic moment (Mo) from
    moment magnitude (M) given a scaling law.

    C is a scaling constant; should be set at 6,
    but is defined elsewhere in the module so
    that all functions using it share a value.
    """
    term1 = 3/2. * C * (np.log(2) + np.log(5) )
    term2 = 3/2. * M * (np.log(2) + np.log(5) )
    
    Mo = np.exp( term1 + term2)
    
    return Mo


def calc_M_from_Mo(Mo, C=C):
    """
    Calculates moment magnitude (M) from seismic moment (Mo)
    given a scaling law.

    C is a scaling constant; should be set at 6,
    but is defined elsewhere in the module so
    that all functions using it share a value.

    """
    return (2/3.) * np.log10(Mo) - C


def calc_Mo_from_fault_params(L=None, z=None, dip=None, mu=mu, D=None,
                              area_dim='km', slip_dim='m', dip_dim='degrees'):
    """
    Calculates the seismic moment Mo (in N m) from fault dimensions,
    shear modulus mu, and mean slip distance.

    Is currently set up to convert from typical dimensions to meters and
    radians; units other than distances in km, slip (D) in m, and
    dips in degrees should be converted to meters and radians before passing
    to function.


    mu should be set at 3e9 (30 GPa) at the top of the module; but it may
    change.  It is defined elsewhere so that all functions use the same value.
    """
    if area_dim == 'km':
        L = L * 1000
        z = z * 1000

    if dip_dim == 'degrees':
        dip = np.radians(dip)

    return (L * z * mu * D) / np.sin(dip)


def calc_recurrence_interval(Mo=None, dip=None, mu=mu, L=None, z=None,
                            slip_rate=None, area_dim='km', 
                            slip_rate_dim='mm/yr', dip_dim='degrees'):
    """
    Calculates the recurrence interval for an earthquake of moment 'Mo'
    given dip, shear modulus 'mu', length 'L', seismogenic thickness 'z',
    and slip rate.

    Is currently set up to convert from typical dimensions to meters and
    radians; units other than distances in km, slip rates in mm/yr, and
    dips in degrees should be converted to meters and radians before passing
    to function.

    Returns recurrence interval in years.
    """

    if area_dim == 'km':
        L = L * 1000
        z = z * 1000

    if slip_rate_dim in ('mm/yr', 'mm/a'):
        slip_rate = slip_rate * 0.001

    if dip_dim == 'degrees':
        dip = np.radians(dip)
    
    return Mo * np.sin(dip) / (mu * L * z * slip_rate)


def calc_cumulative_yrs(recur_interval_sequence):
    """
    Calculates cumulative years from a sequence of recurrence intervals.
    Sequences are input as floats, and returned as integers, to be used
    as in index.

    Returns array of ints, with shape of input array.
    """
    return np.int_(np.cumsum(recur_interval_sequence.round() ) )


def make_eq_time_series(eq_seq=None, cum_years_seq=None ):
    """
    Takes a sequence of earthquake magnitudes (eq_seq) and their years
    in a time series (cum_years_seq) and returns a time series where
    the value of the series represents the magnitude of an earthquake
    in that year.  Each earthquake is separated from the earthquake before
    it by the number of years required to accumulate all the strain/moment
    released in that earthquake.

    Returns an array of length max(cum_years_seq)+1
    """
    eq_time_series = np.zeros( np.max(cum_years_seq) +1)
    
    eq_time_series[cum_years_seq] = eq_seq

    return eq_time_series


def get_probability_above_value(series, vals):
    vals = np.array(vals)

    count_above = np.zeros(vals.shape)
    for i, val in enumerate(vals):
        count_above[i] = len( series[series >= val])
    
    return count_above / float(len(series) )


def get_prob_above_val_in_window(series, value, window):
    """
    Takes a series of events, and calculates the probability that an 
    event >= a size 'value' will be observed in a contiguous time window of 
    size 'window'.

    Basically, this function takes the window and slides it along the time
    series, and counts how many times an event >= val is found in the window.
    
    Returns a decimal [0,1) expressing the number of successes to the total
    number.
    """
    rolling_max_series = pd.rolling_max(series, window)

    return get_probability_above_value(rolling_max_series, value)
