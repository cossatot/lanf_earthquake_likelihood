import numpy as np

def F(M, Mmin=2.5, Mc=7.64, B=0.65):
    """
    F(M) is the tapered Gutenberg-Richter distribution
    given by Shen et al., 2007 SRL with values for constants
    from the CRB values of Bird and Kagan 2004 BSSA.
    """
    term1 = 10**(-1.5 * B * (M - Mmin) )
    term2 = np.exp(10**(1.5 * (Mmin - Mc) - 1.5 * (M-Mc) ) )
    
    return term1 * term2


def lognormal(x, mu = -0.5, sigma=0.5):
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
    
    F_lognormal = lognormal( (M_shift) )
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


def Mo_from_M(M, C=6):
    """
    Calculate seismic moment (Mo) from
    moment magnitude (M) given a scaling law
    """
    term1 = 3/2. * C * (np.log(2) + np.log(5) )
    term2 = 3/2. * M * (np.log(2) + np.log(5) )
    
    Mo = np.exp( term1 + term2)
    
    return Mo


def calc_rec_int(Mo=None, dip=None, mu=6e9, L=None, z=None,
                 slip_rate=None):
    
    return Mo * np.sin(dip) / (mu * L * z * slip_rate)


def prob_above_val(series, val):
    count_above = len( series[series >= val])
    
    return count_above / float(len(series) )



