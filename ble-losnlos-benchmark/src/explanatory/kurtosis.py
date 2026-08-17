import scipy.stats as stats 

def reduce_kurtosis(C):
    if len(C) < 4:
        return np.nan
    return float(stats.kurtosis(C, fisher=True, bias=False))