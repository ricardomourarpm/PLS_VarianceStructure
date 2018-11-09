from scipy.stats import chi2

def chiprod(dimension,degrees):
    product = 1
    for i in range(1,dimension+1):
        product = product*chi2.rvs(degrees-i+1)
    return product