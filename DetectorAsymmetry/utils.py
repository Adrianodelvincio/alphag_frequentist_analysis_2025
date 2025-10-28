import numpy as np
from scipy.stats import norm

def function(B, x):
    # Linear function y = m*x + b
    # B is a vector of the parameters.
    # x is an array of the current x values.
    #
    D = B[0]
    K = B[1]
    ki = 1/(K*(1 - D**2))
    return ki*(1 + D*x)

def Araw(Atrue, D):
        "Convert from theoretical Asymmetry to real asymmetry, considering detector efficiency"
        return (D - Atrue)/(Atrue*D - 1)

def append_to_dict(dictionary, key, item):
    if key in dictionary.keys():
        dictionary[key].append(item)
    else:
        dictionary.update({key : [item]})

# === Funciton for Non Uniform Statistic ===
def NonUniformStat(BIAS, loc, scale, tot_event, item):
    # Calcolo distribuzione normale
    distribution = norm.pdf(list(BIAS.values()), loc=loc, scale=scale)
    distribution = distribution / distribution.sum()
    distribution = tot_event * item * distribution

    distribution = np.array(distribution)

    # Iterativamente forza il minimo a 100
    while True:
        mask = distribution < 100
        if not np.any(mask):
            break

        # Somma da redistribuire
        tot_diff = (100 - distribution[mask]).sum()
        distribution[mask] = 100

        # Aggiorna i rimanenti
        mask_high = distribution > 100
        if not np.any(mask_high):
            # Non ci sono elementi da cui sottrarre, quindi si esce
            break

        subtractable = distribution[mask_high]
        total_high = subtractable.sum()
        if total_high <= tot_diff:
            # Non c'è abbastanza da togliere, quindi portiamo tutti a 100 e usciamo
            distribution[mask_high] = np.maximum(100, distribution[mask_high] - (tot_diff / mask_high.sum()))
            break

        # Distribuisci la sottrazione proporzionalmente
        distribution[mask_high] -= tot_diff / mask_high.sum()

    distribution = np.rint(distribution)
    return distribution