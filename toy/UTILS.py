from scipy.stats import maxwell
from scipy.optimize import brentq
import numpy as np

####### Module variables
dt = 1e-4 #seconds
######


def Gradient_Computation(f, x, h=1e-4):
    """
    Return grad_f(x)
    grad_f(x) from central finite differencies
    """
    dx = h
    g = (f(x + dx) - f(x - dx)) / (2*h)
    return g

def Verlet_Update(Z, V, Time, Bfield, CONSTANTS):
    """
    Return Velocity V(t + dt), x(t + dt)
    Implementation of the Verlet Algorithm
    """
    Potential = lambda z: (.5 * CONSTANTS.bohr_magneton) * Bfield(z) # define the potential
    
    ACC = - Gradient_Computation(Potential,Z)  / CONSTANTS.mass # Compute the acceleration at the current step
    
    Z_nextstep = X + V*dt + .5*ACC*dt**2 # 1. compute the next step
    
    ACC_nextstep = - Gradient_Computation(Potential, Z_nextstep) / CONSTANTS.mass # 2. compute the acceleration at the next step
    
    V_nextstep = V + 0.5*(ACC + ACC_nextstep)*dt # 3. compute the velocity at the next step

    return V_nextstep, Z_nextstep

def InitialCondition(Bfield, CONSTANTS):
    """
    Return v(t = 0), z(t = 0)
    Compute the initial condition of a particle
    from a maxwell distribution and using the potential of the well
    this is not a formally correct calculation, for which you
    would need microcanonical ensemble formalism
    """
    Potential = lambda z: (.5 * CONSTANTS.bohr_magneton) * Bfield(z) # define the potential
    
    # extracting the energy from the maxwell distribution
    scale = np.sqrt(CONSTANTS.kB * CONSTANTS.Temperature/CONSTANTS.mass)

    Ek = 0
    while(True):
        v = maxwell.rvs(scale = scale, size = 1)
        Ek = .5 * CONSTANTS.mass *v*v # Compute kinetic energy
        if(Ek < (Potential(CONSTANTS.Zmax) - Potential(CONSTANTS.Zmid))): # check particle is trapped
            break 
    
    # Now find the corresponding Z at which the U(z) =  Ek
    #print(Ek, Potential(CONSTANTS.Zmax) - Potential(CONSTANTS.Zmid) )
    
    Zsol1 = brentq(lambda z: Potential(z) - Potential(CONSTANTS.Zmid) - Ek, 
                  CONSTANTS.Zmid,
                  CONSTANTS.Zmax)
    Zsol2 = brentq(lambda z: Potential(z) - Potential(CONSTANTS.Zmid) - Ek, 
                  CONSTANTS.Zmin, 
                  CONSTANTS.Zmid)

    Zsample = np.random.uniform(Zsol1, Zsol2)
    
    # compute new kinetic energy
    Ek = Ek - (Potential(Zsample) - Potential(CONSTANTS.Zmid))

    if(Ek < 0):
        print(f"Error!!!, Ek {Ek} < 0, fixing it")
        Ek = 0

    # compute velocity
    v =  np.sqrt(2* Ek / CONSTANTS.mass)
    
    return v, Zsample