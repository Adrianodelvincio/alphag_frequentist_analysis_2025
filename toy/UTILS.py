from scipy.stats import maxwell
from scipy.optimize import brentq
import numpy as np
from numba import njit

####### Module variables
dt = .5e-4 #seconds
mass = 1.6735575e-27 # kg, mass Hbar
bohr_magneton = 9.2740100657e-24 # J/T, CODATA 2022
kB = 1.380649e-23   # J/K
Temperature = 5e-3 # K
Zmin = -0.605 # m
Zmax = -0.481 # m
Zmid = -0.544 # m
zacceptance_min = -0.775
zacceptance_max = -0.325
mu_eff_over_m = .5 * bohr_magneton / mass
######


def Gradient_Computation(f, x, h=0.0005):
    """
    Return grad_f(x)
    grad_f(x) from central finite differencies
    """
    dx = h
    g = (f(x + dx) - f(x - dx)) / (2*h)
    return g

def Verlet_Update(V, Z, Time, Bfield, h=0.0005):
    """
    Return Velocity V(t + dt), x(t + dt)
    Implementation of the Verlet Algorithm
    """

    dB_dz = (Bfield(Z + h) - Bfield(Z - h))/(2*h)
    
    ACC = - .5 * bohr_magneton * dB_dz  / mass # Compute the acceleration at the current step
    
    Z_nextstep = Z + V*dt + .5*ACC*dt**2 # 1. compute the next step

    dB_dznext = (Bfield(Z_nextstep + h) - Bfield(Z_nextstep - h))/(2*h)
    
    ACC_nextstep = - .5 * bohr_magneton * dB_dznext / mass # 2. compute the acceleration at the next step
    
    V_nextstep = V + 0.5*(ACC + ACC_nextstep)*dt # 3. compute the velocity at the next step
    
    #print(f"Z position {Z:.3f} speed {V*1e-3:.3f} acceleration {ACC:.4f} acceleration next step {ACC_nextstep:.4f}")
    #print(Z_nextstep, V_nextstep)
    
    return V_nextstep, Z_nextstep, (Time + dt)

def InitialCondition(Bfield, CONSTANTS):
    """
    Return v(t = 0), z(t = 0)
    Compute the initial condition of a particle
    from a maxwell distribution and using the potential of the well
    this is not a formally correct calculation, for which you
    would need microcanonical ensemble formalism
    """
    Potential = lambda z: ( .5 * CONSTANTS.bohr_magneton) * Bfield(z) # define the potential
    
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
    v = v * (np.random.choice([-1, 1]))
    
    return v, Zsample



#################################################
# njit to increase the performace
################################################

@njit
def idx_nearest(z, z0, dz, N):
    # z0 = zgrid[0], dz = zgrid[1]-zgrid[0], N = len(zgrid)
    idx = int((z - z0)/dz + 0.5)   # round to nearest
    if idx < 0:
        idx = 0
    elif idx >= N:
        idx = N - 1
    return idx

@njit
def dB_seg(z, Tloc, ramplength, z0, dz, dBinit, dBfinal):
    alpha = Tloc / ramplength
    N = dBinit.shape[0]
    i = idx_nearest(z, z0, dz, N)
    dinit = dBinit[i]
    dfinal = dBfinal[i]
    return dinit + (dfinal - dinit)*alpha

@njit
def dB_plateau(z, z0, dz, dBgrid):
    N = dBgrid.shape[0]
    i = idx_nearest(z, z0, dz, N)
    return dBgrid[i]

@njit
def step_all_particles(Z, V, Time,
                       seg_id, Tloc, ramp_len,
                       z0, dz,          # da zgrid[0], zgrid[1]-zgrid[0]
                       dBinit, dBfinal,
                       Annih, T_ann):
    
    N = Z.shape[0] # Get Number of Particles
    for i in range(N):
        if Annih[i]:
            continue

        z = Z[i]

        if(seg_id == 0): # ramp
            dB = dB_seg(z, Tloc, ramp_len, z0, dz, dBinit, dBfinal) # >>>
        elif(seg_id == 1):
            dB = dB_plateau(z, z0, dz, dBfinal) # >>>

        a  = -mu_eff_over_m * dB

        # --- Verlet posizione ---
        z_new = z + V[i]*dt + 0.5*a*dt*dt

        # --- dB/dz al passo successivo ---
        if seg_id == 0: # ramp
            dB_next = dB_seg(z_new, Tloc+dt, ramp_len, z0, dz, dBinit, dBfinal)
        elif seg_id == 1:
            dB_next = dB_plateau(z_new, z0, dz, dBfinal)

        a_next = -mu_eff_over_m * dB_next

        # --- Verlet velocità ---
        V[i] = V[i] + 0.5*(a + a_next)*dt
        Z[i] = z_new

        # --- annichilazione se esce ---
        if (z_new < zacceptance_min) or (z_new > zacceptance_max):
            Annih[i] = True
            T_ann[i] = Time + dt