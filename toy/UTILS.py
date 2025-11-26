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
Temperature_transv = 5e-3 # K
harmonic_degree = 2 # degree of the harmonic potential
harmonic_coefficient = 0.5 # coefficient harmonic potential
Zmin = -0.605 # m
Zmax = -0.481 # m
Zmid = -0.544 # m
zacceptance_min = -0.775
zacceptance_max = -0.325
mu_eff_over_m =  bohr_magneton / mass
######


def Gradient_Computation_1d(f, z, h=0.0005):
    """
    Return grad_f(x)
    grad_f(x) from central finite differencies
    """
    dz = h
    df_dz = (f(z + dz) - f(z - dz)) / (2*h)
    return df_dz

def Gradient_Computation_3d(f, x, y, z,  h=0.0005):
    """
    Return grad_f(x)
    grad_f(x) from central finite differencies
    """
    dx = h
    dy = h
    dz = h
    df_dx = (f(x + dx) - f(x - dx)) / (2*h)
    df_dy = (f(y + dy) - f(y - dy)) / (2*h)
    df_dz = (f(z + dz) - f(z - dz)) / (2*h)
    return [df_dx, df_dy, df_dz]

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

def BfieldExtended(function, Z, r, CONSTANTS):
    "3d extent of the magnetic field model, assuming harmonic potential"
    "Z [m]"
    "r radius [m]"
    return function(Z) * (1 + CONSTANTS.harmonic_coefficient*r**CONSTANTS.harmonic_degree)

def InitialCondition_3d(Bfield, CONSTANTS):
    """
    Return v(t = 0), z(t = 0)
    Compute the initial condition of a particle
    from a maxwell distribution and using the potential of the well
    this is not a formal calculation, for which you
    would need microcanonical ensemble formalism
    """
    Potential = lambda z: ( .5 * CONSTANTS.bohr_magneton) * Bfield(z) # define the potential

    #---------------------------------------------------------
    # AXIAL ENERGY
    #---------------------------------------------------------
    
    # extracting the energy from the maxwell distribution
    scale = np.sqrt(CONSTANTS.kB * CONSTANTS.Temperature/CONSTANTS.mass)
    
    Ek_axial = 0
    while(True):
        v = maxwell.rvs(scale = scale, size = 1).item()
        Ek_axial = .5 * CONSTANTS.mass *v*v # Compute kinetic energy
        if(Ek_axial < (Potential(CONSTANTS.Zmax) - Potential(CONSTANTS.Zmid))): # check particle is trapped
            break 
    
    # Now find the corresponding Z at which the U(z) =  Ek_axial
    #print(Ek_axial, Potential(CONSTANTS.Zmax) - Potential(CONSTANTS.Zmid) )
    
    Zsol1 = brentq(lambda z: Potential(z) - Potential(CONSTANTS.Zmid) - Ek_axial, 
                  CONSTANTS.Zmid,
                  CONSTANTS.Zmax)
    Zsol2 = brentq(lambda z: Potential(z) - Potential(CONSTANTS.Zmid) - Ek_axial, 
                  CONSTANTS.Zmin, 
                  CONSTANTS.Zmid)

    Zsample = np.random.uniform(Zsol1, Zsol2)
    
    # compute new kinetic energy
    Ek_axial = Ek_axial - (Potential(Zsample) - Potential(CONSTANTS.Zmid))

    if(Ek_axial < 0):
        print(f"Error!!!, Ek_axial {Ek_axial} < 0, fixing it")
        Ek_axial = 0

    # compute velocity
    v_axial =  np.sqrt(2* Ek_axial / CONSTANTS.mass)
    v_axial = v_axial * (np.random.choice([-1, 1]))

    #---------------------------------------------------------
    # TRANSVERSE ENERGY
    #---------------------------------------------------------
    
    # extracting the energy from the maxwell distribution
    scale = np.sqrt(CONSTANTS.kB * CONSTANTS.Temperature_transv/CONSTANTS.mass)
    
    v_transv = maxwell.rvs(scale = scale, size = 1).item()
    Ek_transv = .5 * CONSTANTS.mass *v_transv*v_transv # Compute kinetic energy 

    theta = np.random.uniform(0, 2*np.pi)
    vx, vy = v_transv * np.cos(theta), v_transv * np.sin(theta)
    
    return np.array([vx, vy, v_axial], dtype=float), np.array([0, 0, Zsample], dtype=float)


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
def dB_seg(z, x, y,
           Tloc, ramplength, 
           z0, dz, 
           dBinit,
           Binit,
           dBfinal,
           Bfinal):
    # compute the 3d gradient
    # Tloc = local time during the ramp
    # z0, dz needed to find the closest point on the grid
    
    alpha = Tloc / ramplength # time parameter
    N = dBinit.shape[0]
    i = idx_nearest(z, z0, dz, N) # find the closest point on the grid
    dinit  = dBinit[i]
    dfinal = dBfinal[i]

    r = np.sqrt((x*x + y*y)) # compute radius
    coeff_z = (1 + harmonic_coefficient * r**harmonic_degree)
    coeff_x = harmonic_degree*harmonic_coefficient* x**(harmonic_degree - 1)
    coeff_y = harmonic_degree*harmonic_coefficient* y**(harmonic_degree - 1)
    
    # Compute Bfield gradient at alpha = 0, ramp start 
    dBi_dx = Binit[i]  * coeff_x
    dBi_dy = Binit[i]  * coeff_y
    dBi_dz = dBinit[i] * coeff_z
    # Compute the Bfield gradient at alpha = Tramp, 
    dBf_dx = Bfinal[i]  * coeff_x
    dBf_dy = Bfinal[i]  * coeff_y
    dBf_dz = dBfinal[i] * coeff_z

    # Create the ramp
    dBt_dx = dBi_dx + (dBf_dx - dBi_dx) * alpha
    dBt_dy = dBi_dy + (dBf_dy - dBi_dy) * alpha
    dBt_dz = dBi_dz + (dBf_dz - dBi_dz) * alpha
    
    return dBt_dx, dBt_dy, dBt_dz

@njit
def dB_plateau(z, x, y, z0, dz, dBgrid, Bgrid):
    N = dBgrid.shape[0]
    i = idx_nearest(z, z0, dz, N)

    r = np.sqrt((x*x + y*y)) # compute radius
    coeff_z = (1 + harmonic_coefficient * r**harmonic_degree)
    coeff_x = harmonic_degree*harmonic_coefficient* x**(harmonic_degree - 1)
    coeff_y = harmonic_degree*harmonic_coefficient* y**(harmonic_degree - 1)
    
    # Compute Bfield gradient at alpha = 0, ramp start 
    dB_dx = Bgrid[i]  * coeff_x
    dB_dy = Bgrid[i]  * coeff_y
    dB_dz = dBgrid[i]   * coeff_z
    
    return dB_dx, dB_dy, dB_dz

@njit
def step_all_particles(R,V,
                       Time,seg_id, 
                       Tloc, ramp_len,
                       z0, dz,          # da zgrid[0], zgrid[1]-zgrid[0]
                       dBinit,
                       Binit,
                       dBfinal,
                       Bfinal,
                       Annih, T_ann,
                       dB_buffer):
    # Z, X, Y: 3d array for each particle
    
    N = R.shape[0] # Get Number of Particles
    for i in range(N): # Lop on particles
        if Annih[i]:
            continue

        # get coordinates
        x = R[i, 0]
        y = R[i, 1]
        z = R[i, 2]

        if(seg_id == 0): # ramp
            dBx, dBy, dBz = dB_seg(z, x, y,
                        Tloc, ramp_len, 
                        z0, dz, 
                        dBinit,
                        Binit,
                        dBfinal,
                        Bfinal)
        elif(seg_id == 1): # wait
            dBx, dBy, dBz = dB_plateau(z, x, y, 
                            z0, dz, 
                            dBfinal, 
                            Bfinal)
        dB_buffer[0] = dBx
        dB_buffer[1] = dBy
        dB_buffer[2] = dBz
        a  = -mu_eff_over_m * dB_buffer

        # --- Verlet posizione ---
        R_new = R[i] + V[i]*dt + 0.5*a*dt*dt
        # get coordinates
        x_new = R_new[0]
        y_new = R_new[1]
        z_new = R_new[2]
        
        # --- dB/dz al passo successivo ---
        if seg_id == 0: # ramp
            dBx, dBy, dBz = dB_seg(z_new, x_new, y_new,
                             Tloc+dt, ramp_len,
                             z0, dz,
                             dBinit, 
                             Binit,
                             dBfinal,
                             Bfinal)
        elif seg_id == 1: # wait
            dBx, dBy, dBz = dB_plateau(z_new, x_new, y_new, 
                                 z0, dz, 
                                 dBfinal,
                                 Bfinal)
        dB_buffer[0] = dBx
        dB_buffer[1] = dBy
        dB_buffer[2] = dBz
        a_next = -mu_eff_over_m * dB_buffer

        # --- Update Velocity and Position ---
        V[i] = V[i] + 0.5*(a + a_next)*dt
        R[i] = R_new

        # --- annichilazione se esce ---
        if (z_new < zacceptance_min) or (z_new > zacceptance_max):
            Annih[i] = True
            T_ann[i] = Time + dt