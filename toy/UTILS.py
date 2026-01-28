from   scipy.stats    import maxwell
from   scipy.optimize import brentq
import numpy as np
from   numba import njit, prange, set_num_threads
import numba as nb

set_num_threads(7)

####### Module variables
dt                   = 1e-4 #seconds
mass                 = 1.6735575e-27 # kg, mass Hbar
bohr_magneton        = 9.2740100657e-24 # J/T, CODATA 2022
kB                   = 1.380649e-23   # J/K
Temperature          = 5e-3 # K
Temperature_transv   = 5e-3 # K
harmonic_degree      = 2  # degree of the harmonic potential
harmonic_coefficient = 50 # coefficient harmonic potential
Zmin                 = -0.605 # m
Zmax                 = -0.481 # m
Zmid                 = -0.544 # m2
zacceptance_min      = -0.775
zacceptance_max      = -0.325
mu_eff_over_m        =  (bohr_magneton) / mass
# tau_mixing           = int(170/2)
# theta = np.pi/4
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
    Potential = lambda z: (CONSTANTS.bohr_magneton) * Bfield(z) # define the potential

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

# Rotation around an axis (3d)
@njit
def rotation_matrix_axis_angle(axis, theta):
    # normalizzazione manuale
    nx, ny, nz = axis[0], axis[1], axis[2]
    norm = np.sqrt(nx*nx + ny*ny + nz*nz)
    nx, ny, nz = nx/norm, ny/norm, nz/norm

    c, s, C = np.cos(theta), np.sin(theta), 1 - np.cos(theta)

    R = np.empty((3,3))
    R[0,0], R[0,1], R[0,2] = c + nx*nx*C, nx*ny*C - nz*s, nx*nz*C + ny*s
    R[1,0], R[1,1], R[1,2] = ny*nx*C + nz*s, c + ny*ny*C, ny*nz*C - nx*s
    R[2,0], R[2,1], R[2,2] = nz*nx*C - ny*s, nz*ny*C + nx*s, c + nz*nz*C

    return R

@njit
def sample_random_axis():
    # phi uniforme in [0, 2pi]
    phi = 2.0 * np.pi * np.random.rand()
    
    # u = cos(theta) uniforme in [-1,1]
    u = 2.0 * np.random.rand() - 1.0
    
    theta = np.arccos(u)
    
    # coordinate cartesiane dell'asse
    nx = np.sin(theta) * np.cos(phi)
    ny = np.sin(theta) * np.sin(phi)
    nz = np.cos(theta)
    
    return np.array([nx, ny, nz])

@njit
def idx_nearest(z, z0, dz, N):
    # z0 = zgrid[0], dz = zgrid[1]-zgrid[0], N = len(zgrid)
    idx = int((z - z0)/dz + 0.5)   # round to nearest
    if idx < 0:
        idx = 0
    elif idx >= N:
        idx = N - 1
    return idx

@njit(fastmath=True)
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

    r2 = (x*x + y*y) # compute radius

    if r2 > 0:
        coeff_z = (1 + harmonic_coefficient * r2)
        coeff_x = harmonic_degree * harmonic_coefficient * x
        coeff_y = harmonic_degree * harmonic_coefficient * y
    else:
        coeff_z = (1 + harmonic_coefficient * r2)
        coeff_x = 0.0
        coeff_y = 0.0
    
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

@njit(fastmath=True)
def dB_plateau(z, x, y, z0, dz, dBgrid, Bgrid):
    N = dBgrid.shape[0]
    i = idx_nearest(z, z0, dz, N)

    r2 = (x*x + y*y) # compute radius
    
    if r2 > 0:
        coeff_z = (1 + harmonic_coefficient * r2)
        coeff_x = harmonic_degree * harmonic_coefficient * x
        coeff_y = harmonic_degree * harmonic_coefficient * y
    else:
        coeff_z = (1 + harmonic_coefficient * r2)
        coeff_x = 0.0
        coeff_y = 0.0
    
    # Compute Bfield gradient at alpha = 0, ramp start 
    dB_dx = Bgrid[i]  * coeff_x
    dB_dy = Bgrid[i]  * coeff_y
    dB_dz = dBgrid[i] * coeff_z
    
    return dB_dx, dB_dy, dB_dz

@njit(parallel=True,fastmath=False)
def step_all_particles(
    R, V,                             # Coordinates (N,3)
    Time, seg_id, Tloc, ramp_len,     # Time and segment identification
    z0, dz,                           # zgrid[0], zgrid[1]-zgrid[0]
    dBinit, Binit, dBfinal, Bfinal,   # Magnetic Field and Gradient
    Annih, T_ann,                     # Time and flag annihilation
    dB_buffer,                        # (unused now, kept for compatibility)
    theta, tau_mixing
):
    N = R.shape[0]

    # precompute constants used everywhere
    half_dt   = 0.5 * dt
    half_dt2  = 0.5 * dt * dt
    prob      = dt / tau_mixing

    #for i in range(N):  <---- SERIAL
    for i in prange(N):# <---- Parallel
        if Annih[i]:
            continue

        # load scalars
        x  = R[i, 0]
        y  = R[i, 1]
        z  = R[i, 2]
        vx = V[i, 0]
        vy = V[i, 1]
        vz = V[i, 2]

        # --- gradient at current step ---
        if seg_id == 0:  # ramp
            dBx, dBy, dBz = dB_seg(
                z, x, y,
                Tloc, ramp_len,
                z0, dz,
                dBinit, Binit,
                dBfinal, Bfinal
            )
        else:            # seg_id == 1: wait
            dBx, dBy, dBz = dB_plateau(
                z, x, y,
                z0, dz,
                dBfinal, Bfinal
            )

        # acceleration (scalars)
        ax = -mu_eff_over_m * dBx
        ay = -mu_eff_over_m * dBy
        az = -mu_eff_over_m * dBz

        # --- Verlet position update (scalars) ---
        x_new = x + vx*dt + ax*half_dt2
        y_new = y + vy*dt + ay*half_dt2
        z_new = z + vz*dt + az*half_dt2

        # --- gradient at next step ---
        if seg_id == 0:  # ramp
            dBx2, dBy2, dBz2 = dB_seg(
                z_new, x_new, y_new,
                Tloc + dt, ramp_len,
                z0, dz,
                dBinit, Binit,
                dBfinal, Bfinal
            )
        else:            # wait
            dBx2, dBy2, dBz2 = dB_plateau(
                z_new, x_new, y_new,
                z0, dz,
                dBfinal, Bfinal
            )

        ax2 = -mu_eff_over_m * dBx2
        ay2 = -mu_eff_over_m * dBy2
        az2 = -mu_eff_over_m * dBz2

        # --- Velocity update (scalars) ---
        vx = vx + (ax + ax2) * half_dt
        vy = vy + (ay + ay2) * half_dt
        vz = vz + (az + az2) * half_dt

        # --- store back ---
        R[i, 0] = x_new
        R[i, 1] = y_new
        R[i, 2] = z_new
        V[i, 0] = vx
        V[i, 1] = vy
        V[i, 2] = vz

        # --- ?annihilation? ---
        if (z_new < zacceptance_min) or (z_new > zacceptance_max):
            Annih[i] = True
            T_ann[i] = Time + dt
            continue  # opzionale: evita mixing se annichilito

        # --- Mixing ----
        if np.random.rand() < prob:
            # qui puoi lasciare il tuo codice così com'è
            axis = sample_random_axis()
            Matrix = rotation_matrix_axis_angle(axis, theta)

            # Matrix @ V[i] senza temporanei: moltiplicazione scalare
            vx0 = V[i, 0]; vy0 = V[i, 1]; vz0 = V[i, 2]
            V[i, 0] = Matrix[0,0]*vx0 + Matrix[0,1]*vy0 + Matrix[0,2]*vz0
            V[i, 1] = Matrix[1,0]*vx0 + Matrix[1,1]*vy0 + Matrix[1,2]*vz0
            V[i, 2] = Matrix[2,0]*vx0 + Matrix[2,1]*vy0 + Matrix[2,2]*vz0


@njit(fastmath=False)
def evolve_all_particles(R_array, V_array, Time,
        z0, dz,
        Annihilation, Time_Annihilation,
        dB_buffer,
        theta, tau_mixing,
        dBfield_20mT_pregravity_dz,
        Bfield_20mT_pregravity_grid,
        dBfield_17mT_dz,
        Bfield_17mT_grid,
        dBfield_5mT_dz,
        Bfield_5mT_grid,
        dBfield_2p5mT_dz,
        Bfield_2p5mT_grid,
        Tramp1, wait_ramp1, 
        Tramp2, wait_ramp2, 
        Tramp3, wait_ramp3):

    #------------------------------------------
    # Wait for Thermalization of the particles
    for frame in range(0,int(40/dt)):

        inverse_time = -40 + frame*dt
        
        # step di tutte le particelle (in-place)
        seg_id   = 1   # plateau
        Tloc     = 0.0
        ramp_len = 0.0
        dBinit   = dBfield_20mT_pregravity_dz  # non usato
        Binit    = Bfield_20mT_pregravity_grid 
        dBfinal  = dBfield_20mT_pregravity_dz  # usi questo
        Bfinal   = Bfield_20mT_pregravity_grid
        step_all_particles(
            R_array, V_array, inverse_time,
            seg_id, Tloc, ramp_len,
            z0, dz,
            dBinit, Binit, dBfinal, Bfinal,
            Annihilation, Time_Annihilation,
            dB_buffer,
            theta, tau_mixing)
        # No time update
        #Time += dt
    #------------------------------------------
    
    for frame in range(0,int(wait_ramp3/dt)):
        #print(Time)
    
        # ================================================================
        # Identify Segment
        # ================================================================
            # identifica il segmento e i campi da passare
        if 0 <= Time < Tramp1:
            seg_id   = 0
            Tloc     = Time - 0.0
            ramp_len = Tramp1
            dBinit   = dBfield_20mT_pregravity_dz
            Binit    = Bfield_20mT_pregravity_grid
            dBfinal  = dBfield_17mT_dz
            Bfinal   = Bfield_17mT_grid
    
        elif Tramp1 <= Time < wait_ramp1:
            seg_id   = 1   # plateau
            Tloc     = 0.0
            ramp_len = 0.0
            dBinit   = dBfield_20mT_pregravity_dz  # non usato
            Binit    = Bfield_20mT_pregravity_grid 
            dBfinal  = dBfield_17mT_dz             # usi questo
            Bfinal   = Bfield_17mT_grid
    
        elif wait_ramp1 <= Time < Tramp2:
            seg_id   = 0
            Tloc     = Time - wait_ramp1
            ramp_len = Tramp2 - wait_ramp1
            dBinit   = dBfield_17mT_dz
            Binit    = Bfield_17mT_grid
            dBfinal  = dBfield_5mT_dz
            Bfinal   = Bfield_5mT_grid
    
        elif Tramp2 <= Time < wait_ramp2:
            seg_id   = 1
            Tloc     = 0.0
            ramp_len = 0.0
            dBinit   = dBfield_17mT_dz
            Binit    = Bfield_17mT_grid
            dBfinal  = dBfield_5mT_dz
            Bfinal   = Bfield_5mT_grid
    
        elif wait_ramp2 <= Time < Tramp3:
            seg_id   = 0
            Tloc     = Time - wait_ramp2
            ramp_len = Tramp3 - wait_ramp2
            dBinit   = dBfield_5mT_dz
            Binit    = Bfield_5mT_grid
            dBfinal  = dBfield_2p5mT_dz
            Bfinal   = Bfield_2p5mT_grid
    
        else:
            seg_id   = 1
            Tloc     = 0.0
            ramp_len = 0.0
            dBinit   = dBfield_5mT_dz
            Binit    = Bfield_5mT_grid
            dBfinal  = dBfield_2p5mT_dz
            Bfinal   = Bfield_2p5mT_grid
    
        # step di tutte le particelle (in-place)
        step_all_particles(
            R_array, V_array, Time,
            seg_id, Tloc, ramp_len,
            z0, dz,
            dBinit, Binit, dBfinal, Bfinal,
            Annihilation, Time_Annihilation,
            dB_buffer,
            theta, tau_mixing)
    
        # Avanza il tempo UNA volta per frame
        Time += dt