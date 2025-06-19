import numpy as np
from exactpack.solvers.riemann.ep_riemann import IGEOS_Solver

def VanLeer(r):
    return (r+np.abs(r))/(1.0+np.abs(r)+1e-8)
def Minmod(r):
    if r<=0:
        return 0
    elif 0<r<=1:
        return r
    else:
        return 1

def Floor_Energy(sol):
    rho = np.maximum(sol[0,:],1e-8)
    mom = sol[1,:]
    E = sol[2,:]
    
    u = mom/rho
    kinetic = 0.5 * rho * u**2
    
    # minimal internal energy，防止 p<0
    min_internal_energy=1e-8
    E_min=kinetic+min_internal_energy

    return np.maximum(E,E_min)

def exact(t,L,n):
    riem1_ig_soln=IGEOS_Solver(rl=1.0,  ul=0.,  pl=1.0, gl=1.4,
                             rr=0.125,ur=0.,  pr=0.1, gr=1.4,
                             xmin=-L/2.0, xd0=0.0,xmax=L/2.0,t=t)
    x=np.linspace(-L/2,L/2,n)
    riem1_ig_result=riem1_ig_soln._run(x,t)
    return riem1_ig_result
    