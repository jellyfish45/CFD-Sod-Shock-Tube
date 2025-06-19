import Sob
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from utils import exact

NCELL=[50,100,200,400]
Length=[1.0,10]

RM_dict={0: "Steger-Warming",1: "HLL",2: "Lax-Friedrichs"}
RC_dict={0: "TVD_GVC",1: "2nd_order_TVD",2: "5th_order_WENO"}

RM=[0,1,2]
RC=[0,1,2]
t_bar=0.14


results=[]

for L in Length: #iterate through different computational fields
    solution_exact=exact(t_bar,L,NCELL[-1])  #exact answer
    mesh_exact=np.linspace(-L/2,L/2,NCELL[-1])
    exact_solver=np.zeros((3,NCELL[-1]))
    exact_solver[0,:]=solution_exact['density']
    exact_solver[1,:]=solution_exact['velocity']
    exact_solver[2,:]=solution_exact['pressure']

    for rm_idx in RM: #different 雷曼 solver
        for rc_idx in RC: #different restruction method
            fig,ax=plt.subplots(1,3,figsize=(15,4),dpi=200,sharex=True)
            y_list=[r"$\rho$",r"$u$",r"$p$"]

            for Ncell in NCELL: #iterate mesh sizes
                FLUXLIMITTER=0 if rc_idx == 1 else 3
                solver=Sob.Sob_solver(Ncells=Ncell,Length=L,
                                        RIEMANNSOLVER=rm_idx,
                                        FLUXLIMITTER=FLUXLIMITTER,
                                        RCONSTRUCTION=rc_idx)
                cfl=0.1
                global_t=0.0
                while global_t<t_bar:
                    dt=solver.time_advancement(cfl=cfl)
                    global_t+=dt
                solver.con2Prim()

                for k in range(3):
                    ax[k].plot(solver.mesh,solver.sol[k,:],label=f"N={Ncell}")
                    
                results.append({
                'Length': L,
                'RiemannSolver': RM_dict[rm_idx],
                'Reconstruction': RC_dict[rc_idx],
                'Ncell': Ncell,
                'mesh': solver.mesh.copy(),
                'density': solver.sol[0,:].copy(),
                'velocity': solver.sol[1,:].copy(),
                'pressure': solver.sol[2,:].copy()
            })

            for k in range(3):
                ax[k].plot(mesh_exact,exact_solver[k,:],label="Exact",linestyle="dashed",linewidth=1)
                ax[k].legend()
                ax[k].grid()
                ax[k].set_xlabel("$x$")
                ax[k].set_ylabel(y_list[k])
                ax[k].set_xlim(-L/2,L/2)
            plt.title(f"Using {RM_dict[rm_idx]} and {RC_dict[rc_idx]} Based on L={L} and Different Cell Numbers")
            plt.tight_layout()
            plt.savefig(f"{RM_dict[rm_idx]}_{RC_dict[rc_idx]}_L{L}.png")  
            #plt.savefig(f"Try_{RM_dict[rm_idx]}_{RC_dict[rc_idx]}_L{L}_N{Ncell}.png")  

df_results=pd.DataFrame(results)
df_results.to_pickle("sod_shock_tube_results.pkl")