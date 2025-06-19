import numpy as np
#left state
rho1=1
u1=0
p1=1
#right state
rho2=0.125
u2=0
p2=0.1

interface=0
gamma=1.4
Riemann_solver_list=[0,1,2,3,4]
flux_limiter_list=[0,1,3]
reconstruction_list=[0,1,2]
from utils import VanLeer,Minmod,Floor_Energy
class Sob_solver:
    def __init__(self,Ncells=100,Length=1.0,RIEMANNSOLVER=None,FLUXLIMITTER=None,RCONSTRUCTION=None):
      
        self.ncells=Ncells
        print(f"The number of cells is \n{self.ncells}\n")         
        self.RSopt=RIEMANNSOLVER
        if self.RSopt==0:
            print("============ Selected RS: Steger-Warming ============\n")
        elif self.RSopt==1:
            print("============ Selected RS: HLL ============\n")
        elif self.RSopt==2:
            print("============ Selected RS: Lax_Friedrichs ============\n")
        if self.RSopt not in Riemann_solver_list:
            raise Exception("The available Riemann solvers are:\n\
                             0: Steger-Warming\n\
                             1: HLL\n\
                             2: Lax_Friedrichs")
            
        self.FLopt=FLUXLIMITTER
        if self.FLopt not in flux_limiter_list:
            raise Exception("The available flux limiters are:\n\
                             0: minmod\n\
                             1: Van Leer")
            
        self.reconstruction=RCONSTRUCTION
        if self.reconstruction not in reconstruction_list:
            raise Exception("The available reconstruction methods are:\n\
                             0: 2nd order GVC\n\
                             1: 2nd order TVD\n\
                             2: 5th order WENO")

        
        self.sol=np.zeros((3,self.ncells),dtype=np.float64)
        self.sol_old=np.zeros((3,self.ncells),dtype=np.float64)
        

        #store restructed value
        self.rec_l=np.zeros((3,self.ncells+1),dtype=np.float64)
        self.rec_r=np.zeros((3,self.ncells+1),dtype=np.float64)

        #for TVD restruction
        self.D=np.zeros_like(self.sol)
        self.del_U=np.zeros_like(self.rec_l)

        self.flux=np.zeros((3,self.ncells+1),dtype=np.float64)

        #boundary handling--2 ghost cells
        self.vLL=np.zeros((3,1),dtype=np.float64)
        self.vL=np.zeros((3,1),dtype=np.float64)
        self.vRR=np.zeros((3,1),dtype=np.float64)
        self.vR=np.zeros((3,1),dtype=np.float64)

        self.dx=Length/float(self.ncells)
        self.mesh=np.linspace(-Length/2,Length/2,self.ncells)

        for i in range(self.ncells):
            if self.mesh[i]<=interface:
                self.sol[0][i]=rho1
                self.sol[1][i]=u1
                self.sol[2][i]=p1
            else:
                self.sol[0][i]=rho2
                self.sol[1][i]=u2
                self.sol[2][i]=p2

        self.set_boundary()
        self.prim2Con()
    
    def set_boundary(self):
        '''
        2 ghost cells each side
        '''
        self.vL=self.sol[:,0].copy().reshape(-1,1)
        self.vLL=self.sol[:,0].copy().reshape(-1,1)
        self.vR=self.sol[:,-1].copy().reshape(-1,1)
        self.vRR=self.sol[:,-1].copy().reshape(-1,1)
        
    def con2Prim(self):
        '''
        conservative variables(e.g.:rho,momentum,energy) to primary variables(e.g.:rho,velocity,pressure)
        '''
        temp1=self.sol[0,:].copy()
        temp2=self.sol[1,:]/self.sol[0,:]
        temp3=(1.4-1.0)*self.sol[2,:]-(1.4-1.0)/2.0*self.sol[1,:]*self.sol[1,:]/self.sol[0,:]
        self.sol[0,:]=temp1.copy()
        self.sol[1,:]=temp2.copy()
        self.sol[2,:]=temp3.copy()

        bc_list=[self.vLL,self.vL,self.vR,self.vRR]
        for bc in bc_list:
            temp1=bc[0][0]
            temp2=bc[1][0]/bc[0][0]
            temp3=(1.4-1.0)*bc[2][0]-(1.4-1.0)/2.0*bc[1][0]*bc[1][0]/bc[0][0]
            bc[0][0],bc[1][0],bc[2][0]=temp1,temp2,temp3

    def prim2Con(self):
        '''
        primary variables(e.g.:rho,velocity,pressure) to conservative variables(e.g.:rho,momentum,energy)
        '''
        temp1=self.sol[0,:].copy()
        temp2=self.sol[0,:]*self.sol[1,:]
        temp3=self.sol[0,:]*(0.5*self.sol[1,:]*self.sol[1,:])+1.0/(1.4-1.0)*self.sol[2,:]
        self.sol[0,:]=temp1.copy()
        self.sol[1,:]=temp2.copy()
        self.sol[2,:]=temp3.copy()

        bc_list=[self.vLL,self.vL,self.vR,self.vRR]
        for bc in bc_list:
            temp1=bc[0][0]
            temp2=bc[1][0]*bc[0][0]
            temp3=bc[0][0]*(0.5*bc[1][0]*bc[1][0])+1.0/(1.4-1.0)*bc[2][0]
            bc[0][0],bc[1][0],bc[2][0]=temp1,temp2,temp3
            
    def reconstruction_TVD_centered(self):
        '''
        Important: when reconstruction, use unify tensor operation expressions instead of handling it point by point
        -->it easily causes pressure(energy) to be negative
        '''
        #calculate the difference between the center values
        self.del_U=np.hstack((self.sol,self.vR))-np.hstack((self.vL,self.sol))
        del_UL=self.vL-self.vLL
        del_UR=self.vRR-self.vR

        self.D=self.B_minmod(self.del_U[:,0:-1],self.del_U[:,1:])
        DL=self.B_minmod(del_UL,self.del_U[:,0].reshape(-1,1))
        DR=self.B_minmod(self.del_U[:,-1].reshape(-1,1),del_UR)

        self.rec_l[:,1:]=self.sol+0.5*self.D  #U_{j+1/2}^-=U_j+0.5*slope
        self.rec_l[:,0]=(self.vL+0.5*DL).ravel()
        self.rec_r[:,0:-1]=self.sol-0.5*self.D
        self.rec_r[:,-1]=(self.vR-0.5*DR).ravel()

###    
    # def reconstruction_GVC(self):
    #     U_pad=np.hstack([self.vLL,self.vL,self.sol,self.vR,self.vRR])
    #     UL=np.zeros((3,self.ncells),dtype=np.float64)
    #     UR=np.zeros((3,self.ncells),dtype=np.float64)
        
    #     for j in range(self.ncells): 
    #         for k in range(3): 
    #             u_0 =U_pad[k,j+1]   # U_j
    #             u_n1=U_pad[k,j]     # U_{j-1}
    #             u_p1=U_pad[k,j+2]   # U_{j+1}
    #             slope_L=self.B_minmod(u_0-u_n1,u_p1-u_0)
    #             UL[k,j]=u_0+0.5*slope_L
    #             slope_R=self.B_minmod(u_p1-u_0,u_0-u_n1)
    #             UR[k,j]=u_0-0.5*slope_R

        #         u_p2=U_pad[k,j+3]   # U_{j+2}
    
        #         dif1=abs(u_p1-u_0)
        #         dif2=abs(u_0-u_n1)
        #         dif3=abs(u_p1-u_p2)
        #         if dif2 < dif1:
        #             UL[k,j]=(3*u_0-u_n1)/2
        #         else:
        #             UL[k,j]=(u_0+u_p1)/2
    
        #         if dif3 < dif2:
        #             UR[k,j]=(3*u_p1-u_p2)/2
        #         else:
        #             UR[k,j]=(u_p1+u_0)/2
###
    

    def reconstruction_GVC(self):
        U_pad=np.hstack([self.vL,self.sol,self.vR]) #one ghost cell each side is enough
    
        diff_backward=self.sol-U_pad[:,:-2]         # U_j-U_{j-1},shape (3,ncells)
        diff_forward =U_pad[:,2:]-self.sol          # U_{j+1}-U_j,shape (3,ncells)
        slope=self.B_minmod(diff_backward,diff_forward) 

        self.rec_l[:,1:]=self.sol+0.5*slope
        self.rec_r[:,:-1]=self.sol-0.5*slope
    
        dlL=self.sol[:,[0]]-self.vL                
        dlLL=self.vL-self.vLL                        
        slope_L=self.B_minmod(dlLL,dlL) 
        drR=self.vR-self.sol[:,[-1]]                
        drRR=self.vRR-self.vR                       
        slope_R=self.B_minmod(drR,drRR)   
        self.rec_l[:,0]=(self.vL+0.5*slope_L).ravel()           
        self.rec_r[:,-1]=(self.vR-0.5*slope_R).ravel()
    
        self.rec_l=np.clip(self.rec_l,1e-8,1e8)
        self.rec_r=np.clip(self.rec_r,1e-8,1e8)

   
    def reconstruction_WENO5(self):
        eps=1e-6
        U_pad=np.hstack([self.vLL,self.vL,self.sol,self.vR,self.vRR])
        UL=np.zeros((3,self.ncells),dtype=np.float64)
        UR=np.zeros((3,self.ncells),dtype=np.float64)
        for j in range(self.ncells): 
            for k in range(3): 
                u_n2=U_pad[k,j]
                u_n1=U_pad[k,j+1]
                u_0 =U_pad[k,j+2]
                u_p1=U_pad[k,j+3]
                u_p2=U_pad[k,j+4]
    
                # candidate
                p0=(1/3)*u_n2-(7/6)*u_n1+(11/6)*u_0
                p1=-(1/6)*u_n1+(5/6)*u_0+(1/3)*u_p1
                p2=(1/3)*u_0+(5/6)*u_p1-(1/6)*u_p2
    
                # smoothness
                beta0=(13/12)*(u_n2-2*u_n1+u_0)**2+(1/4)*(u_n2-4*u_n1+3*u_0)**2
                beta1=(13/12)*(u_n1-2*u_0+u_p1)**2+(1/4)*(u_n1-u_p1)**2
                beta2=(13/12)*(u_0-2*u_p1+u_p2)**2+(1/4)*(3*u_0-4*u_p1+u_p2)**2
    
                # ideal weights
                g0,g1,g2=0.1,0.6,0.3
                
                alpha0=g0/(eps+beta0)**2
                alpha1=g1/(eps+beta1)**2
                alpha2=g2/(eps+beta2)**2
                alpha_sum=alpha0+alpha1+alpha2
                w0=alpha0/alpha_sum
                w1=alpha1/alpha_sum
                w2=alpha2/alpha_sum
    
                UL[k,j]=w0*p0+w1*p1+w2*p2
    
                p0r=(1/3)*u_p2-(7/6)*u_p1+(11/6)*u_0
                p1r=-(1/6)*u_p1+(5/6)*u_0+(1/3)*u_n1
                p2r=(1/3)*u_0+(5/6)*u_n1-(1/6)*u_n2
                beta0r=(13/12)*(u_p2-2*u_p1+u_0)**2+(1/4)*(u_p2-4*u_p1+3*u_0)**2
                beta1r=(13/12)*(u_p1-2*u_0+u_n1)**2+(1/4)*(u_p1-u_n1)**2
                beta2r=(13/12)*(u_0-2*u_n1+u_n2)**2+(1/4)*(3*u_0-4*u_n1+u_n2)**2
                alpha0r=g0/(eps+beta0r)**2
                alpha1r=g1/(eps+beta1r)**2
                alpha2r=g2/(eps+beta2r)**2
                alpha_sum_r=alpha0r+alpha1r+alpha2r    
                w0r=alpha0r/alpha_sum_r
                w1r=alpha1r/alpha_sum_r
                w2r=alpha2r/alpha_sum_r    
                UR[k,j]=w0r*p0r+w1r*p1r+w2r*p2r
                
        self.rec_l[:,1:]=UL
        self.rec_r[:,:-1]=UR
        self.rec_l[:,0]=self.vL.ravel()
        self.rec_r[:,-1]=self.vR.ravel()
    
    def SW_flux(self):
        fluxL=np.zeros((3,self.ncells+1),dtype=np.float64)
        fluxR=np.zeros((3,self.ncells+1),dtype=np.float64)
        # Left state
        rL=self.rec_l[0,:]
        uL=self.rec_l[1,:]/rL
        EL=self.rec_l[2,:]/rL
        pL=(gamma-1)*(self.rec_l[2,:]-0.5*rL*uL**2)
        aL=np.sqrt(gamma*pL/rL)
        HL=(self.rec_l[2,:]+pL)/rL
    
        # Right state
        rR=self.rec_r[0,:]
        uR=self.rec_r[1,:]/rR
        ER=self.rec_r[2,:]/rR
        pR=(gamma-1)*(self.rec_r[2,:]-0.5*rR*uR**2)
        aR=np.sqrt(gamma*pR/rR)
        HR=(self.rec_r[2,:]+pR)/rR
        
        for i in range(self.ncells+1):
            lam1L=uL[i]-aL[i]
            lam2L=uL[i]
            lam3L=uL[i]+aL[i]
    
            lam1L_p=0.5*(lam1L+abs(lam1L))
            lam2L_p=0.5*(lam2L+abs(lam2L))
            lam3L_p=0.5*(lam3L+abs(lam3L))
    
            alpha1L=(pL[i]/(2*aL[i]**2))
            alpha2L=rL[i]-pL[i]/aL[i]**2
            alpha3L=(pL[i]/(2*aL[i]**2))
    
            fluxL=lam1L_p*np.array([1,uL[i]-aL[i],HL[i]-uL[i]*aL[i]])*alpha1L+lam2L_p*np.array([1,uL[i],0.5*uL[i]**2])*alpha2L+lam3L_p*np.array([1,uL[i]+aL[i],HL[i]+uL[i]*aL[i]])*alpha3L
    
            lam1R=uR[i]-aR[i]
            lam2R=uR[i]
            lam3R=uR[i]+aR[i]    
            lam1R_n=0.5*(lam1R-abs(lam1R))
            lam2R_n=0.5*(lam2R-abs(lam2R))
            lam3R_n=0.5*(lam3R-abs(lam3R))   
            alpha1R=(pR[i]/(2*aR[i]**2))
            alpha2R=rR[i]-pR[i]/aR[i]**2
            alpha3R=(pR[i]/(2*aR[i]**2))
    
            fluxR=lam1R_n*np.array([1,uR[i]-aR[i],HR[i]-uR[i]*aR[i]])*alpha1R+lam2R_n*np.array([1,uR[i],0.5*uR[i]**2])*alpha2R +lam3R_n*np.array([1,uR[i]+aR[i],HR[i]+uR[i]*aR[i]])*alpha3R
    
            self.flux[:,i]=fluxL+fluxR

    def LF_flux(self):
        rL=self.rec_l[0,:]
        uL=self.rec_l[1,:]/rL
        EL=self.rec_l[2,:]
        pL=(gamma-1)*(EL-0.5*rL*uL**2)
        aL=np.sqrt(gamma*pL/rL)
    
        rR=self.rec_r[0,:]
        uR=self.rec_r[1,:]/rR
        ER=self.rec_r[2,:]
        pR=(gamma-1)*(ER-0.5*rR*uR**2)
        aR=np.sqrt(gamma*pR/rR)
    
        FL=np.zeros((3,self.ncells+1))
        FR=np.zeros((3,self.ncells+1))
        FR[0,:]=rR*uR
        FR[1,:]=rR*uR**2+pR
        FR[2,:]=uR*(ER+pR)
    
        FL[0,:]=rL*uL
        FL[1,:]=rL*uL**2+pL
        FL[2,:]=uL*(EL+pL)
    
        alpha=np.maximum(np.abs(uL)+aL,np.abs(uR)+aR)
        self.flux[:,:]=0.5*(FL+FR)-0.5*alpha*(self.rec_r-self.rec_l)

    def HLL_flux(self):
        ncells=self.ncells
        rL=self.rec_l[0,:].copy()
        uL=self.rec_l[1,:]/rL
        EL=self.rec_l[2,:]/rL
        pL=(1.4-1)*(self.rec_l[2,:]-rL*uL*uL/2.0)
        aL=np.sqrt(1.4*pL/rL)
        HL=(self.rec_l[2,:]+pL)/rL

        rR=self.rec_r[0,:].copy()
        uR=self.rec_r[1,:]/rR
        ER=self.rec_r[2,:]/rR
        pR=(1.4-1)*(self.rec_r[2,:]-rR*uR*uR/2.0)
        aR=np.sqrt(1.4*pR/rR)
        HR=(self.rec_r[2,:]+pR)/rR

        # roe averages
        ratio=np.sqrt(rR/rL)
        r=ratio*rL
        u=(uL+ratio*uR)/(1+ratio)
        H=(HL+ratio*HR)/(1+ratio)
        a=np.sqrt((1.4-1)*(H-u*u/2.0))

        FL=np.vstack((rL* uL,rL*uL**2+pL,(rL*EL+pL)*uL))
        FR=np.vstack((rR* uR,rR*uR**2+pR,(rR*ER+pR)*uR))

        S_L=(u-a).reshape(1,-1)
        S_R=(u+a).reshape(1,-1)

        den=S_R-S_L
        den=np.where(np.abs(denom) < 1e-6,1e-6,denom)
        F_hll=(S_R*FL-S_L*FR+S_L*S_R*(self.rec_r-self.rec_l))/den
        for i in range(ncells+1):
            if S_L[0,i] >= 0:
                self.flux[:,i]=FL[:,i]
            elif S_R[0,i]<=0:
                self.flux[:,i]=FR[:,i]
            else:
                self.flux[:,i]=F_hll[:,i]

            
            
    def time_advancement(self,cfl):
        r=np.maximum(self.sol[0,:],1e-8)
        u=self.sol[1,:]/r
        E=self.sol[2,:]/r
        p=np.maximum((1.4-1)*(self.sol[2,:]-r*u**2/2.0),1e-8)
        a=np.sqrt(1.4*p/r)
        Smax=np.max(np.vstack((abs(u+a),abs(u-a),abs(u))))
        dt=cfl*self.dx/Smax

        U0= self.sol.copy()

        if self.reconstruction==1:
            Reconstruction=self.reconstruction_TVD_centered
        elif self.reconstruction==0:
            Reconstruction=self.reconstruction_GVC
        elif self.reconstruction==2:
            Reconstruction=self.reconstruction_WENO5

        # choose Riemann solver
        if self.RSopt==0:
            Flux=self.SW_flux
        elif self.RSopt==1:
            Flux=self.HLL_flux
        elif self.RSopt==2:
            Flux=self.LF_flux

        print("Before Reconstruction: min/max/nan in sol:",
            np.min(self.sol),np.max(self.sol),np.isnan(self.sol).any())

        Reconstruction()
        Flux()
        F=self.flux      
        L0=-(F[:,1:]-F[:,:-1])/self.dx
        U1=U0+dt*L0
        
        self.sol=U1
        # self.sol[0,:]=np.maximum(self.sol[0,:],1e-6)      
        # self.sol[2,:]=Floor_Energy(self.sol)
        self.set_boundary()

        print("Before Reconstruction: min/max/nan in sol:",
            np.min(self.sol),np.max(self.sol),np.isnan(self.sol).any())
        Reconstruction()
        Flux()
        F=self.flux
        L1=-(F[:,1:]-F[:,:-1])/self.dx
        U2=0.75*U0+0.25*(U1+dt*L1)

        self.sol=U2
        # self.sol[0,:]=np.maximum(self.sol[0,:],1e-6)     
        # self.sol[2,:]=Floor_Energy(self.sol)
        self.set_boundary()

        print("Before Reconstruction: min/max/nan in sol:",
            np.min(self.sol),np.max(self.sol),np.isnan(self.sol).any())
        Reconstruction()
        Flux()
        F=self.flux
        L2=-(F[:,1:]-F[:,:-1])/self.dx
        self.sol=(1.0/3.0)*U0+(2.0/3.0)*(U2+dt*L2)
        # self.sol[0,:]=np.maximum(self.sol[0,:],1e-6)      
        # self.sol[2,:]=Floor_Energy(self.sol)
        self.set_boundary()

        
        return dt

    def B_minmod(self,a: np.ndarray,b: np.ndarray):
        return 0.5*(np.sign(a)+np.sign(b))*np.minimum(np.abs(a),np.abs(b))
        