import numpy as np
#left state
rho1=1
u1=0
p1=1
#right state
rho2=0.125
u2=0
p2=0.1
#interface
interface=0
gamma=1.4
from uils import VanLeer, Minmod
class Sob_solver:
    def __init__(self, fname = "settings.txt", Length=1.0):
        file = open(fname, mode = "r")
        key_dict = {
                    "NCELLS": 1,
                    "RIEMANNSOLVER": 2,
                    "FLUXLIMITTER": 3,
                    "RCONSTRUCTION": 4}
        lines = file.readlines()
        iL = 0
        found_key = 0
        reconstruction_list = [0, 1, 2]
        flux_limiter_list = [0, 1]
        Riemann_solver_list = [0, 1, 2, 3, 4]
        while iL<len(lines):
            line = lines[iL].split()
            if len(line) < 1:
                iL += 1
                continue
            
            if found_key == 0:
                if line[0] in key_dict:
                    found_key = key_dict[line[0]]
            elif found_key == 1:
                self.ncells = int(lines[iL])
                print(f"The number of cells is \n{self.ncells}\n")
                found_key = 0
            elif found_key == 2:
                self.RSopt = int(lines[iL])
                if self.RSopt == 0:
                    print("============ Selected RS: Exact ===========\n")
                elif self.RSopt == 1:
                    print("============ Selected RS: Roe =============\n")
                elif self.RSopt == 2:
                    print("============ Selected RS: HLL ============\n")
                elif self.RSopt == 3:
                    print("============ Selected RS: SW ============\n")
                elif self.RSopt == 4:
                    print("============ Selected RS: Lax_Friedrichs ============\n")
                if self.RSopt not in Riemann_solver_list:
                    raise Exception("The available Riemann solvers are:\n\
                                     0: Exact\n\
                                     1: Roe\n\
                                     2: AUSM\n\
                                     3: HLLC\n\
                                     4: Lax_Friedrichs")
                found_key = 0
            elif found_key == 3:
                self.FLopt = int(lines[iL])
                if self.FLopt not in flux_limiter_list:
                    raise Exception("The available flux limiters are:\n\
                                     0: minmod\n\
                                     1: Van Leer")
                    raise Exception("Other Flux limitter options not supported yet!")
                found_key = 0
            elif found_key == 4:
                self.reconstruction = int(lines[iL])
                if self.reconstruction not in reconstruction_list:
                    raise Exception("The available reconstruction methods are:\n\
                                     0: 2nd order GVC\n\
                                     1: 2nd order TVD\n\
                                     2: 5th order WENO")

            iL += 1

        
        # store the solution on the cell centers
        self.sol = np.zeros((3, self.ncells), dtype = np.float64)
        self.sol_old = np.zeros((3, self.ncells), dtype = np.float64)

        # store the reconstructed values on the cell interface
        self.rec_l = np.zeros((3, self.ncells+1), dtype = np.float64)
        self.rec_r = np.zeros((3, self.ncells+1), dtype = np.float64)

        # store the limitted slope for centered TVD-MUSCL reconstruction
        self.D = np.zeros_like(self.sol)

        # store the difference between every cell interface
        self.del_U = np.zeros_like(self.rec_l)

        # store the averaged value on the interface 
        self.avg = np.zeros((3, self.ncells+1), dtype = np.float64)

        # store the flux on the cell interface
        self.flux = np.zeros((3, self.ncells+1), dtype = np.float64)

        # store the virtual cells to impose boundary conditions
        self.vLL = np.zeros((3,1), dtype = np.float64)
        self.vL = np.zeros((3,1), dtype = np.float64)
        self.vRR = np.zeros((3,1), dtype = np.float64)
        self.vR = np.zeros((3,1), dtype = np.float64)

        # store the eigen vectors for every interface
        self.Li = np.zeros((3,3,self.ncells+1),dtype=np.float64)
        self.Ri = np.zeros((3,3,self.ncells+1),dtype=np.float64)

        # store the eigen value for every interface
        self.eigi = np.zeros((3, self.ncells+1),dtype=np.float64)
    
        # calculate the mesh information
        self.dx = Length / float(self.ncells)
        self.mesh = np.linspace(-0.0 + 0.5*self.dx, Length - 0.5*self.dx, self.ncells)

        # set the initial condition
        for i in range(self.ncells):
            if self.mesh[i] <= self.interface:
                self.sol[0][i] = self.rho1
                self.sol[1][i] = self.u1
                self.sol[2][i] = self.p1
            else:
                self.sol[0][i] = self.rho2
                self.sol[1][i] = self.u2
                self.sol[2][i] = self.p2

        self.set_boundary()
        self.prim2Con(0)
    
    def set_boundary(self):
        self.vL = self.sol[:, 0].copy().reshape(-1,1)
        self.vLL = self.sol[:, 0].copy().reshape(-1,1)
        self.vR = self.sol[:, -1].copy().reshape(-1,1)
        self.vRR = self.sol[:, -1].copy().reshape(-1,1)
    def con2Prim(self):
        '''
        ### Descripition:

        Convert the conservative variables to primitive variables. 

        ### Input:

        `mode`: if `mode == 0`, then convert the cell center values (inlcuding virtual cells)
        elsewise, the reconstructed value will be converted
        '''
        temp1 = self.sol[0, :].copy()
        temp2 = self.sol[1, :] / self.sol[0, :]
        temp3 = (1.4-1.0)*self.sol[2, :]-(1.4-1.0)/2.0*self.sol[1, :]*self.sol[1, :]/self.sol[0, :]

        self.sol[0, :] = temp1.copy()
        self.sol[1, :] = temp2.copy()
        self.sol[2, :] = temp3.copy()

        bc_list = [self.vLL, self.vL, self.vR, self.vRR]
        for bc in bc_list:
            temp1 = bc[0][0]
            temp2 = bc[1][0] / bc[0][0]
            temp3 = (1.4-1.0)*bc[2][0] - (1.4-1.0) / 2.0 * bc[1][0] * bc[1][0] / bc[0][0]
            bc[0][0], bc[1][0], bc[2][0] = temp1, temp2, temp3

    def prim2Con(self, mode = 0):
        temp1 = self.sol[0, :].copy()
        temp2 = self.sol[0, :] * self.sol[1, :]
        temp3 = self.sol[0, :] * (0.5 * self.sol[1, :] * self.sol[1, :]) + 1.0 / (1.4 - 1.0) * self.sol[2, :]

        self.sol[0, :] = temp1.copy()
        self.sol[1, :] = temp2.copy()
        self.sol[2, :] = temp3.copy()

        bc_list = [self.vLL, self.vL, self.vR, self.vRR]
        for bc in bc_list:
            temp1 = bc[0][0]
            temp2 = bc[1][0] * bc[0][0]
            temp3 = bc[0][0] * (0.5 * bc[1][0] * bc[1][0]) + 1.0 / (1.4 - 1.0) * bc[2][0]
            bc[0][0], bc[1][0], bc[2][0] = temp1, temp2, temp3

    def reconstruction_TVD_centered(self):
        UL = np.zeros((3, self.ncells), dtype=np.float64)
        UR = np.zeros((3, self.ncells), dtype=np.float64)
        U_pad = np.hstack([self.vLL, self.vL, self.sol, self.vR, self.vRR])
        limiter = Minmod if self.FLopt == 0 else VanLeer
        for j in range(self.ncells):
            for k in range(3): 
                u_jm1 = U_pad[k, j]
                u_j   = U_pad[k, j+1]
                u_jp1 = U_pad[k, j+2]
                u_jp2 = U_pad[k, j+3]
                delta = u_jp1 - u_j
    
                if abs(delta) < 1e-8:
                    phi_L = phi_R = 0.0
                else:
                    f_L = (u_j - u_jm1) / delta
                    f_R = (u_jp2 - u_jp1) / delta
                    phi_L = limiter(f_L)
                    phi_R = limiter(f_R)
    
                UL[k, j] = u_j + 0.5 * phi_L * delta
                UR[k, j] = u_jp1 - 0.5 * phi_R * delta
    
        self.rec_l[:, 1:] = UL
        self.rec_r[:, :-1] = UR
        self.rec_l[:, 0] = self.vL.ravel()
        self.rec_r[:, -1] = self.vR.ravel()

    
    def reconstruction_WENO5(self):
        eps = 1e-6
        U_pad = np.hstack([self.vLL, self.vL, self.sol, self.vR, self.vRR])
    
        for j in range(self.ncells + 1): 
            for k in range(3): 
                u_n2 = U_pad[k, j]
                u_n1 = U_pad[k, j+1]
                u_0  = U_pad[k, j+2]
                u_p1 = U_pad[k, j+3]
                u_p2 = U_pad[k, j+4]
    
                # candidate stencils (left-biased)
                p0 = (1/3)*u_n2 - (7/6)*u_n1 + (11/6)*u_0
                p1 = -(1/6)*u_n1 + (5/6)*u_0 + (1/3)*u_p1
                p2 = (1/3)*u_0 + (5/6)*u_p1 - (1/6)*u_p2
    
                # smoothness indicators (beta)
                beta0 = (13/12)*(u_n2 - 2*u_n1 + u_0)**2 + (1/4)*(u_n2 - 4*u_n1 + 3*u_0)**2
                beta1 = (13/12)*(u_n1 - 2*u_0 + u_p1)**2 + (1/4)*(u_n1 - u_p1)**2
                beta2 = (13/12)*(u_0 - 2*u_p1 + u_p2)**2 + (1/4)*(3*u_0 - 4*u_p1 + u_p2)**2
    
                # ideal weights
                g0, g1, g2 = 0.1, 0.6, 0.3
    
                # nonlinear weights
                alpha0 = g0 / (eps + beta0)**2
                alpha1 = g1 / (eps + beta1)**2
                alpha2 = g2 / (eps + beta2)**2
                alpha_sum = alpha0 + alpha1 + alpha2
    
                w0 = alpha0 / alpha_sum
                w1 = alpha1 / alpha_sum
                w2 = alpha2 / alpha_sum
    
                # final reconstruction (left value of interface j)
                UL[k, j] = w0*p0 + w1*p1 + w2*p2
    
                # symmetric stencil for right-biased (for rec_r[:, j])
                p0r = (1/3)*u_p2 - (7/6)*u_p1 + (11/6)*u_0
                p1r = -(1/6)*u_p1 + (5/6)*u_0 + (1/3)*u_n1
                p2r = (1/3)*u_0 + (5/6)*u_n1 - (1/6)*u_n2
    
                beta0r = (13/12)*(u_p2 - 2*u_p1 + u_0)**2 + (1/4)*(u_p2 - 4*u_p1 + 3*u_0)**2
                beta1r = (13/12)*(u_p1 - 2*u_0 + u_n1)**2 + (1/4)*(u_p1 - u_n1)**2
                beta2r = (13/12)*(u_0 - 2*u_n1 + u_n2)**2 + (1/4)*(3*u_0 - 4*u_n1 + u_n2)**2
    
                alpha0r = g0 / (eps + beta0r)**2
                alpha1r = g1 / (eps + beta1r)**2
                alpha2r = g2 / (eps + beta2r)**2
                alpha_sum_r = alpha0r + alpha1r + alpha2r
    
                w0r = alpha0r / alpha_sum_r
                w1r = alpha1r / alpha_sum_r
                w2r = alpha2r / alpha_sum_r
    
                UR[k, j] = w0r*p0r + w1r*p1r + w2r*p2r
                
        self.rec_l[:, 1:] = UL
        self.rec_r[:, :-1] = UR
        self.rec_l[:, 0] = self.vL.ravel()
        self.rec_r[:, -1] = self.vR.ravel()
    
    def SW_flux(self):
        fluxL=np.zeros((3, self.ncells+1), dtype=np.float64)
        fluxR=np.zeros((3, self.ncells+1), dtype=np.float64)
        # Left state
        rL = self.rec_l[0, :]
        uL = self.rec_l[1, :] / rL
        EL = self.rec_l[2, :] / rL
        pL = (gamma - 1) * (self.rec_l[2, :] - 0.5 * rL * uL**2)
        aL = np.sqrt(gamma * pL / rL)
        HL = (self.rec_l[2, :] + pL) / rL
    
        # Right state
        rR = self.rec_r[0, :]
        uR = self.rec_r[1, :] / rR
        ER = self.rec_r[2, :] / rR
        pR = (gamma - 1) * (self.rec_r[2, :] - 0.5 * rR * uR**2)
        aR = np.sqrt(gamma * pR / rR)
        HR = (self.rec_r[2, :] + pR) / rR
        for i in range(self.ncells + 1):
            # Left eigen decomposition and flux splitting
            lam1L = uL[i] - aL[i]
            lam2L = uL[i]
            lam3L = uL[i] + aL[i]
    
            lam1L_p = 0.5 * (lam1L + abs(lam1L))
            lam2L_p = 0.5 * (lam2L + abs(lam2L))
            lam3L_p = 0.5 * (lam3L + abs(lam3L))
    
            alpha1L = (pL[i] / (2 * aL[i]**2))
            alpha2L = rL[i] - pL[i] / aL[i]**2
            alpha3L = (pL[i] / (2 * aL[i]**2))
    
            fluxL = lam1L_p * np.array([1, uL[i] - aL[i], HL[i] - uL[i] * aL[i]]) * alpha1L \
                  + lam2L_p * np.array([1, uL[i], 0.5 * uL[i]**2]) * alpha2L \
                  + lam3L_p * np.array([1, uL[i] + aL[i], HL[i] + uL[i] * aL[i]]) * alpha3L
    
            # Right eigen decomposition and flux splitting
            lam1R = uR[i] - aR[i]
            lam2R = uR[i]
            lam3R = uR[i] + aR[i]
    
            lam1R_n = 0.5 * (lam1R - abs(lam1R))
            lam2R_n = 0.5 * (lam2R - abs(lam2R))
            lam3R_n = 0.5 * (lam3R - abs(lam3R))
    
            alpha1R = (pR[i] / (2 * aR[i]**2))
            alpha2R = rR[i] - pR[i] / aR[i]**2
            alpha3R = (pR[i] / (2 * aR[i]**2))
    
            fluxR = lam1R_n * np.array([1, uR[i] - aR[i], HR[i] - uR[i] * aR[i]]) * alpha1R \
                  + lam2R_n * np.array([1, uR[i], 0.5 * uR[i]**2]) * alpha2R \
                  + lam3R_n * np.array([1, uR[i] + aR[i], HR[i] + uR[i] * aR[i]]) * alpha3R
    
            # Total flux
            self.flux[:, i] = fluxL + fluxR

    def LF_flux(self):
        # 左状态变量
        rL = self.rec_l[0, :]
        uL = self.rec_l[1, :] / rL
        EL = self.rec_l[2, :]
        pL = (gamma - 1) * (EL - 0.5 * rL * uL**2)
        aL = np.sqrt(gamma * pL / rL)
    
        # 右状态变量
        rR = self.rec_r[0, :]
        uR = self.rec_r[1, :] / rR
        ER = self.rec_r[2, :]
        pR = (gamma - 1) * (ER - 0.5 * rR * uR**2)
        aR = np.sqrt(gamma * pR / rR)
    
        # 通量（F = [rho*u, rho*u^2 + p, u*(E + p)])
        FL = np.zeros((3, self.ncells + 1))
        FR = np.zeros((3, self.ncells + 1))
    
        FL[0, :] = rL * uL
        FL[1, :] = rL * uL**2 + pL
        FL[2, :] = uL * (EL + pL)
    
        FR[0, :] = rR * uR
        FR[1, :] = rR * uR**2 + pR
        FR[2, :] = uR * (ER + pR)
    
        # 最大波速 alpha = max(|u ± a|)
        alpha = np.maximum(np.abs(uL) + aL, np.abs(uR) + aR)
    
        # LF 通量公式
        self.flux[:, :] = 0.5 * (FL + FR) - 0.5 * alpha * (self.rec_r - self.rec_l)

    def ROE_flux(self):
        ncells = self.ncells
        # left state
        rL = self.rec_l[0, :].copy()
        uL = self.rec_l[1, :]/rL
        EL = self.rec_l[2, :]/rL
        pL = (1.4 - 1) * (self.rec_l[2, :] - rL * uL * uL / 2.0)
        aL = np.sqrt(1.4 * pL / rL)
        HL = (self.rec_l[2, :] + pL) / rL

        # right state
        rR = self.rec_r[0, :].copy()
        uR = self.rec_r[1, :]/rR
        ER = self.rec_r[2, :]/rR
        pR = (1.4 - 1) * (self.rec_r[2, :] - rR * uR * uR / 2.0)
        aR = np.sqrt(1.4 * pR / rR)
        HR = (self.rec_r[2, :] + pR) / rR

        # roe averages
        ratio = np.sqrt(rR / rL)
        r = ratio * rL
        u = (uL + ratio * uR) / (1 + ratio)
        H = (HL + ratio * HR) / (1 + ratio)
        a = np.sqrt((1.4 - 1)*(H - u * u / 2.0))

        delta = 1e-6
        def abs_lambda(lmbd): return np.where(np.abs(lmbd) < delta, 0.5 * (lmbd**2 / delta + delta), np.abs(lmbd))
        λ1 = abs_lambda(u - a)
        λ2 = abs_lambda(u)
        λ3 = abs_lambda(u + a)
        
        drho = rR - rL
        dvel = uR - uL
        dpress = pR - pL
        alpha2 = (1.4 - 1) * (dpress - a * r * dvel) / (a ** 2)
        alpha1 = 0.5 * (drho - dpress / (a ** 2) + alpha2)
        alpha3 = drho - (alpha1 + alpha2)

        r1 = np.array([np.ones(ncells), u - a, H - u * a])
        r2 = np.array([np.ones(ncells), u,     0.5 * u**2])
        r3 = np.array([np.ones(ncells), u + a, H + u * a])

        flux_diff = 0.5 * (
            r1 * (λ1 * alpha1) +
            r2 * (λ2 * alpha2) +
            r3 * (λ3 * alpha3)
        )

        FL = np.vstack((rL * uL, rL * uL * uL + pL, uL * (rL * EL + pL)))
        FR = np.vstack((rR * uR, rR * uR * uR + pR, uR * (rR * ER + pR)))

        self.flux = (FL + FR)/2.0-flux_diff

    def HLL_flux(self):
        ncells = self.ncells
        # left state
        rL = self.rec_l[0, :].copy()
        uL = self.rec_l[1, :]/rL
        EL = self.rec_l[2, :]/rL
        pL = (1.4 - 1) * (self.rec_l[2, :] - rL * uL * uL / 2.0)
        aL = np.sqrt(1.4 * pL / rL)
        HL = (self.rec_l[2, :] + pL) / rL

        # right state
        rR = self.rec_r[0, :].copy()
        uR = self.rec_r[1, :]/rR
        ER = self.rec_r[2, :]/rR
        pR = (1.4 - 1) * (self.rec_r[2, :] - rR * uR * uR / 2.0)
        aR = np.sqrt(1.4 * pR / rR)
        HR = (self.rec_r[2, :] + pR) / rR

        # roe averages
        ratio = np.sqrt(rR / rL)
        r = ratio * rL
        u = (uL + ratio * uR) / (1 + ratio)
        H = (HL + ratio * HR) / (1 + ratio)
        a = np.sqrt((1.4 - 1)*(H - u * u / 2.0))

        FL = np.vstack((rL* uL, rL*uL**2 + pL, (rL*EL + pL) * uL))
        FR = np.vstack((rR* uR, rR*uR**2 + pR, (rR*ER + pR) * uR))

        S_L = (u - a).reshape(1, -1)
        S_R = (u + a).reshape(1, -1)

        F_hll = (S_R * FL - S_L * FR + S_L * S_R * (self.rec_r - self.rec_l)) / (S_R - S_L)
        for i in range(ncells):
            if S_L[0, i] >= 0:
                self.flux[:, i] = FL[:, i]
            elif S_R[0, i] <= 0:
                self.flux[:, i] = FR[:, i]
            else:
                self.flux[:, i] = F_hll[:, i]
            
            
    def time_advancement(self, cfl):
        r = self.sol[0, :]
        u = self.sol[1, :]/r
        E = self.sol[2, :]/r
        p = (1.4 - 1) * (self.sol[2, :] - r * u * u / 2.0)
        a = np.sqrt(1.4 * p / r)
        Smax = np.max(np.vstack((abs(u+a),abs(u-a), abs(u))))
        dt = cfl * self.dx / Smax

        self.sol_old = self.sol.copy()
        alpha1 = [1.0, 3.0/4.0, 1.0/3.0]
        alpha2 = [0.0, 1.0/4.0, 2.0/3.0]
        alpha3 = [1.0, 1.0/4.0, 2.0/3.0]

        for j in range(3):
            # choose Reconstruction method
            if self.reconstruction == 1:
                self.reconstruction_TVD_centered()
            elif self.reconstruction == 0:
                self.reconstruction_GVC()
            elif self.reconstruction == 2:
                self.reconstruction_WENO5()

            # choose Riemann solver
            if self.RSopt == 0:
                self.exact_flux()
            elif self.RSopt == 1:
                self.ROE_flux()
            elif self.RSopt == 2:
                self.HLL_flux()
            elif self.RSopt == 3:
                self.SW_flux(choice = 3)
            elif self.RSopt == 4:
                self.LF_flux(choice = 4)
            
            self.sol = alpha1[j] * self.sol_old + alpha2[j] * self.sol - alpha3[j] * dt/self.dx * (self.flux[:, 1:] - self.flux[:, 0:-1])
        
        return dt
        


