import numpy as np
import pyopencl as cl
from .utils import openCLEnv

# Black-Scholes
from scipy.stats import norm

def BlackScholes(S0, K, r, sigma, T, opttype='P'):
    d1 = (np.log(S0/K) + (r + sigma**2/2)*T) / (sigma * np.sqrt(T))
    d2 = d1 - sigma*np.sqrt(T)
    
    call_price = S0*norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)
    put_price = call_price - S0 + np.exp(-r*T)*K
    
    if opttype == 'C':
        # price = S0*norm.cdf(d1) - np.exp(-r*T)*K*norm.cdf(d2)
        price = call_price
    elif opttype == 'P':
        # price = np.exp(-r*T)*K*norm.cdf(-d2) - S0*norm.cdf(-d1) 
        price = put_price
    return price

def BlackScholes_matrix(St, K, r, sigma, T, nPeriod, opttype='P'):
    BS = np.zeros_like(St, dtype=np.float32)
    dt = T / nPeriod

    for t in range(nPeriod):
        new_T = dt * (nPeriod - t)
        BS[:,t] = BlackScholes(St[:,t], K, r, sigma, new_T, 'P')
    return BS

class MonteCarloBase:
    # built-in seeds
    __seed = 1001
    # __seed = np.nan
    def __init__(self, S0, r, sigma, T, nPath, nPeriod, K, opttype):
        # init simulation parameters
        self.S0 = S0
        self.r = r
        self.sigma = sigma
        self.T = T
        self.nPath = nPath
        self.nPeriod = nPeriod
        self.K = K
        self.opttype = opttype
        self.opt = None
        match self.opttype:
            case 'C':
                self.opt = -1
            case 'P':
                self.opt = 1
        self.dt = self.T / self.nPeriod
        # generate St simulation
        self.Z = self.__getZ()
        self.St = self.__getSt()
        self.BS = self.__getBlackScholes_matrix()
        
    @classmethod
    def getSeed(cls):
        return cls.__seed
    
    @classmethod
    def setSeed(cls, seed):
        cls.__seed = seed
        # self.Z = self.__getZ()
        # self.St = self.__getSt()
        return
    
    def __getZ(self):
        # if self.__seed is np.nan:
        #     rng = np.random.default_rng()  
        # else:
        #     rng = np.random.default_rng(seed=self.__seed)  
        #xiaohu update Apri 9th with default seed
        seed = type(self).__seed  # or MonteCarloBase.getSeed()
        rng = np.random.default_rng(seed=seed)
        Z = rng.normal(size=(self.nPath, self.nPeriod)).astype(np.float32)  
        return Z
    
    def __getSt(self):
        # pre-compute Geometric Brownian Motion parameters
        nudt = (self.r - 0.5 * self.sigma**2) * self.dt       # drift component
        volsdt = self.sigma * np.sqrt(self.dt)                # diffusion component
        lnS0 = np.log(self.S0)                           # using log normally distributed feature
        
        # log price approach
        delta_lnSt = nudt + volsdt * self.Z    # nPeriod by nPath
        lnSt = lnS0 + np.cumsum(delta_lnSt, axis=1) 
        # lnSt = np.concatenate( (np.full(shape=(1, nPath), fill_value=lnS0), lnSt))
        St = np.exp(lnSt).astype(np.float32)
        
        return St
    
    def __getBlackScholes_matrix(self):

        BS = np.zeros_like(self.St, dtype=np.float32)
        dt = self.T / self.nPeriod
        
        for t in range(self.nPeriod):
            new_T = dt * (self.nPeriod - t)
            BS[:,t] = BlackScholes(self.St[:,t], self.K, self.r, self.sigma, new_T, self.opttype)
        return BS
    
    # get MC St payoffs
    def getPayoffs(self):
        # immediate exercise payoffs for each path and time step
        payoffs = np.maximum(0, (self.K - self.St) * self.opt)
        
        return payoffs

    
# Monte Carlo Simulation for American Put Option Pricing
class hybridMonteCarlo(MonteCarloBase):
    def __init__(self, S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish):
        super().__init__(S0, r, sigma, T, nPath, nPeriod, K, opttype)
        # initialize parameters
        self.nFish = nFish
        
        # prepare kernel, buffer
        prog = cl.Program(openCLEnv.context, open("./models/kernels/knl_source_pso_computeCosts.c").read()%(nPath, nPeriod)).build()
        self.knl_getEuroOption = cl.Kernel(prog, 'getEuroOption')
        self.knl_psoAmerOption_gb = cl.Kernel(prog, 'psoAmerOption_gb')
        
        # init buffer for Z and St for Pso 
        self.Z_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.Z)
        self.St_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_ONLY | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.St)
        
        # array and memory objects for Pso fitFunction costPsoAmerOption_cl()
        # init boundary index to maturity and exercise to last period St, as track early exercise backwards in time (set in kernel code)
        self.boundary_idx = np.zeros(shape=(self.nPath, self.nFish), dtype=np.int32) #+ nPeriod
        self.exercise = np.zeros(shape=(self.nPath, self.nFish), dtype=np.float32) #+ self.St[:, -1].reshape(nPath, 1)
        self.boundary_idx_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.boundary_idx)
        self.exercise_d = cl.Buffer(openCLEnv.context, cl.mem_flags.READ_WRITE | cl.mem_flags.COPY_HOST_PTR, hostbuf=self.exercise)

        # # Initialize shared arrays for Pso child classes
        nDim = self.nPeriod
        self.pos_init = np.random.uniform(size=(nDim, nFish)).astype(np.float32) * self.S0
        self.vel_init = np.random.uniform(size=(nDim, nFish)).astype(np.float32) * 5.0
        self.r1 = np.random.uniform(size=(nDim, nFish)).astype(np.float32)
        self.r2 = np.random.uniform(size=(nDim, nFish)).astype(np.float32)

    def __str__(self):
        return str(self.__class__) + ": " + str(self.__dict__)
    
    # Monte Carlo European option - CPU
    def getEuroOption_np(self):
        assert (self.St.shape[0] == (np.exp(-self.r*self.T) * np.maximum(0, (self.K - self.St[:, -1]) * self.opt) ).shape[0])
        C_hat_Euro = (np.exp(-self.r*self.T) * np.maximum(0, (self.K - self.St[:, -1]) * self.opt) ).sum() / self.nPath
    
        print(f"MonteCarlo Numpy European price: {C_hat_Euro}")
        return C_hat_Euro

    # Monte Carlo European option - GPU
    def getEuroOption_cl(self):            
        # prepare result array, length of nPath for kernel threads
        results = np.empty(self.nPath, dtype=np.float32)  # length of npath
        results_d = cl.Buffer(openCLEnv.context, cl.mem_flags.WRITE_ONLY, size=results.nbytes)
            
        self.knl_getEuroOption.set_args(self.Z_d, np.float32(self.S0), np.float32(self.K), 
                                  np.float32(self.r), np.float32(self.sigma), 
                                  np.float32(self.T), np.int8(self.opt), results_d)
        
        # run kernel
        global_size = (self.nPath, )
        local_size = None
        evt = cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_getEuroOption, global_size, local_size)
        
        cl.enqueue_copy(openCLEnv.queue, results, results_d, wait_for=[evt])
        openCLEnv.queue.finish()    # <------- sychrnozation
        
        C_hat_Euro = results.sum() / self.nPath
        
        print(f"MonteCarlo {openCLEnv.deviceName} European price: {C_hat_Euro}")
        return C_hat_Euro

    # Monte Carlo pso American option - CPU: take one particle each time and loop thru PSO
    def costPsoAmerOption_np(self, in_particle):
        # # udpated on 6 Apr. 2025 
        # 1. for unified Z, St shape as nPath by nPeriod, synced and shared by PSO and Longstaff
        # 2. No concatenation of spot price
        # 3. handle index of time period, spot price at time zero (present), St from time 1 to T

        # get the boundary index where early cross (particle period > St period), as if an early exercise judgement by this fish/particle
        boundaryIdx = np.argmax(self.St < in_particle[None, :], axis=1)   # [0, 1] as of true or false of early cross
        # if no, set boundary index to last time period, meaning no early exercise suggested for that path
        boundaryIdx[boundaryIdx==0] = self.nPeriod - 1    # to handle time T index for boundary index to match St time wise dimension (i.e. indexing from zero)
        
        # determine exercise prices by getting the early cross St_ij on path i and period j
        exerciseSt = self.St[np.arange(len(boundaryIdx)), boundaryIdx]    # len of boundaryIdx is nPath
        
        # discounted back to time zero, hence boundaryIdx+1
        searchCost = (np.exp(-self.r * (boundaryIdx+1) * self.dt) * np.maximum(0, (self.K - exerciseSt)*self.opt) ).sum() / self.nPath

        return searchCost

    # Monte Carlo pso American option - GPU: take the whole PSO and process once
    def costPsoAmerOption_cl(self, pso_buffer, costs_buffer):
        self.knl_psoAmerOption_gb.set_args(self.St_d, pso_buffer, costs_buffer, 
                                           self.boundary_idx_d, self.exercise_d, 
                                           np.float32(self.r), np.float32(self.T), np.float32(self.K), np.int8(self.opt))

        # execute kernel
        global_size = (self.nFish, )
        local_size = None
        evt = cl.enqueue_nd_range_kernel(openCLEnv.queue, self.knl_psoAmerOption_gb, global_size, local_size)
        openCLEnv.queue.finish()    # <------- sychrnozation

        return 

    def cleanUp(self):
        self.Z_d.release()
        self.St_d.release()
        self.boundary_idx_d.release()
        self.exercise_d.release()
        return 



def main():
    S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish = 100.0, 0.03, 0.3, 1.0, 10, 3, 100.0, 'P', 500
    mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)
    print(mc.St)

if __name__ == "__main__":
    main()
