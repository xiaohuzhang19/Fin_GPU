from models.mc import MonteCarloBase, hybridMonteCarlo
from models.longstaff import LSMC_Numpy, LSMC_OpenCL
import models.benchmarks as bm
from models.pso import PSO_Numpy, PSO_OpenCL
from models.utils import checkOpenCL
    

if __name__ == "__main__":
    checkOpenCL()

    S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish = 617.7, 0.0519, 0.120634, 30/365, 20000, 30, 622.4753, 'P', 500
    #S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish = 100.0, 0.03, 0.3, 1.0, 20000, 200, 110.0, 'P', 500
    # S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish = 100.0, 0.03, 0.3, 1.0, 50000, 200, 110.0, 'P', 500
    mc = hybridMonteCarlo(S0, r, sigma, T, nPath, nPeriod, K, opttype, nFish)
    print(mc.St.shape)

    # benchmarks
    bm.blackScholes(S0, K, r, sigma, T, opttype)
    bm.binomialEuroOption(S0, K, r, sigma, nPeriod, T, opttype)
    mc.getEuroOption_np()
    mc.getEuroOption_cl()
    bm.binomialAmericanOption(S0, K, r, sigma, nPeriod, T, opttype)

    # longstaff
    lsmc_np = LSMC_Numpy(mc)
    lsmc_np.longstaff_schwartz_itm_path_fast()

    lsmc_cl = LSMC_OpenCL(mc)
    lsmc_cl.longstaff_schwartz_itm_path_fast_hybrid()

    # pso
    pso_np = PSO_Numpy(mc, nFish, mc.costPsoAmerOption_np)
    pso_np.solvePsoAmerOption_np()

    pso_cl = PSO_OpenCL(mc, nFish, mc.costPsoAmerOption_cl)
    pso_cl.solvePsoAmerOption_cl()

    # clear up memory
    pso_cl.cleanUp()
    mc.cleanUp()
    
    