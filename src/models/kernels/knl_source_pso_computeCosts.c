#define n_PATH %d
#define n_PERIOD %d

/* Unified Z, St shape as [nPath by nPeriod] */

__kernel void getEuroOption(__global float *Z, float S0, float K, float r, float sigma, float T, 
    char opt, __global float *out){
    
    /* pre calc dt, nudt, volsdt, lnS0*/
    private float dt = T / n_PERIOD;
    private float nudt = (r - 0.5* sigma*sigma) * dt;
    private float volsdt = sigma * sqrt(dt);
    private float lnS0 = log(S0);
    
    /* one thread per path, gid is current working thread/path */
    int path_id = get_global_id(0);
    
    float St = S0;
    float last_tmp = S0;
    float deltaSt = 0.0f;
    
    // simulate log normal price
    /* loop thru n_PERIOD for each path to get maturity price */
    for(int cur_t = 0; cur_t < n_PERIOD; cur_t++){
        last_tmp = deltaSt;      // set St to lastSt for next period calc
        /* get corresponding z */
        float z = Z[cur_t + n_PERIOD * path_id];    // updated on 6 Apr. 2025 for unified Z, St shape as nPath by nPeriod
        deltaSt = nudt + volsdt * z;
        deltaSt += last_tmp;
    }
    St = exp(deltaSt + lnS0); 
    
    /* return C_hat of Euro option */
    float path_C_hat = exp(-r*T) * max(0.0f, (K - St)*opt);
    out[path_id] = path_C_hat; 
}


/* 
        # # udpated on 6 Apr. 2025 
        # 1. Unified Z, St shape as [nPath by nPeriod], synced and shared by PSO and Longstaff
        # 2. No concatenation of spot price
        # 3. handle index of time period, spot price at time zero (present), St from time 1 to T
        # 4. init boundary index to maturity and exercise to last period St, as track early exercise backwards in time 
*/
__kernel void psoAmerOption_gb(global float *St, global float *pso, global float *C_hat, 
                        global int *boundary_idx, global float *exercise,
                        float r, float T, float K, char opt){
    
    //global variables, one fish per thread (work-item), check for early aross, for all paths
    int gid = get_global_id(0);            //thread id, per fish
    int nParticle = get_global_size(0);    //number of fishes
    
    int boundary_gid;                 // shared global access id for boundary_idx & exercise
    float cur_fish_val;               // current fish element value, pointer to loop thru current fish dimension, i.e. time t for St
    float cur_St_val;                 // current St element value, pointer to loop thru current path at time t of St
    int St_T_idx;                     // St_T id for all paths

    // init intermediate buffer: boundary index to maturity and exercise to last period St, as track early exercise backwards in time 
    for (int path = 0; path < n_PATH; path++){         
        boundary_gid = gid + path * nParticle;        // calc shared global access id for boundary_idx & exercise
        St_T_idx = (n_PERIOD - 1) + path * n_PERIOD;  // calc St_T id for all paths
        boundary_idx[boundary_gid] = n_PERIOD - 1;    // reset boundary index to time T
        exercise[boundary_gid] = St[St_T_idx];        // reset exercise to St_T
    }
    
    /* set intermediate arrays of index and exercise */
    //outer loop thru periods (Note that fish dimension is equal to St periods), loop backwards in time to track early exercise point for each path
    for (int prd= n_PERIOD - 1; prd > -1 ; prd--){
        //e.g. total 5 fishes, nParticle = 5; total 3 time steps, nPeriod = 3
        //gid=0: prd:[2, 1, 0] --> 0 + [2 1 0] * nParticle =  PSO global index [10, 5, 0] for fish 0
        //gid=1: prd:[2, 1, 0] --> 1 + [2 1 0] * nParticle =  PSO global index [11, 6, 1] for fish 1
        cur_fish_val = pso[gid + prd * nParticle];    // PSO global index & value

        //inner loop thru all St paths at current period
        //St global pointer from 0 to (nPath * nPeriod -1)
        for (int path= 0; path < n_PATH; path++){
            //e.g. total 3 periods, nPeriod = 3
            //prd: 2  path:[0, 1, 2, 3] --> 2 + [0, 1, 2, 3] * nPeriod = St global index [2, 5, 8, 11] for period 2
            //prd: 1  path:[0, 1, 2, 3] --> 1 + [0, 1, 2, 3] * nPeriod = St global index [1, 4, 7, 10] for period 2
            cur_St_val = St[prd + path * n_PERIOD];    // get St path value at same period
            
            // each fish access to corresponding column of boundary_idx and exercise matrix, both nPath by nFish/nParticle
            // calc shared global access id for boundary_idx & exercise
            // e.g. total 5 fishes, nParticle=5, total 10 paths, nPath=10
            // gid=0: --> 0 + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * nParticle = boundary_idx/exercise global index [0, 5, 10, 15, 20..45]
            // gid=1: --> 1 + [0, 1, 2, 3, 4, 5, 6, 7, 8, 9] * nParticle = boundary_idx/exercise global index [1, 6, 11, 16, 21..46]
            boundary_gid = gid + path * nParticle;  
            
            // check if first cross: 1) pso > St; 2) corresponding boundary index not set from previous loops
            if (cur_fish_val > cur_St_val){  
                boundary_idx[boundary_gid] = prd;        //store current time step
                exercise[boundary_gid] = cur_St_val;     //store current price
            } 
            
            // // if no cross for certain path, set index to last time step and exercise to path last price
            // // check if no cross: 1) current period is last period; 2) boundary index not set from previous loops
            // if (prd == 0  &&  boundary_idx[boundary_gid]==0){    
            //     boundary_idx[boundary_gid] = n_PERIOD;             // set index to maturity time step
            //     exercise[boundary_gid] = cur_St_val;          // exercise price at maturity
            // }
        }
    }
    
    /* calc C_hat for current fish */
    float tmp_C = 0.0f;
    float dt = T / n_PERIOD;
    // input parameter opt is the Put/Call flag, 1 for Put, -1 for Call
    for (int path = 0; path < n_PATH; path++){         // sum all path C_hat
        boundary_gid = gid + path * nParticle;         // calc shared global access id for boundary_idx & exercise
        tmp_C += exp(-r * (boundary_idx[boundary_gid]+1) * dt) * max(0.0f, (K - exercise[boundary_gid]) * opt);   // boudnary_idx +1 to reflect actual time step, considering present is time zero
    }
    
    C_hat[gid] = tmp_C / n_PATH;     // get average C_hat for current fish/thread investigation
}
