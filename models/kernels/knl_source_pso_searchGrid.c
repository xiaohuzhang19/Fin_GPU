#define n_Dim %d

/* update nParticle positions & velocity, each thread handles on particle */
__kernel void searchGrid(global float *position, global float *velocity, 
                        global float *pbest_pos, global float *gbest_pos,
                        global float *r1, global float *r2, float w, float c1, float c2){
                        
    int gid = get_global_id(0);              // thread id presenting one particle per work-item
    int fish_size = get_global_size(0);      // number of threads/particles
    int item_id = 0;                         // item_id for accessing position, velocity, pbest_pos, r1, r2
    
    /* global memory access, this one works!! */
    for (int i=0; i<n_Dim; i++){             // i for gbest_pos indexing
        item_id = gid + i * fish_size;       // determine item_id
        
        velocity[item_id] = w * velocity[item_id] + c1 * r1[item_id] * (pbest_pos[item_id]-position[item_id]) + \
                            c2 * r2[item_id] * (gbest_pos[i] - position[item_id]);
                            
        position[item_id] += velocity[item_id];
    }
}