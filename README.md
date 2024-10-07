# ADV HPC
## Exercise 1: Distributed Multi-GPU Matrix Multiplication
Here I test the scaling of various matrix multiplication implementations.
This is done as follows:

 - Strong scaling: keep the size of the matrix the same while increasing number 
   of nodes.
 - Weak scaling: Increase the size of the matrix linearly with the number of 
   nodes.

Scaling is done by node, with 4 processes per node.
Although both the Naive and DGEMM implementations could use all cores on a node 
in a single process, the GPU implementation cannot (cores need to be split 
between GPUS).

### P1: Naive CPU Only
This implementation involves a simple parallelized loop in regular C + 
OpenACC (CPU). The matrices are in column major layout because it's faster.

### P2: CPU Only with DGEMM (cblas_dgemm)
Here, instead of using a loop we use the BLAS implementation (cblas_dgemm).
Because this is automatically multithreaded there's no need for OpenACC, 
although the rest of the code still uses it.

### P3: CPU with GPU and DGEMM (cublas_dgemm)
Lastly, we test the Cuda BLAS implementation (cublasDgemm). 
As described earlier, each process is assigned to it's nearest GPU which it uses
to accelerate the code.

## Exercise 2: Jacobi Algorithm

