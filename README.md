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

For the Jacobi iteration algorithm, I have 3 backends:

### OpenACC + MPI + CPU

This implementation paralizes the Jacobi iteration using OpenACC and MPI on the 
cpu. 

### OpenACC + MPI + GPU

Pretty much the same as the previous one, but this time the Jacobi iteration is 
done on the GPU. CUDA-aware MPI is used to facilitate direct GPU-GPU 
communication on clusters that support it.

### OpenACC + CUDA Graphs + NCCL

This implementation uses CUDA graphs and NCCL to try and speed up the code.
It tends to perform about the same as OpenACC + MPI + GPU, with a little bit 
of extra overhead coming from the CUDA graph instantiation.
