#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=jacobi_scaling_benchmark
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --mem=480gb
#SBATCH --time=03:00:00
#SBATCH --output=./logs/ex1test%j.out
#SBATCH --gpus-per-task=1
set -a; source .env set +a

# Load all the required modules
module load "$CUDA_MOD"
module load "$GCC_MOD"
module load "$CMAKE_MOD"
module load "$NVHPC_MOD"
module load "$MPI_MOD"
module load "$PYTHON_MOD"
module load "$BLAS_MOD"


export NCCL_LIB_DIR=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/comm_libs/nccl/lib
export NCCL_INCLUDE_DIR=/leonardo/prod/spack/06/install/0.22/linux-rhel8-icelake/gcc-8.5.0/nvhpc-24.5-torlmnyzcexnrs6pq4cccabv7ehkv3xy/Linux_x86_64/24.5/comm_libs/nccl/include
export LIBRARY_PATH="$NCCL_LIB_DIR:$LIBRARY_PATH"
export CMAKE_BUILD_DIR="cmake-build-dir-jac$SLURM_NTASKS"
export C_COMP=nvc
export CXX_COMP=nvc++

#echo "Compiling naive algorithm:"

#cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$C_COMP -DCMAKE_CXX_COMPILER=$CXX_COMP -DNAIVE=ON -S . -B $CMAKE_BUILD_DIR
#cd $CMAKE_BUILD_DIR || exit 1
#make
#cd ..
#
#echo "Testing naive algorithm:"
#python ex2/test_jacobi.py -b $CMAKE_BUILD_DIR/ex_2 -o jacobi_naive
#
#echo "Deleting the build directory:"
#rm -r $CMAKE_BUILD_DIR
#
#
# regular CUDA
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$C_COMP -DCMAKE_CXX_COMPILER=$CXX_COMP -DUSE_GPU=ON -S . -B $CMAKE_BUILD_DIR
cd $CMAKE_BUILD_DIR || exit 1
make
cd ..

echo "Testing CUDA algorithm:"
python ex2/test_jacobi.py -b $CMAKE_BUILD_DIR/ex_2 -o jacobi_gpu -g

echo "Deleting the build directory:"
rm -r $CMAKE_BUILD_DIR


# CUDA Graphs
#cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=$C_COMP -DCMAKE_CXX_COMPILER=$CXX_COMP -DUSE_GPU=ON -DUSE_CUDA_GRAPHS=ON -S . -B $CMAKE_BUILD_DIR
#cd $CMAKE_BUILD_DIR || exit 1
#make
#cd ..
#
#echo "Testing CUDA Graphs algorithm:"
#python ex2/test_jacobi.py -b $CMAKE_BUILD_DIR/ex_2 -o jacobi_gpu_graphs -g
#
#echo "Deleting the build directory:"
#rm -r $CMAKE_BUILD_DIR