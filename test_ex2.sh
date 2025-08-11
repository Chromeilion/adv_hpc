#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=jacobi_scaling_benchmark
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=128
#SBATCH --mem=480gb
#SBATCH --time=01:30:00
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
module load "$NCCL_MOD"

CMAKE_BUILD_DIR="cmake-build-dir-jac"

echo "Compiling naive algorithm:"

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=nvc -DNAIVE=ON -S . -B $CMAKE_BUILD_DIR
cd $CMAKE_BUILD_DIR || exit 1
make
cd ..

echo "Testing naive algorithm:"
python ex2/test_jacobi.py -b $CMAKE_BUILD_DIR/ex_2 -o jacobi_naive.json

echo "Deleting the build directory:"
rm -r $CMAKE_BUILD_DIR


# CUDA
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=nvc -DUSE_GPU=ON -S . -B $CMAKE_BUILD_DIR
cd $CMAKE_BUILD_DIR || exit 1
make
cd ..

echo "Testing CUDA algorithm:"
python ex2/test_jacobi.py -b $CMAKE_BUILD_DIR/ex_2 -o jacobi_gpu.json -g

echo "Deleting the build directory:"
rm -r $CMAKE_BUILD_DIR
