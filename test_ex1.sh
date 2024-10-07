#!/bin/bash
#SBATCH --partition=boost_usr_prod
#SBATCH --job-name=matr_matr_scaling_benchmark
#SBATCH --cpus-per-task=8
#SBATCH --ntasks-per-node=4
#SBATCH --ntasks=256
#SBATCH --mem=300gb
#SBATCH --time=00:40:00
#SBATCH --output=./logs/ex1test%j.out
#SBATCH --gpus-per-task=1
set -a; source .env set +a

# Load all the required modules
module load "$CMAKE_MOD"
module load "$NVHPC_MOD"
module load "$MPI_MOD"
module load "$PYTHON_MOD"
module load "$BLAS_MOD"

CMAKE_BUILD_DIR="cmake-build-dir"

echo "Compiling naive algorithm:"

cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=nvc -DNAIVE=ON -S . -B $CMAKE_BUILD_DIR
cd $CMAKE_BUILD_DIR || exit 1
make
cd ..

echo "Testing naive algorithm:"
python ex1/test_matr_mult.py -b $CMAKE_BUILD_DIR/ex_1 -o naive.json

echo "Deleting the build directory:"
rm -r $CMAKE_BUILD_DIR

# BLAS
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=nvc -DUSE_BLAS=ON -S . -B $CMAKE_BUILD_DIR
cd $CMAKE_BUILD_DIR || exit 1
make
cd ..

echo "Testing BLAS algorithm:"
python ex1/test_matr_mult.py -b $CMAKE_BUILD_DIR/ex_1 -o blas.json

echo "Deleting the build directory:"
rm -r $CMAKE_BUILD_DIR


# CUDA
cmake -DCMAKE_BUILD_TYPE=Release -DCMAKE_C_COMPILER=nvc -DUSE_GPU=ON -S . -B $CMAKE_BUILD_DIR
cd $CMAKE_BUILD_DIR || exit 1
make
cd ..

echo "Testing CUDA algorithm:"
python ex1/test_matr_mult.py -b $CMAKE_BUILD_DIR/ex_1 -o gpu.json

echo "Deleting the build directory:"
rm -r $CMAKE_BUILD_DIR
