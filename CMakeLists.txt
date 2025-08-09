cmake_minimum_required(VERSION 3.27)
project(adv_hpc C)

set(CMAKE_C_STANDARD 11)
find_package(MPI REQUIRED)
find_package(OpenACC)

include_directories(SYSTEM ${MPI_INCLUDE_PATH})
SET(CMAKE_C_FLAGS "${CMAKE_C_FLAGS}")

add_executable(
        ex_1
        ex1/matr_mult.c
)
add_executable(
        ex_2
        ex2/jacobi.c
)

if (NAIVE)
    add_compile_definitions(NAIVE)
    target_compile_options(ex_1 PRIVATE "${OpenACC_C_OPTIONS}=multicore")
    target_link_libraries(ex_1 PRIVATE ${MPI_C_LIBRARIES} ${OpenACC_C_OPTIONS} m)

    target_compile_options(ex_2 PRIVATE "${OpenACC_C_OPTIONS}=multicore")
    target_link_libraries(ex_2 PRIVATE "${MPI_C_LIBRARIES}" "${OpenACC_C_OPTIONS}" m)
elseif (USE_GPU)
    find_package(CUDAToolkit REQUIRED)
    include(cmake/Modules/FindNCCL.cmake)
    add_compile_definitions(USE_GPU)
    target_compile_options(ex_1 PRIVATE "${OpenACC_C_OPTIONS}")
    target_link_libraries(ex_1 PRIVATE "${MPI_C_LIBRARIES}" "${OpenACC_C_OPTIONS}" CUDA::cublas m)
    target_compile_options(ex_2 PRIVATE "${OpenACC_C_OPTIONS}" -gpu=tripcount:host)
    target_link_libraries(ex_2 PRIVATE "${MPI_C_LIBRARIES}" "${OpenACC_C_OPTIONS}" CUDA::cublas "${NCCL_LIBNAME}" m)
elseif (USE_BLAS)
    include(FindBLAS)
    if (BLAS_FOUND)
        add_compile_definitions(USE_BLAS)
        target_compile_options(ex_1 PRIVATE "${BLAS_LINKER_FLAGS} ${OpenACC_C_OPTIONS}")
        target_link_libraries(ex_1 "${BLAS_LIBRARIES}" "${MPI_C_LIBRARIES}" m )
    endif ()
    target_compile_options(ex_2 PRIVATE "${OpenACC_C_OPTIONS}")
    target_link_libraries(ex_2 PRIVATE "${MPI_C_LIBRARIES}" "${OpenACC_C_OPTIONS}" m )
endif()

