//
// Created by chromeilion on 10/4/24.
//
#include <stdio.h>
#include <stdlib.h>
#include <mpi.h>
#include <openacc.h>
#include <time.h>
#ifdef USE_BLAS
#include <cblas.h>
#endif
#ifdef USE_GPU
#include <cublas_v2.h>
#endif

#ifdef USE_GPU
#define CHECK_CUDA(call) \
    do { \
        cublasStatus_t _st = (call); \
        if (_st != CUBLAS_STATUS_SUCCESS) { \
            int _mpi_inited = 0; \
            MPI_Initialized(&_mpi_inited); \
            int _rank = 0; \
            if (_mpi_inited) MPI_Comm_rank(MPI_COMM_WORLD, &_rank); \
            fprintf(stderr, "[rank %d] ERROR: cuBLAS call failed: %s:%d, status=%d\n", \
                    _rank, __FILE__, __LINE__, (int)_st); \
            if (_mpi_inited) MPI_Abort(MPI_COMM_WORLD, (int)_st); \
            exit((int)_st); \
        } \
    } while (0)
#endif

void print_loc( double * mat, int n_row, int n_col){
    for( int i = 0; i < n_row; i++ ){
        for ( int j = 0; j < n_col; j++) {
            fprintf( stdout, "%.6g ", mat[i*n_col+j] );
        }
        fprintf( stdout, "\n" );
    }
}

void print_par( double * mat, int size, int rank, int npes, int flipped){
    MPI_Barrier( MPI_COMM_WORLD );
    int count;
    if( rank )
        MPI_Send( mat, size*(size/npes), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD );
    else{
        double * buf = (double *) calloc( size*(size/npes), sizeof(double) );
        if (flipped) {print_loc( mat, size, size / npes );}
        else {print_loc( mat, size / npes, size );}


        for( count = 1; count < npes; count ++){
            MPI_Recv( buf, size*(size/npes), MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            if (flipped) {print_loc( buf, size, size / npes );}
            else {print_loc( buf, size / npes, size );}
        }
        free( buf );
    }
    fprintf( stdout, "\n" );
}

int main( int argc, char * argv[] ){
    clock_t start;
    int npes, rank;
    double * mat_a, * mat_b, * res, * buf;
    long int n_cols, size_mata, size_matb, n_chunks, current_chunk;
    long int n_rows, res_size, n_rows_loc, idx_matb, matb_sendcount;
    long int stride_mata, stride_matb, current_col;
    long int i, j, k;
    unsigned int size_buf;
    MPI_Datatype column_type;
    double alpha;
    double beta;
    MPI_Init( &argc, & argv );
    MPI_Comm_size( MPI_COMM_WORLD, &npes );
    MPI_Comm_rank( MPI_COMM_WORLD, &rank );
    if (argc != 2) {
        fprintf(stdout, "Please supply both the rows and columns as arguments!!!");
        return 1;
    }
#ifdef USE_GPU
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    CHECK_CUDA(status);
    int ngpu = acc_get_num_devices(acc_device_nvidia);
    int igpu = rank % ngpu;
    acc_set_device_num(igpu, acc_device_nvidia);
    acc_init(acc_device_nvidia);
#endif
    alpha = 1;
    beta = 0;
    n_rows = atoi(argv[1]);
    n_cols = n_rows;
    if (n_rows % npes != 0) {
        fprintf(stdout, "The size of the matrix must be divisible by the number "
                        "of processes!!!");
        return 1;
    }
    n_rows_loc = n_rows / npes;
    n_chunks = npes;
    MPI_Barrier( MPI_COMM_WORLD );
    start = clock();
    fprintf(stdout, "%i 0 s | done initializing\n", rank);
    size_mata = n_cols * n_rows_loc;
    size_matb = size_mata;
    matb_sendcount = n_rows_loc * n_rows_loc;
    size_buf = size_mata;
    stride_mata = n_cols;
    stride_matb = n_rows;
    MPI_Type_vector(n_rows_loc, n_rows_loc, n_cols, MPI_DOUBLE, &column_type);
    MPI_Type_commit(&column_type);
    mat_a = (double *) calloc( size_mata, sizeof(double) );
    mat_b = (double *) calloc( size_matb, sizeof(double) );
    buf = (double *) calloc( size_buf, sizeof(double) );
    res = (double *) calloc( size_mata, sizeof(double) );

    #pragma acc enter data create ( mat_a[ 0 : size_mata ], mat_b[ 0 : size_matb ], res[ 0 : size_mata ], buf [ 0 : size_buf])

    fprintf(stdout, "%i %f s | data allocated\n", rank, (double)(clock()-start)/CLOCKS_PER_SEC);

    // Fill the matrices with values
    #pragma acc parallel loop collapse(2) present( mat_a )
    for ( i = 0; i < n_rows_loc; i++ ) {
        for ( j = 0; j < n_cols; j++ ){
            long int global_i = rank * n_rows_loc + i;
            mat_a[i*n_cols + j] = 0.02 * j + 1.0 + global_i;
        }
    }
    #pragma acc parallel loop collapse(2) present( mat_b )
    for ( i = 0; i < n_rows_loc; i++ ) {
        for( j = 0; j < n_cols; j++ ){
            long int global_i = rank * n_rows_loc + i;
            mat_b[i*n_cols + j] = 0.003 * (j + 1) + 1.0 + global_i;
        }
    }
    fprintf(stdout, "%i %f p | matricies filled with values\n", rank, (double)(clock()-start)/CLOCKS_PER_SEC);
#ifndef NDEBUG
    #pragma acc update self ( mat_b[ 0 : size_matb ] )
    print_par( mat_b, n_cols, rank, npes, 0 );

    #pragma acc update self ( mat_a[ 0 : size_mata ] )
    print_par( mat_a, n_cols, rank, npes, 0 );
#endif
    for ( current_chunk = 0; current_chunk < n_chunks; current_chunk++ ) {
        current_col = current_chunk * n_rows_loc;

        #pragma acc host_data use_device(buf, mat_b)
        MPI_Allgather(&mat_b[current_col], 1, column_type, buf, matb_sendcount, MPI_DOUBLE, MPI_COMM_WORLD);
        fprintf(stdout, "%i %f m | finished Allgather\n", rank, (double)(clock()-start+1)/CLOCKS_PER_SEC);
#ifndef NDEBUG
        if (rank == 0) {
            #pragma acc update self ( buf[ 0 : size_buf] )
            print_loc( buf, n_cols, n_rows_loc);
        }
#endif

#ifdef NAIVE
        #pragma acc loop independent collapse(2)
        for (int i = 0; i < n_rows_loc; ++i) {
            for (int j = 0; j < n_rows_loc; ++j) {
                double sum = 0.0;
                #pragma acc loop independent reduction(+: sum)
                for (int k = 0; k < n_cols; ++k) {
                    sum += mat_a[i * n_cols + k] * buf[k * n_rows_loc + j];
                }
                res[i * n_cols + (current_col + j)] = sum;
            }
        }
#endif
#ifdef USE_BLAS
        #pragma acc host_data use_device(mat_a, buf, res)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_rows_loc, n_rows_loc, n_cols, alpha, mat_a, n_cols, buf, n_rows_loc, beta, &res[current_col], n_cols);
#endif
#ifdef USE_GPU
        #pragma acc host_data use_device(mat_a, buf, res)
        CHECK_CUDA(cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_rows_loc,
                   n_rows_loc, n_cols, &alpha, buf, n_rows_loc, mat_a, n_cols,
                   &beta, &res[current_col], n_cols));
#endif
        fprintf(stdout, "%i %f c | finished Matrix computation\n", rank, (double)(clock()-start)/CLOCKS_PER_SEC);
    }
    #pragma acc exit data copyout( res[ 0 : size_mata ] ) finalize
#ifndef NDEBUG
    print_par( res, n_cols, rank, npes, 0);
#endif
    MPI_Finalize();
    return 0;
}

