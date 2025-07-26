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

void print_loc( double * mat, int n_row, int n_col){
    for( int i = 0; i < n_row; i++ ){
        for ( int j = 0; j < n_col; j++) {
            fprintf( stdout, "%.6g ", mat[i*n_col+j] );
        }
        fprintf( stdout, "\n" );
    }
}

void print_par( double * mat, int size, int rank, int npes, int flipped){
    int count;
    if( rank )
        MPI_Send( mat, size*(size/npes), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD );
    else{
        double * buf = (double *) calloc( size*(size/npes), sizeof(double) );
        if (flipped) {print_loc( buf, size, size / npes );}
        else {print_loc( buf, size / npes, size );}


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
    long int n_cols, size_mata;
    long int n_rows, res_size, n_rows_loc;
    long int i, j;
    unsigned int size_buf;
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
    MPI_Barrier( MPI_COMM_WORLD );
    start = clock();
    fprintf(stdout, "%i 0 s | done initializing\n", rank);
    size_mata = n_cols * n_rows_loc;
    mat_a = (double *) calloc( size_mata, sizeof(double) );
    size_buf = n_cols*n_rows;
    mat_b = (double *) calloc( size_mata, sizeof(double) );
    buf = (double *) calloc( size_buf, sizeof(double) );
    res_size = n_rows * n_rows_loc;
    res = (double *) calloc( res_size, sizeof(double) );

    #pragma acc enter data create ( mat_a[ 0 : size_mata ], mat_b[ 0 : size_mata ], res[ 0 : res_size ], buf [ 0 : size_buf])

    fprintf(stdout, "%i %f s | data allocated\n", rank, (double)(clock()-start)/CLOCKS_PER_SEC);

    // Fill the matrices with values
    #pragma acc parallel loop collapse(2) present( mat_a )
    for (j = 0; j < n_rows_loc; j++) {
        for ( i = 0; i < n_cols; i++ ){
            mat_a[j*n_cols + i] = 0.02 * i + rank + 1 + j;
        }
    }

    #pragma acc parallel loop collapse(1) present( mat_b )
    for (j = 0; j < n_rows_loc; j++) {
        for( i = 0; i < n_cols; i++ ){
            mat_b[j*n_cols + i] = 0.003 * (i + 1) + rank + 1 + j;
        }
    }

#ifndef NDEBUG
    #pragma acc update self ( mat_a[ 0 : size_mata ] )
    print_par( mat_a, n_cols, rank, npes, 0);
#endif
    fprintf(stdout, "%i %f p | matricies filled with values\n", rank, (double)(clock()-start)/CLOCKS_PER_SEC);
    //2) perform allgather
    #pragma acc host_data use_device(buf, mat_b)
    MPI_Allgather(mat_b, size_mata, MPI_DOUBLE, buf, size_mata, MPI_DOUBLE, MPI_COMM_WORLD);
    fprintf(stdout, "%i %f m | finished Allgather\n", rank, (double)(clock()-start+1)/CLOCKS_PER_SEC);

#ifndef NDEBUG
    if (rank == 0) {
        #pragma acc update self ( buf[ 0 : size_buf] )
        print_loc( buf, n_cols, n_rows);
    }
#endif

#ifdef NAIVE
    // local matrix matrix multiplication
    #pragma acc parallel loop collapse(2) present( mat_a, res, buf)
    for (i = 0; i < n_rows_loc; ++i) {
        for (j = 0; j < n_rows; ++j) {
            for (k = 0; k < n_cols; ++k) {
                res[i*n_cols+j] += mat_a[i*n_cols + k] * buf[k*n_rows+j];
         }
      }
    }
#endif
#ifdef USE_BLAS
    #pragma acc host_data use_device(buf, mat_a, res)
    cblas_dgemm(CblasColMajor, CblasNoTrans, CblasNoTrans,
                n_rows_loc, n_rows, n_cols, alpha, mat_a, n_rows_loc, buf, n_cols, beta, res, n_rows_loc);
#endif
#ifdef USE_GPU
    #pragma acc host_data use_device(mat_a, buf, res)
    status = cublasDgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_rows_loc, n_rows, n_cols, &alpha, mat_a, n_rows_loc, buf, n_cols, &beta, res, n_rows_loc);
#endif
    fprintf(stdout, "%i %f c | finished Matrix computation\n", rank, (double)(clock()-start)/CLOCKS_PER_SEC);

    #pragma acc update self ( res[ 0 : res_size ] )
    fprintf(stdout, "final val: %d\n", res[0]);

    MPI_Finalize();
    return 0;
}

