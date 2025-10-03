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


void print_loc( const double * mat, int n_row, int n_col){
    for( int i = 0; i < n_row; i++ ){
        for ( int j = 0; j < n_col; j++) {
            fprintf( stdout, "%.6g ", mat[i*n_col+j] );
        }
        fprintf( stdout, "\n" );
    }
}

void print_par( const double * mat, int size, int rank, int npes, int flipped){
    MPI_Barrier( MPI_COMM_WORLD );
    if( rank )
        MPI_Send( mat, size*(size/npes), MPI_DOUBLE, 0, rank, MPI_COMM_WORLD );
    else{
        double * buf = (double *) calloc( size*(size/npes), sizeof(double) );
        if (flipped) {print_loc( mat, size, size / npes );}
        else {print_loc( mat, size / npes, size );}


        for( int count = 1; count < npes; count ++){
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
    long int n_cols, size_mata, size_matb, n_chunks;
    long int n_rows, n_rows_loc, matb_sendcount;
    long int current_col;
    long int i, j;
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
    int ngpu = acc_get_num_devices(acc_device_nvidia);
    int igpu = rank % ngpu;
    acc_set_device_num(igpu, acc_device_nvidia);
    acc_init(acc_device_nvidia);
    cublasHandle_t handle;
    cublasCreate(&handle);
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
    for ( long int current_chunk = 0; current_chunk < n_chunks; current_chunk++ ) {
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
        #pragma acc parallel loop collapse(2) present( mat_a, res, buf)
        for (int i = 0; i < n_rows_loc; ++i) {
            for (int j = 0; j < n_rows_loc; ++j) {
                for (int k = 0; k < n_cols; ++k) {
                    res[i * n_cols + (current_col + j)] += mat_a[i * n_cols + k] * buf[k * n_rows_loc + j];
                }
            }
        }
#endif
#ifdef USE_BLAS
        #pragma acc host_data use_device(mat_a, buf, res)
        cblas_dgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, n_rows_loc, n_rows_loc, n_cols, alpha, mat_a, n_cols, buf, n_rows_loc, beta, &res[current_col], n_cols);
#endif
#ifdef USE_GPU
        // I'm using the trick outlined here to make cublas work with row-major:
        // https://leimao.github.io/blog/cuBLAS-Transpose-Column-Major-Relationship/
        #pragma acc host_data use_device(mat_a, buf, res)
        {
            cublasStatus_t stat = cublasDgemm(
                handle,
                CUBLAS_OP_N,
                CUBLAS_OP_N,
                n_rows_loc,
                n_rows_loc,
                n_cols,
                &alpha,
                buf,
                n_rows_loc,
                mat_a,
                n_cols,
                &beta,
                &res[current_col],
                n_cols
            );
            if (stat != CUBLAS_STATUS_SUCCESS) {
                fprintf(stderr, "cublasDgemm failed with status %d\n", (int) stat);
                MPI_Abort(MPI_COMM_WORLD, -1);
            }
        }
#endif
        fprintf(stdout, "%i %f c | finished Matrix computation\n", rank, (double)(clock()-start)/CLOCKS_PER_SEC);
    }
    #pragma acc exit data copyout( res[ 0 : size_mata ] ) finalize
#ifndef NDEBUG
    print_par( res, n_cols, rank, npes, 0);
#endif
    MPI_Finalize();
    // Required so that the compiler doesn't optimize out the res array...
    fprintf(stdout, "%i %f s | %f algorithm complete\n", rank, (double)(clock()-start)/CLOCKS_PER_SEC, res[0]);
    free(mat_a); free(mat_b); free(buf); free(res);
    return 0;
}

