//
// Created by chromeilion on 10/5/24.
//
#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <openacc.h>


void print_loc( double * mat, int n_row, int n_col, bool skip, FILE *output){
    int start;
    if(skip) {
        start = 1;
    } else {
        start = 0;
    }
    for(int i = start; i < n_row; i++ ){
        for ( int j = 0; j < n_col; j++) {
            fprintf(output, "%.6g ", mat[i*n_col+j]);
        }
        fprintf(output, "\n");
    }
}

void print_par(double * mat, int n_rows, int n_cols, int rank, int npes, FILE *output) {
    int count;
    if(rank)
        MPI_Send(mat, n_cols*n_rows, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    else{
        double * buf = (double *) calloc(n_cols*n_rows, sizeof(double));
        if (npes == 1) {
            print_loc(mat, n_rows, n_cols, 0, output);
        } else {
            print_loc(mat, n_rows-1, n_cols, 0, output);
        }
        for( count = 1; count < npes; count ++){
            MPI_Recv( buf, n_cols*n_rows, MPI_DOUBLE, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            if (count == npes - 1) {
                print_loc(buf, n_rows, n_cols, 1, output);
                continue;
            }
            print_loc(buf, n_rows-1, n_cols, 1, output);
        }
        free( buf );
        fprintf( stdout, "\n" );
        fflush(stdout);
    }
}


int main( int argc, char * argv[] ) {
    int size;
    // Regular MPI WORLD communicator
    int npes, rank;

    // Timer
    double start;

    double *mat, *mat_new;
    int i, j;

    // For communicating with the process above and below me.
    MPI_Comm top_com, bot_com;

    // Jacobi iter vars
    double error, tol;
    int iter, max_iter;

    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &rank);


    if (argc < 4) {
        fprintf(stdout, "The size of the matrix and max iters needs to be "
                        "provided...\n");
        MPI_Finalize();
        return 1;
    }
    size = atoi(argv[1]);
    if (size % npes != 0) {
        fprintf(stdout, "The size of the matrix plus 2 is not divisible by the "
                        "number of processors!!!\n");
        MPI_Finalize();
        return 1;
    }
    max_iter = atoi(argv[2]);
    sscanf(argv[3], "%lf", &tol);

    bool save_to_file;
    if (argc == 5) {
        save_to_file = atoi(argv[4]);
    }

    #ifdef USE_GPU
    int ngpu = acc_get_num_devices(acc_device_nvidia);
    if (ngpu == 0) {
        fprintf(stdout, "No GPU's found!!!");
        return 1;
    }
    acc_set_device_num(rank % ngpu, acc_device_nvidia);
    acc_init(acc_device_nvidia);
    #endif

    // Add 2 because of the static values on the edges
    int n_cols_global = size+2;
    int n_rows_global = size+2;
    int i_col_max = n_cols_global-1;
    int i_col_min = 1;
    int top_color = rank+1;
    int bot_color = rank;
    int n_rows_loc = size / npes + 2;
    int n_rows_inner = size / npes;
    int i_row_max = n_rows_loc - 1;
    int i_row_min = 1;

    fprintf(stdout, "%i 0 s | detected %i processes and %i rows per proc. "
                    "i_row_min/max are %i %i and col vals are %i %i, "
                    "n_cols_global is %i\n",
            rank, npes, n_rows_loc, i_row_min, i_row_max, i_col_min, i_col_max,
            n_cols_global);

    MPI_Comm_split(MPI_COMM_WORLD, top_color, rank, &top_com);
    MPI_Comm_split(MPI_COMM_WORLD, bot_color, rank, &bot_com);

    int mat_size = n_cols_global*n_rows_loc;

    // Wait for everyone to be ready for better timing.
    MPI_Barrier(MPI_COMM_WORLD);

    fprintf(stdout, "%i 0 s | finished initialization\n", rank);
    start = clock();


    mat = (double *)calloc(mat_size, sizeof(double));
    mat_new = (double *)calloc(mat_size, sizeof(double));

    // Switch to the accelerator
    #pragma acc enter data create(mat[0:mat_size], mat_new[0:mat_size])

    fprintf(stdout, "%i %f s | initialized local matrix memory\n",
            rank, (double)(clock()-start)/CLOCKS_PER_SEC);

    #pragma acc parallel loop collapse(2) present(mat)
    for (i = i_row_min; i < i_row_max; i++) {
        for ( j = i_col_min; j < i_col_max; j++) {
            mat[i*n_cols_global+j] = 0.5;
        }
    }
    double scale_factor = 100/n_rows_global;
    int global_start_idx = rank*n_rows_inner+1;

    fprintf(stdout, "%i %f s | start idx is %i with %i processes and %i rows "
                    "per process\n",
            rank, (double)(clock()-start)/CLOCKS_PER_SEC, global_start_idx, npes,
            n_rows_inner);

    if (rank == npes-1) {
        // If we're the bottom process, we need to fill in the bottom section.
        #pragma acc parallel loop collapse(1) present(mat)
        for (i = 0; i < n_cols_global; i++) {
            mat[(n_rows_loc-1)*n_cols_global + i] = scale_factor * (n_cols_global - i);
        }
    }
    #pragma acc parallel loop collapse(1) present(mat)
    for (i = 0; i < n_rows_loc - 1; i++) {
        double val = scale_factor * (global_start_idx + i);
        mat[i*n_cols_global] = val;
    }
    #ifndef NDEBUG
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    #pragma acc update self(mat[0:mat_size])
    print_par(mat, n_rows_loc, n_cols_global, rank, npes, stdout);
    fflush(stdout);
    #endif

    fprintf(stdout, "%i %f s | initialized local matrix values\n",
            rank, (double)(clock()-start)/CLOCKS_PER_SEC);

    error = INFINITY;
    iter = 0;
    while ( error > tol && iter < max_iter ) {
        // Jacobi iteration using MPI on the border values
        #pragma acc host_data use_device(mat, mat_new)
        MPI_Allreduce(&mat[i_row_min*n_cols_global],
                      &mat_new[i_row_min*n_cols_global],
                      n_cols_global, MPI_DOUBLE,
                      MPI_SUM, top_com);

        #pragma acc host_data use_device(mat, mat_new)
        MPI_Allreduce(&mat[(i_row_max-1)*n_cols_global],
                      &mat_new[(i_row_max-1)*n_cols_global], n_cols_global,
                      MPI_DOUBLE, MPI_SUM, bot_com);
        fprintf(stdout, "%i %f m | mpi allreduce complete\n",
                rank, (double)(clock()-start)/CLOCKS_PER_SEC);
        // Regular Jacobi iteration loop
        error = 0;
        #pragma acc parallel loop collapse(2) present(mat, mat_new) reduction(max:error)
        for (i = i_row_min; i < i_row_max; i++) {
            for (j = i_col_min; j < i_col_max; j++) {
                double right = mat[i*n_cols_global + j + 1];
                double left = mat[i*n_cols_global + j-1];
                double up = mat[(i-1)*n_cols_global + j];
                double down = mat[(i+1)*n_cols_global + j];
                mat_new[i*n_cols_global + j] = 0.25 * (right + left + up + down);
                error = fmax(error, fabs(mat_new[i*n_cols_global + j] - mat[i*n_cols_global + j]));
            }
        }
        #pragma acc parallel loop collapse(2) present(mat, mat_new)
        for (i = i_row_min; i < i_row_max; i++) {
            for (j = i_col_min; j < i_col_max; j++) {
                mat[i*n_cols_global + j] = mat_new[i*n_cols_global + j];
            }
        }
        fprintf(stdout, "%i %f c | Jacobi iteration complete\n",
                rank, (double)(clock()-start)/CLOCKS_PER_SEC);
        iter += 1;
    }

    fprintf(stdout, "%i %f s | done calculating Jacobi with an error of %f\n",
            rank, (double)(clock()-start)/CLOCKS_PER_SEC, error);

    #pragma acc exit data copyout(mat[0:mat_size]) delete(mat_new) finalize

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("\n");
    FILE *out;
    if (save_to_file) {
        out = fopen( "jacobi_output.txt", "w" );
    } else {
        out = stdout;
    }
    print_par(mat, n_rows_loc, n_cols_global, rank, npes, out);
    fclose(out);
    MPI_Finalize();
    free(mat); free(mat_new);
    return 0;
}