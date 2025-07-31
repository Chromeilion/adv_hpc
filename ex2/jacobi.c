//
// Created by chromeilion on 10/5/24.
//
#define KERNEL_WIDTH 3
#define KERNEL_HEIGHT 3
#define KERNEL_SIZE (KERNEL_WIDTH * KERNEL_HEIGHT)
#define KERNEL {0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f, 0.25f, 0.0f}

#include <mpi.h>
#include <stdio.h>
#include <stdlib.h>
#include <time.h>
#include <stdbool.h>
#include <math.h>
#include <openacc.h>
#ifdef USE_GPU
#include <npp.h>
#endif
#ifdef USE_BLAS
#include <cblas.h>
#endif

void print_loc(const float * mat, const int n_row, const int n_col,
               const bool skip, FILE *output){
    int start;
    if(skip) {
        start = 1;
    } else {
        start = 0;
    }
    for(int i = start; i < n_row; i++ ){
        for ( int j = 0; j < n_col; j++) {
            fprintf(output, "%.6f ", mat[i*n_col+j]);
        }
        fprintf(output, "\n");
    }
}

void print_par(const float * mat, const int n_rows, const int n_cols,
               const int rank, const int npes, FILE *output) {
    int count;
    if(rank)
        MPI_Send(mat, n_cols*n_rows, MPI_FLOAT, 0, rank, MPI_COMM_WORLD);
    else{
        float * buf = (float *) calloc(n_cols*n_rows, sizeof(float));
        if (npes == 1) {
            print_loc(mat, n_rows, n_cols, 0, output);
        } else {
            print_loc(mat, n_rows-1, n_cols, 0, output);
        }
        for( count = 1; count < npes; count ++){
            MPI_Recv( buf, n_cols*n_rows, MPI_FLOAT, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
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

void swap_ptrs(float** a, float** b) {
    float* temp = *a;
    *a = *b;
    *b = temp;
}

// Dynamically adds the correct boarder pixels
float im2col_get_pixel(float *im, int height, int width,
                       int row, int col, int channel, int pad,
                       float *topfill, float *botfill, float *leftfill)
{
    row -= pad;
    col -= pad;
    if (row < 0 && col < 0) return 0;
    if (row < 0 && col >= width) return 0;
    if (row >= height && col < 0) return 0;
    if (row >= height && col >= width) return 0;
    if (col < 0 && row < height && row >= 0) {
        return leftfill[row];
    }
    if (col < width && row >= height && col >= 0) {
        return botfill[col];
    }
    if (row < 0 && col < width && col >= 0) {
        return topfill[col];
    }
    if (col >= width) return 0;
    return im[col + width*(row + height*channel)];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col(float* data_im, int height,  int width,
            int ksize,  int stride, int pad, float* data_col,
            float *topfill, float *botfill, float *leftfill)
{
    int c,h,w, w_offset, h_offset, c_im, im_row, im_col, col_index;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        w_offset = c % ksize;
        h_offset = (c / ksize) % ksize;
        c_im = c / ksize / ksize;
#pragma acc parallel loop collapse(2) independent present(data_im, data_col, \
    topfill, botfill, leftfill)
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                im_row = h_offset + h * stride;
                im_col = w_offset + w * stride;
                col_index = (c * height_col + h) * width_col + w;
                data_col[col_index] = im2col_get_pixel(
                        data_im, height, width, im_row, im_col,
                        c_im, pad, topfill, botfill, leftfill);
            }
        }
    }
}

void fill_matr(float *mat, int rows, int cols, float val) {
    #pragma acc parallel loop collapse(2) present(mat)
    for (int i = 0; i < rows; i++) {
        for (int j = 0; j < cols; j++) {
            mat[i*cols+j] = val;
        }
    }
}

inline void do_j_op(float *mat_new, float *mat, const int i, const int j,
                    const int n_cols_global) {
    mat_new[i*n_cols_global + j] =
            0.25 * (mat[i*n_cols_global + j + 1] +
                    mat[i*n_cols_global + j-1] +
                    mat[(i-1)*n_cols_global + j] +
                    mat[(i+1)*n_cols_global + j]);
}

int main( int argc, char * argv[] ) {
    int size;
    int npes, rank;

    double start_time;

    float *mat, *mat_new;
    int i, j;

    // For communicating with the process above and below me.
    MPI_Comm top_com, bot_com;
    MPI_Request top_request, bottom_request;

    // Jacobi iter vars
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

    bool save_to_file;
    if (argc == 5) {
        save_to_file = atoi(argv[4]);
    }

    int ngpu = acc_get_num_devices(acc_device_nvidia);
    if (ngpu == 0) {
        fprintf(stdout, "No GPU's found!!!");
        return 1;
    }
    acc_set_device_num(rank % ngpu, acc_device_nvidia);
    acc_init(acc_device_nvidia);
    NppStatus npp_status;

    // Add 2 because of the static values on the edges
    const int n_cols_global = size + 2;
    const int n_rows_global = size + 2;
    const int top_color = rank + 1;
    const int bot_color = rank;
    const int n_rows_loc = size / npes + 2;

    // Convolution parameters
    const int ksize = 3;

    fprintf(stdout, "%i 0 s | detected %i processes and %i rows per proc. "
                    "n_cols_global is %i\n",
            rank, npes, n_rows_loc, n_cols_global);

    MPI_Comm_split(MPI_COMM_WORLD, top_color, rank, &top_com);
    MPI_Comm_split(MPI_COMM_WORLD, bot_color, rank, &bot_com);

    const int mat_size = n_cols_global * n_rows_loc;

    MPI_Barrier(MPI_COMM_WORLD);

    fprintf(stdout, "%i 0 s | finished initialization\n", rank);
    start_time = clock();

    mat = (float *) calloc(mat_size, sizeof(float));
    mat_new = (float *) calloc(mat_size, sizeof(float));
    float kernel[KERNEL_SIZE] = KERNEL;

    NppiSize oSize = {n_cols_global-2, n_rows_loc-4};
    NppiSize k_size = {3, 3};
    NppiPoint anchor = {1, 1};
    const float scale_factor = (float) 100 / (float)n_rows_global;
    const int global_top_row = n_rows_loc * rank + 1;

    // Switch to the accelerator
    #pragma acc enter data create(mat[0:mat_size], mat_new[0:mat_size]), copyin(kernel[0:ksize*ksize])

    fprintf(stdout, "%i %f s | initialized local matrix memory\n",
            rank, (double) (clock() - start_time) / CLOCKS_PER_SEC);

    if (rank == npes-1) {
        // If we're the bottom process, we need to fill in the bottom section.
        #pragma acc parallel loop collapse(1) present(mat)
        for (i = 0; i < n_cols_global; i++) {
            mat[(n_rows_loc-1)*n_cols_global + i] = (float)100 - scale_factor * (float)i;
            mat_new[(n_rows_loc-1)*n_cols_global + i] = (float)100 - scale_factor * (float)i;
        }
    }
    #pragma acc parallel loop collapse(1) present(mat)
    for (i = 0; i < n_rows_loc - 1; i++) {
        mat[i*n_cols_global] = scale_factor * ((float)i+global_top_row);
        mat_new[i*n_cols_global] = scale_factor * ((float)i+global_top_row);
    }

    fprintf(stdout, "%i %f s | start idx is %i with %i processes and %i rows "
                    "per process\n",
            rank, (double) (clock() - start_time) / CLOCKS_PER_SEC,
            global_top_row, npes,
            n_rows_loc);

#ifndef NDEBUG
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    #pragma acc update self(mat[0:mat_size])
    print_par(mat, n_rows_loc, n_cols_global, rank, npes, stdout);
    fflush(stdout);
#endif

    fprintf(stdout, "%i %f s | initialized local matrix values\n",
            rank, (double) (clock() - start_time) / CLOCKS_PER_SEC);

    iter = 0;
    while (iter < max_iter) {
        #pragma acc host_data use_device(mat, mat_new)
        MPI_Iallreduce(
                &mat[n_cols_global],
                &mat_new[n_cols_global],
                n_cols_global, MPI_FLOAT,
                MPI_SUM, top_com, &top_request);

        #pragma acc host_data use_device(mat, mat_new)
        MPI_Iallreduce(
                &mat[(n_rows_loc - 2) * n_cols_global],
                &mat_new[(n_rows_loc - 2) * n_cols_global],
                n_cols_global,
                MPI_FLOAT, MPI_SUM, bot_com, &bottom_request);

        #pragma acc host_data use_device(mat, mat_new, kernel)
        npp_status = nppiFilter_32f_C1R(&mat[(n_cols_global*1)],
                           sizeof(float)*n_cols_global,
                           &mat_new[(n_cols_global*2)+1],
                           sizeof(float)*n_cols_global, oSize, kernel,
                           k_size, anchor);

        if (npp_status != NPP_SUCCESS) {
            fprintf(stderr, "Rank %d encountered NPP Error: %d\n", rank, npp_status);
        }

        // Wait for our Allreduce's to complete
        MPI_Wait(&top_request, MPI_STATUS_IGNORE);
        MPI_Wait(&bottom_request, MPI_STATUS_IGNORE);

        #pragma acc wait

        // Update the top and bottom rows since we have what we need now
        #pragma acc parallel loop independent present(mat, mat_new)
        for (j = 1; j < n_cols_global-1; j++) {
            do_j_op(mat_new, mat, 1, j, n_cols_global);
            do_j_op(mat_new, mat, n_rows_loc-2, j, n_cols_global);
        }
        swap_ptrs(&mat, &mat_new);
        iter += 1;
    }
    fprintf(stdout, "%i %f s | done calculating Jacobi\n", rank, (double)(clock()-start_time)/CLOCKS_PER_SEC);

    #pragma acc exit data copyout(mat[0:mat_size]) finalize

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