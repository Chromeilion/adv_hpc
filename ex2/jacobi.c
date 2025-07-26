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
#ifdef USE_GPU
#include <cublas_v2.h>
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
            fprintf(output, "%.6g ", mat[i*n_col+j]);
        }
        fprintf(output, "\n");
    }
}

void print_par(const float * mat, const int n_rows, const int n_cols,
               const int rank, const int npes, FILE *output) {
    int count;
    if(rank)
        MPI_Send(mat, n_cols*n_rows, MPI_DOUBLE, 0, rank, MPI_COMM_WORLD);
    else{
        float * buf = (float *) calloc(n_cols*n_rows, sizeof(float));
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

void swap_ptrs(float** a, float** b) {
    float* temp = *a;
    *a = *b;
    *b = temp;
}

// Dynamically adds the correct boarder pixels
float im2col_get_pixel(float *im, int height, int width, int channels,
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
    return im[col + width*row];
}

//From Berkeley Vision's Caffe!
//https://github.com/BVLC/caffe/blob/master/LICENSE
void im2col(float* data_im, int channels,  int height,  int width,
            int ksize,  int stride, int pad, float* data_col,
            float *topfill, float *botfill, float *leftfill)
{
    int c,h,w;
    int height_col = (height + 2*pad - ksize) / stride + 1;
    int width_col = (width + 2*pad - ksize) / stride + 1;

    int channels_col = channels * ksize * ksize;
    for (c = 0; c < channels_col; ++c) {
        int w_offset = c % ksize;
        int h_offset = (c / ksize) % ksize;
        int c_im = c / ksize / ksize;
        for (h = 0; h < height_col; ++h) {
            for (w = 0; w < width_col; ++w) {
                int im_row = h_offset + h * stride;
                int im_col = w_offset + w * stride;
                int col_index = (c * height_col + h) * width_col + w;
                float val = im2col_get_pixel(
                        data_im, height, width, channels, im_row, im_col, c_im,
                        pad, topfill, botfill, leftfill);
                data_col[col_index] = val;
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

#ifdef USE_GPU
void convolve(float *mat, float *kernel, float *colbuf, int channels, int n_rows,
                int n_cols, int ksize, int stride, int pad, cublasHandle_t handle) {
    // Prepare input image for GEMM
    im2col(mat, channels, n_rows, n_cols, ksize, stride, pad, colbuf);
    status = cublasSgemm(handle, CUBLAS_OP_N, CUBLAS_OP_N, n_rows, ksize, alpha, colbuf, n_rows, kernel, kstride, beta, new_mat, n_rows);
}
#endif
#ifdef USE_BLAS
void convolve(float *mat, float *res, float *kernel, float *colbuf, int channels,
              int n_rows, int n_cols, int colbuf_cols, int colbuf_rows,
              int ksize, int stride, int pad, float alpha, float beta,
              float *topfill, float *botfill, float *leftfill) {
    int kcol = 1;
    // Prepare input image for GEMM
    im2col(mat, channels, n_rows, n_cols, ksize, stride, pad, colbuf,
           topfill, botfill, leftfill);
    cblas_sgemm(CblasRowMajor, CblasNoTrans, CblasNoTrans, colbuf_rows, kcol,
                colbuf_cols, alpha, colbuf, colbuf_cols, kernel, kcol, beta, res,
                1);
}
#endif


int main( int argc, char * argv[] ) {
    int size;
    int npes, rank;

    double start_time;

    float *mat, *mat_new, *colbuf, *colbuf_s, *botpad, *toppad, *leftpad;
    int i;

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

#ifdef USE_GPU
    cublasHandle_t handle;
    cublasStatus_t status = cublasCreate(&handle);
    int ngpu = acc_get_num_devices(acc_device_nvidia);
    if (ngpu == 0) {
        fprintf(stdout, "No GPU's found!!!");
        return 1;
    }
    acc_set_device_num(rank % ngpu, acc_device_nvidia);
    acc_init(acc_device_nvidia);
#endif

    // Add 2 because of the static values on the edges
    const int n_cols_global = size;
    const int n_rows_global = size;
    const int top_color = rank + 1;
    const int bot_color = rank;
    const int n_rows_loc = size / npes;

    // Convolution parameters
    const int ksize = 3;
    const int stride = 1;
    const int pad = 1;
    const int channels = 1;
    const float alpha = 1.0;
    const float beta = 0.0;

    fprintf(stdout, "%i 0 s | detected %i processes and %i rows per proc. "
                    "n_cols_global is %i\n",
            rank, npes, n_rows_loc, n_cols_global);

    MPI_Comm_split(MPI_COMM_WORLD, top_color, rank, &top_com);
    MPI_Comm_split(MPI_COMM_WORLD, bot_color, rank, &bot_com);

    const int mat_size = n_cols_global * n_rows_loc;
    const int height_col_l = ((n_rows_loc - 2) - ksize) / stride + 1 + 2 * pad;
    const int width_col = (n_cols_global - ksize) / stride + 1 + 2 * pad;
    const int buf_rows_l = height_col_l * width_col;
    const int buf_cols_l = ksize * ksize;
    // We take out two from the height because those are done separately
    // e.g., we run calculations on the top/bottom rows during a different step
    int colbuf_size = buf_cols_l * buf_rows_l;
    int colbuf_s_size = n_cols_global * ksize * ksize;

    MPI_Barrier(MPI_COMM_WORLD);

    fprintf(stdout, "%i 0 s | finished initialization\n", rank);
    start_time = clock();

    mat = (float *) calloc(mat_size, sizeof(float));
    mat_new = (float *) calloc(mat_size, sizeof(float));
    colbuf = (float *) calloc(colbuf_size, sizeof(float));
    colbuf_s = (float *) calloc(colbuf_s_size, sizeof(float));
    toppad =  (float *) calloc(n_cols_global, sizeof(float));
    botpad =  (float *) calloc(n_cols_global, sizeof(float));
    leftpad =  (float *) calloc(n_cols_global, sizeof(float));
    float kernel[] = {0, 0.25, 0, 0.25, 0, 0.25, 0, 0.25, 0};

    const float scale_factor = (float) 100 / ((float) n_rows_global);
    const int global_top_row = n_rows_loc * rank + 1;

    // Switch to the accelerator
    #pragma acc enter data create(mat[0:mat_size], mat_new[0:mat_size], colbuf[0:colbuf_size], kernel[0:ksize*ksize], toppad[0:n_cols_global], botpad[0:n_cols_global], colbuf_s[0:colbuf_s_size]) leftpad[0:n_cols_global])

    fprintf(stdout, "%i %f s | initialized local matrix memory\n",
            rank, (double) (clock() - start_time) / CLOCKS_PER_SEC);

    fill_matr(mat, n_rows_loc, n_cols_global, 0.5);
    fill_matr(toppad, 1, n_cols_global, 0);
    if (rank == npes-1) {
        // If we're the bottom process, we need to fill in the bottom section.
        #pragma acc parallel loop present(botpad)
        for (i = 0; i < n_cols_global; i++) {
            botpad[i] = (float)100 - scale_factor * (float)i;
        }
    } else {
        fill_matr(botpad, 1, n_cols_global, 0);
    }
    #pragma acc parallel loop collapse(1) present(leftpad)
    for (i = 0; i < n_rows_loc; i++) {
        leftpad[i] = scale_factor * ((float)i+global_top_row);
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
    swap_ptrs(&mat, &mat_new);
    while (iter < max_iter) {
        swap_ptrs(&mat, &mat_new);
        // Jacobi iteration using MPI on the border values
        #pragma acc host_data use_device(mat, mat_new)
        MPI_Iallreduce(
                mat,
                mat_new,
                n_cols_global, MPI_DOUBLE,
                MPI_SUM, top_com, &top_request);

        #pragma acc host_data use_device(mat, mat_new)
        MPI_Iallreduce(
                &mat[(n_rows_loc - 1) * n_cols_global],
                &mat_new[(n_rows_loc - 1) * n_cols_global],
                n_cols_global,
                MPI_DOUBLE, MPI_SUM, bot_com, &bottom_request);

        // Do our Jacobi iteration step
        convolve(&mat[n_cols_global], &mat_new[n_cols_global], kernel,
                 colbuf, channels, n_rows_loc-2, n_cols_global, buf_cols_l,
                 buf_rows_l, ksize, stride, pad, alpha, beta, mat,
                 &mat[(n_rows_loc-1)*n_cols_global], &leftpad[1]);

        // Wait for our Allreduce's to complete
        MPI_Wait(&top_request, MPI_STATUS_IGNORE);
        MPI_Wait(&bottom_request, MPI_STATUS_IGNORE);

        convolve(mat, mat_new, kernel,
                 colbuf_s, channels, 1, n_cols_global, ksize * ksize,
                 n_cols_global, ksize, stride, pad, alpha, beta,
                 toppad, &mat[n_cols_global], leftpad);

        convolve(&mat[n_cols_global*(n_rows_loc-1)],
                 &mat_new[n_cols_global*(n_rows_loc-1)], kernel,
                 colbuf_s, channels, 1, n_cols_global, ksize*ksize, n_cols_global,
                 ksize, stride, pad, alpha, beta, &mat[n_cols_global*(n_rows_loc-2)],
                 botpad, &leftpad[n_rows_loc-1]);

        iter += 1;
    }
    fprintf(stdout, "%i %f s | done calculating Jacobi\n", rank, (double)(clock()-start_time)/CLOCKS_PER_SEC);

    #pragma acc exit data copyout(mat[0:mat_size]) delete(mat_new) delete(toppad) delete(botpad) delete(colbuf) finalize

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
//    free(mat); free(mat_new); free(colbuf); free(colbuf_s); free(toppad);
//    free(botpad);
    return 0;
}