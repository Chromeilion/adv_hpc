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
#include <nccl.h>
#include <cuda_runtime.h>
#endif

#ifdef USE_GPU
struct ComParams {
    ncclUniqueId id;
    int rank;
    int npes;
    int top_target;
    int bot_target;
    ncclComm_t comm;
    MPI_Request top_request;
    MPI_Request bottom_request;
};
#else
struct ComParams {
    int rank;
    int npes;
    int top_target;
    int bot_target;
    MPI_Request top_request;
    MPI_Request bottom_request;
};
#endif

struct ArrayParams {
    int n_rows_loc;
    int n_rows_inner;
    int i_row_max;
    int i_row_min;
    int i_col_max;
    int i_col_min;
    int n_cols_global;
    int global_top_row;
    int n_rows_global;
    int mat_size;
    float scale_factor;
};

#ifdef USE_GPU
struct GpuParams {
    int ngpu;
    cudaGraph_t graph_1;
    cudaGraph_t graph_2;
    cudaGraphExec_t graph_exec_1;
    cudaGraphExec_t graph_exec_2;
    cudaStream_t stream_comm;
    cudaStream_t stream_comp;
    cudaEvent_t comm_done;
    cudaEvent_t comm_done_2;
    int acc_stream_comm;
    int acc_stream_comp;
};

struct GpuParams setup_gpu(const struct ComParams *mpi_params) {
    struct GpuParams params;
    params.ngpu = acc_get_num_devices(acc_device_nvidia);
    if (params.ngpu == 0) {
        fprintf(stdout, "No GPU's found!!!");
        return params;
    }
    acc_set_device_num(mpi_params->rank % params.ngpu, acc_device_nvidia);
    acc_init(acc_device_nvidia);
    cudaStreamCreateWithFlags(&params.stream_comm, cudaStreamNonBlocking);
    cudaStreamCreateWithFlags(&params.stream_comp, cudaStreamNonBlocking);
    cudaEventCreateWithFlags(&params.comm_done, cudaEventDisableTiming);
    params.acc_stream_comp = 1;
    params.acc_stream_comm = 2;
    acc_set_cuda_stream(params.acc_stream_comm, params.stream_comm);
    acc_set_cuda_stream(params.acc_stream_comp, params.stream_comp);
    return params;
}
#else
struct GpuParams {
    int ngpu;
};
#endif

struct ArrayParams setup_array(const int size,
        const struct ComParams *mpi_params) {
    struct ArrayParams params;
    // Add 2 because of the static values on the edges
    params.n_cols_global = size+2;
    params.n_rows_global = size+2;
    params.i_col_max = params.n_cols_global-1;
    params.i_col_min = 1;
    params.n_rows_loc = size / mpi_params->npes + 2;
    params.n_rows_inner = size / mpi_params->npes;
    params.i_row_max = params.n_rows_loc - 1;
    params.i_row_min = 1;
    params.mat_size = params.n_cols_global*params.n_rows_loc;
    params.scale_factor = (float)100/((float)params.n_rows_global);
    params.global_top_row = params.n_rows_inner*mpi_params->rank+1;
    return params;
}

struct ComParams setup_mpi(int argc, char *argv[]) {
    struct ComParams params;
    MPI_Init(&argc, &argv);
    MPI_Comm_size(MPI_COMM_WORLD, &params.npes);
    MPI_Comm_rank(MPI_COMM_WORLD, &params.rank);
#ifdef USE_GPU
    if (params.rank == 0) ncclGetUniqueId(&params.id);
    MPI_Bcast(&params.id, sizeof(params.id), MPI_BYTE, 0, MPI_COMM_WORLD);
    ncclCommInitRank(&params.comm, params.npes, params.id, params.rank);
#else
    int top_target = params.rank + 1;
    int bot_target = params.rank - 1;
    if (top_target >= params.npes) {
        params.top_target = MPI_PROC_NULL;
    } else {
        params.top_target = top_target;
    }
    if (bot_target < 0) {
        params.bot_target = MPI_PROC_NULL;
    } else {
        params.bot_target = bot_target;
    }
#endif
    return params;
}

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
               const struct ComParams *mpi_params, FILE *output) {
    int count;
    if(mpi_params->rank)
        MPI_Send(mat, n_cols*n_rows, MPI_FLOAT, 0, mpi_params->rank, MPI_COMM_WORLD);
    else{
        float * buf = (float *) calloc(n_cols*n_rows, sizeof(float));
        if (mpi_params->npes == 1) {
            print_loc(mat, n_rows, n_cols, 0, output);
        } else {
            print_loc(mat, n_rows-1, n_cols, 0, output);
        }
        for( count = 1; count < mpi_params->npes; count ++){
            MPI_Recv( buf, n_cols*n_rows, MPI_FLOAT, count, count, MPI_COMM_WORLD, MPI_STATUS_IGNORE );
            if (count == mpi_params->npes - 1) {
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

#pragma acc routine seq
void do_j_op(float *mat_new, float *mat, float* error, const int i,
                    const int j, const struct ArrayParams *array_params) {
    mat_new[i*array_params->n_cols_global + j] =
            0.25 * (mat[i*array_params->n_cols_global + j + 1] +
                    mat[i*array_params->n_cols_global + j-1] +
                    mat[(i-1)*array_params->n_cols_global + j] +
                    mat[(i+1)*array_params->n_cols_global + j]);
    *error = fmax(*error, fabs(mat_new[i*array_params->n_cols_global + j] -
    mat[i*array_params->n_cols_global + j]));
}

void jacobi_loop_first_half(float *mat, float *mat_new, float *error,
                            const struct ArrayParams *array_params) {
    #pragma acc serial present(error) async(1)
    *error = INFINITY;

    #pragma acc parallel loop independent collapse(2) present(mat, mat_new, error, array_params) reduction(max:error[0:1]) async(1)
    for (int i = array_params->i_row_min + 1; i < array_params->i_row_max - 1; i++) {
        for (int j = array_params->i_col_min; j < array_params->i_col_max; j++) {
            do_j_op(mat_new, mat, error, i, j, array_params);
        }
    }
}

void jacobi_loop_second_half(float *mat, float *mat_new, float *error,
                                    const struct ArrayParams *array_params) {
    #pragma acc parallel loop independent present(mat, mat_new, error, array_params) reduction(max:error[0:1]) async(1)
    for (int j = array_params->i_col_min; j < array_params->i_col_max; j++) {
        do_j_op(mat_new, mat, error, array_params->i_row_min, j, array_params);
    }
    #pragma acc parallel loop independent present(mat, mat_new, error, array_params) reduction(max:error[0:1]) async(1)
    for (int j = array_params->i_col_min; j < array_params->i_col_max; j++) {
        do_j_op(mat_new, mat, error, array_params->i_row_max-1, j, array_params);
    }
};

#ifdef USE_GPU
void send_recv(const float *sendbuf, float *recbuf, const int count,
               const int recv_from, const int target, MPI_Request *req,
               struct ComParams *com_params, struct GpuParams *gpu_params) {
    const void* d_send = acc_deviceptr((void*)sendbuf);
    void* d_recv = acc_deviceptr((void*)recbuf);
    ncclGroupStart();
    if (target != MPI_PROC_NULL && recv_from != MPI_PROC_NULL) {
        ncclSend(d_send, count, ncclFloat, target, com_params->comm,
                 gpu_params->stream_comm);
        ncclRecv(d_recv, count, ncclFloat, recv_from, com_params->comm,
                 gpu_params->stream_comm);
    } else if (target != MPI_PROC_NULL && recv_from == MPI_PROC_NULL) {
        ncclSend(d_send, count, ncclFloat, target, com_params->comm,
                 gpu_params->stream_comm);
    } else if (target == MPI_PROC_NULL && recv_from != MPI_PROC_NULL) {
        ncclRecv(d_recv, count, ncclFloat, recv_from, com_params->comm,
                 gpu_params->stream_comm);
    }
    ncclGroupEnd();
}
#else
void send_recv(const float *sendbuf, float *recbuf, const int count,
               const int target, const int recv_from, MPI_Request *req,
               struct ComParams *com_params, struct GpuParams *gpu_params) {
    #pragma acc host_data use_device(sendbuf, recbuf)
    MPI_Isendrecv(sendbuf, count, MPI_FLOAT, target, 0, recbuf, count, MPI_FLOAT,
                  recv_from, MPI_ANY_TAG, MPI_COMM_WORLD, req);
}
#endif

void make_mpi_sendrecs(float *mat, float *mat_new,
                       struct ComParams *com_params,
                       const struct ArrayParams *array_params,
                       struct GpuParams *gpu_params,
                       void *comm_done) {
    float *top_recbuf = mat;
    float *top_sendbuf = &mat[array_params->i_row_min*array_params->n_cols_global];
    float *bot_recbuf = &mat[array_params->i_row_max*array_params->n_cols_global];
    float *bot_sendbuf = &mat[(array_params->i_row_max-1)*array_params->n_cols_global];
    send_recv(top_sendbuf, top_recbuf, array_params->n_cols_global,
              com_params->bot_target, com_params->bot_target,
              &com_params->bottom_request, com_params, gpu_params);
    send_recv(bot_sendbuf, bot_recbuf, array_params->n_cols_global,
              com_params->top_target, com_params->top_target,
              &com_params->top_request, com_params, gpu_params);
#ifdef USE_GPU
    cudaEventRecord((cudaEvent_t)comm_done, gpu_params->stream_comm);
#endif
}

static inline void fill_side(float *mat, float *mat_new, const struct ArrayParams *array_params) {
    #pragma acc parallel loop collapse(1) present(mat, mat_new) async(1)
    for (int i = 0; i < array_params->n_rows_loc - 1; i++) {
        mat[i*array_params->n_cols_global] = array_params->scale_factor *
                ((float)i+array_params->global_top_row);
        mat_new[i*array_params->n_cols_global] = array_params->scale_factor *
                ((float)i+array_params->global_top_row);
    }
}

static inline void fill_bottom(float *mat, float *mat_new,
                 const struct ArrayParams *array_params) {
    #pragma acc parallel loop collapse(1) present(mat, mat_new) async(1)
    for (int i = 0; i < array_params->n_cols_global; i++) {
        mat[(array_params->n_rows_loc-1)*array_params->n_cols_global + i] =
                (float)100 - array_params->scale_factor * (float)i;
        mat_new[(array_params->n_rows_loc-1)*array_params->n_cols_global + i] =
                (float)100 - array_params->scale_factor * (float)i;
    }
}

static inline void jacobi_iter(float *mat,
                               float* mat_new,
                               float *error,
                               struct ComParams *mpi_params,
                               const struct ArrayParams *array_params,
                               struct GpuParams *gpu_params,
                               void *comm_done) {
    make_mpi_sendrecs(mat, mat_new, mpi_params, array_params, gpu_params, comm_done);
    jacobi_loop_first_half(mat, mat_new, error, array_params);
#ifdef USE_GPU
    cudaStreamWaitEvent(gpu_params->stream_comp, gpu_params->comm_done, 0);
#else
    MPI_Wait(&mpi_params->top_request, MPI_STATUS_IGNORE);
    MPI_Wait(&mpi_params->bottom_request, MPI_STATUS_IGNORE);
#endif
    jacobi_loop_second_half(mat, mat_new, error, array_params);
}


int main( int argc, char * argv[] ) {
    int size;
    double start_time;
    float *mat, *mat_new, *error, *tol;
    int *iter, *max_iter;

    struct ComParams mpi_params = setup_mpi(argc, argv);

    if (argc < 4) {
        fprintf(stdout, "The size of the matrix and max iters needs to be "
                        "provided...\n");
        MPI_Finalize();
        return 1;
    }
    size = atoi(argv[1]);
    if (size % mpi_params.npes != 0) {
        fprintf(stdout, "The size of the matrix plus 2 is not divisible by the "
                        "number of processors!!!\n");
        MPI_Finalize();
        return 1;
    }
    max_iter = (int *)calloc(1, sizeof(int));
    tol = (float *)calloc(1, sizeof(float));

    *max_iter = atoi(argv[2]);
    sscanf(argv[3], "%f", tol);

    bool save_to_file = false;
    if (argc == 5) {
        save_to_file = true;
    }

    #ifdef USE_GPU
    struct GpuParams gpu_params = setup_gpu(&mpi_params);
    #else
    struct GpuParams gpu_params;
    #endif

    const struct ArrayParams array_params = setup_array(size, &mpi_params);

    fprintf(stdout, "%i 0 s | detected %i processes and %i rows per proc. "
                    "i_row_min/max are %i %i and col vals are %i %i, "
                    "n_cols_global is %i\n",
            mpi_params.rank, mpi_params.npes, array_params.n_rows_loc,
            array_params.i_row_min, array_params.i_row_max,
            array_params.i_col_min, array_params.i_col_max,
            array_params.n_cols_global);

    MPI_Barrier(MPI_COMM_WORLD);

    fprintf(stdout, "%i 0 s | finished initialization\n", mpi_params.rank);
    start_time = clock();


    mat = (float *)calloc(array_params.mat_size, sizeof(float));
    mat_new = (float *)calloc(array_params.mat_size, sizeof(float));
    error = (float *)calloc(1, sizeof(float));
    iter = (int *)calloc(1, sizeof(int));

    *iter = 0;
    *error = INFINITY;

    // Add our data to the accelerator
    #pragma acc enter data create(mat[0:array_params.mat_size], \
    mat_new[0:array_params.mat_size]) copyin(error[0:1], array_params)

    fprintf(stdout, "%i %f s | initialized local matrix memory\n",
            mpi_params.rank, (double)(clock()-start_time)/CLOCKS_PER_SEC);

    fprintf(stdout, "%i %f s | start idx is %i with %i processes and %i rows "
                    "per process\n",
            mpi_params.rank, (double)(clock()-start_time)/CLOCKS_PER_SEC,
            array_params.global_top_row, mpi_params.npes,
            array_params.n_rows_inner);

    if (mpi_params.rank == mpi_params.npes-1) {
        fill_bottom(mat, mat_new, &array_params);
    }
    fill_side(mat, mat_new, &array_params);

    // fill bottom and fill side are async, so we wait for them here
    #pragma acc wait

    #ifndef NDEBUG
    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    #pragma acc update self(mat[0:array_params.mat_size])
    print_par(mat, array_params.n_rows_loc, array_params.n_cols_global,
              &mpi_params, stdout);
    fflush(stdout);
    #endif

    fprintf(stdout, "%i %f s | initialized local matrix values\n",
            mpi_params.rank, (double)(clock()-start_time)/CLOCKS_PER_SEC);

#ifdef USE_GPU
    #pragma acc wait
    cudaStreamBeginCapture(gpu_params.stream_comm, cudaStreamCaptureModeGlobal);
    cudaStreamBeginCapture(gpu_params.stream_comp, cudaStreamCaptureModeGlobal);
    jacobi_iter(mat, mat_new, error, &mpi_params, &array_params, &gpu_params,
                &gpu_params.comm_done);
    cudaStreamEndCapture(gpu_params.stream_comm, NULL);
    cudaStreamEndCapture(gpu_params.stream_comp, &gpu_params.graph_1);
    cudaGraphInstantiate(&gpu_params.graph_exec_1, gpu_params.graph_1, 0);
    #pragma acc wait
    cudaStreamBeginCapture(gpu_params.stream_comm, cudaStreamCaptureModeGlobal);
    cudaStreamBeginCapture(gpu_params.stream_comp, cudaStreamCaptureModeGlobal);
    jacobi_iter(mat_new, mat, error, &mpi_params, &array_params, &gpu_params,
                &gpu_params.comm_done_2);
    cudaStreamEndCapture(gpu_params.stream_comm, NULL);
    cudaStreamEndCapture(gpu_params.stream_comp, &gpu_params.graph_2);
    cudaGraphInstantiate(&gpu_params.graph_exec_2, gpu_params.graph_2, 0);
    #pragma acc serial async(1)
    *iter = 2;
    #pragma acc wait
    #pragma acc update self(error[0:1])
#endif
    while (*iter < *max_iter && *error > *tol) {
        if (*iter & 1) {
            #ifdef USE_GPU
            cudaGraphLaunch(gpu_params.graph_exec_2, gpu_params.stream_comp);
            #else
            jacobi_iter(mat_new, mat, error, &mpi_params, &array_params, &gpu_params);
            #endif
        } else {
            #ifdef USE_GPU
            cudaGraphLaunch(gpu_params.graph_exec_1, gpu_params.stream_comp);
            #else
            jacobi_iter(mat, mat_new, error, &mpi_params, &array_params, &gpu_params);
            #endif
        }
        #pragma acc wait
        #pragma acc update self(error[0:1])
        *iter += 1;
    }

    fprintf(stdout, "%i %f s | done calculating Jacobi with an error of %f\n",
            mpi_params.rank, (double)(clock()-start_time)/CLOCKS_PER_SEC, *error);

    #pragma acc exit data copyout(mat[0:array_params.mat_size], error[0:1]) finalize

    fflush(stdout);
    MPI_Barrier(MPI_COMM_WORLD);
    printf("\n");
    FILE *out;
    if (save_to_file) {
        out = fopen( "jacobi_output.txt", "w" );
    } else {
        out = stdout;
    }
    print_par(mat, array_params.n_rows_loc, array_params.n_cols_global,
              &mpi_params, out);
    fclose(out);
#ifdef USE_GPU
    ncclCommDestroy(mpi_params.comm);
#endif
    MPI_Finalize();
    free(mat); free(mat_new); free(error); free(max_iter); free(tol); free(iter);
    return 0;
}