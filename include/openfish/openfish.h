#ifndef OPENFISH_H
#define OPENFISH_H

#include <stdbool.h>
#include <stdlib.h>
#include <stdint.h>

#include "openfish_error.h"

#ifdef __cplusplus
extern "C" {
#endif

#define OPENFISH_DECODER_INIT DECODER_INIT
#define DECODER_INIT {32, 100.0, 2.0, 0.0, 1.0}

typedef struct openfish_gpubuf {
    float *bwd_NTC;
    float *post_NTC;
    uint8_t *moves;
    char *sequence;
    char *qstring;
    void *beam_vector;
    void *states;
    float *qual_data;
    float *base_probs;
    float *total_probs;
} openfish_gpubuf_t;

typedef struct openfish_opt {
    size_t beam_width;
    float beam_cut;
    float blank_score;
    float q_shift;
    float q_scale;
} openfish_opt_t;

openfish_opt_t openfish_decoder_default_opts(void);

void openfish_decode_cpu(
    const int T,
    const int N,
    const int C,
    int nthreads,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

// same as openfish_decode_cpu, but scores_TNC is fp16
// scores are converted as they are read, so no fp32 copy of the tensor is made
void openfish_decode_cpu_fp16(
    const int T,
    const int N,
    const int C,
    int nthreads,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

void openfish_rotary_emb_cpu(
    void *x,
    void *sin_buf,
    void *cos_buf,
    int batch_size,
    int seqlen,
    int nheads,
    int head_dim,
    int rotary_half,
    int stride_batch,
    int stride_seq,
    int stride_head,
    int nthreads
);

void openfish_decode_gpu(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    const openfish_gpubuf_t *gpubuf,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

/* Split decode, for running the beam search on the CPU while the GPU works on the next batch.
 *
 * openfish_scan_gpu: the backward scan and the forward scan + posterior of openfish_decode_gpu, and
 * nothing else. scores_TNC is fp16 [T][N][C]; bwd_NTC and post_NTC receive N * (T + 1) * num_states
 * floats each. All three must be device-accessible; on an integrated GPU (Jetson) they can live in
 * pinned host memory from openfish_host_alloc, so the CPU then reads them without a copy.
 * Synchronises before returning. */
void openfish_scan_gpu(
    const int T,
    const int N,
    const int C,
    void *scores_TNC,
    const int state_len,
    const openfish_opt_t *options,
    float *bwd_NTC,
    float *post_NTC
);

/* openfish_scan_gpu's other half: beam search, per-base qualities and sequence on the CPU, from
 * fp16 scores and the scan outputs (same layouts as above), over nthreads threads. Output as for
 * openfish_decode_cpu (moves, sequence, qstring: N * T each, caller frees). */
void openfish_beam_cpu_fp16(
    const int T,
    const int N,
    const int C,
    int nthreads,
    const void *scores_TNC,
    const float *bwd_NTC,
    const float *post_NTC,
    const int state_len,
    const openfish_opt_t *options,
    uint8_t **moves,
    char **sequence,
    char **qstring
);

/* Pinned host memory mapped into the GPU's address space (cudaHostAllocMapped). Returns the host
 * pointer, or NULL on failure; *device_ptr receives the pointer GPU kernels use. */
void *openfish_host_alloc(size_t bytes, void **device_ptr);
void openfish_host_free(void *host_ptr);

/* 1 if the GPU shares physical memory with the CPU (Jetson), so zero-copy buffers run at full speed;
 * 0 for a discrete GPU, where every GPU access to them crosses PCIe. */
int openfish_gpu_is_integrated(void);

openfish_gpubuf_t *openfish_gpubuf_init(
    const int T,
    const int N,
    const int state_len
);

void openfish_gpubuf_free(
    openfish_gpubuf_t *gpubuf
);

void openfish_rotary_emb_gpu(
    void *x_gpu,
    void *sin_gpu,
    void *cos_gpu,
    int batch_size,
    int seqlen,
    int nheads,
    int head_dim,
    int rotary_half,
    int stride_batch,
    int stride_seq,
    int stride_head
);

void openfish_silu_mul_gpu(
    void *x_gpu,
    void *o_gpu,
    uint64_t MN,
    uint64_t K
);

void openfish_rmsnorm_gpu(
    const void* input,
    const void* residual,
    const void* weight,
    void* output,
    int MN,
    int K,
    float alpha,
    float eps
);

void openfish_rmsnorm_quant_gpu(
    const void* input,
    const void* weight,
    void* residual,
    void* residual_scale,
    int MN,
    int K,
    float alpha,
    float eps
);

void openfish_flstm_step_gpu(
    const void* scratch,
    const void* ih_t,
    void* cell,
    void* hh_next,
    int batch_size,
    int hidden_dim
);

size_t openfish_gpubuf_size(
    int n_timesteps,
    int batch_size,
    int state_len
);

#ifdef __cplusplus
}
#endif

#endif // OPENFISH_H