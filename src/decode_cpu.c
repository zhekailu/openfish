#include "beam_search.h"
#include "error.h"
#include "misc.h"
#include "scores.h"

#include <openfish/openfish.h>
#include <openfish/openfish_error.h>

#include <math.h>
#include <pthread.h>

#define MIN(X, Y) (((X) < (Y)) ? (X) : (Y))

// The scans read chunk `chunk` of the scores and write row `row` of their outputs: row == chunk for
// batch-sized buffers, 0 for one-chunk (per-thread) buffers.
static void backward_scan(const void *scores_in, const score_type_t score_type, float *out, const uint64_t chunk, const uint64_t row, const uint64_t T, const uint64_t N, const uint64_t num_states) {
    const float fixed_stay_score = 2.0f;

    const uint64_t ts_states = num_states * NUM_BASES;

    const void* const chunk_in = score_ptr(scores_in, score_type, chunk * ts_states);
    float* const chunk_out = out + row * (T+1) * num_states;
    float* const alpha_init = chunk_out + num_states * T;
    for (uint64_t state = 0; state < num_states; ++state) { // (for GPU impl) its 1 thread per state, but below we iterate through all the states on 1 thread
        alpha_init[state] = 0.0f;
    }

    for (uint64_t ts = 0; ts < T; ++ts) {
        // threadgroup_barrier(mem_flags::medevice); // synchronize all threads before next time step (for GPU impl)
        const void* const ts_in = score_ptr(chunk_in, score_type, N * ts_states * (T - ts - 1));
        float* const ts_alpha_in = alpha_init - num_states * ts;
        float* const ts_alpha_out = ts_alpha_in - num_states;

        for (uint64_t state = 0; state < num_states; ++state) { // we should have 1 thread for each state (for GPU impl)
            const uint64_t stay_state_idx = state;
            const uint64_t step_state_idx_a = (state * NUM_BASES) % num_states;
            const uint64_t step_trans_idx_a = step_state_idx_a * NUM_BASES +
                ((state * NUM_BASES) / num_states);

            float vals[NUM_TRANSITIONS];
            vals[0] = ts_alpha_in[stay_state_idx] + fixed_stay_score;
            float max_val = vals[0];
            for (uint64_t base = 0; base < NUM_BASES; ++base) {
                vals[base + 1] = ts_alpha_in[step_state_idx_a + base] +
                    load_score(ts_in, score_type, step_trans_idx_a + base * NUM_BASES);
                max_val = max_val > vals[base + 1] ? max_val : vals[base + 1];
            }
            float sum = 0.0f;
            for (uint64_t i = 0; i < NUM_TRANSITIONS; ++i) {
                sum += expf(vals[i] - max_val);
            }
            ts_alpha_out[state] = max_val + logf(sum);
        }
    }
}

static void forward_scan(const void *scores_in, const score_type_t score_type, const float *bwd, float *out, const uint64_t chunk, const uint64_t row, const uint64_t _T, const uint64_t N, const uint64_t num_states) {
    const uint64_t T = _T+1; 
    const float kFixedStayScore = 2.0f;
    
    const uint64_t msb = num_states / NUM_BASES;
    const uint64_t ts_states = num_states * NUM_BASES;

    // This batch element's scores.
    const void *const chunk_scores = score_ptr(scores_in, score_type, chunk * ts_states);

    // Alternating forward guide buffers used for successive time steps.
    float ts_fwd[2][MAX_STATES]; // threadgroup

    // The forward guide input for the first step is 0.
    for (uint64_t state = 0; state < num_states; ++state) {
        ts_fwd[0][state] = 0.0f;
    }
    // threadgroup_barrier(mem_flags::mem_threadgroup); // ------------------------------------------------------------------

    for (uint64_t ts = 0; ts < T; ++ts) {
        // We read forward guide values written to TG memory in the previous step as
        // inputs to this step.  However, there has already been a TG barrier since
        // they were written.
        const uint64_t ts_idx = (row * T + ts) * num_states;

        // This time step's scores.
        const void *const ts_scores = score_ptr(chunk_scores, score_type, N * ts_states * ts);

        // Alternating TG buffer twiddling.
        const float *const ts_alpha_in = ts_fwd[ts & 1];
        float *const ts_alpha_out = ts_fwd[(ts & 1) ^ 1];

        // Calculate the next time step's forward guide from this time step's scores
        // and forward guide.  It's written to threadgroup memory for use in the
        // next iteration.
        for (uint64_t state = 0; state < num_states; ++state) { // we should have 1 thread for each state (for GPU impl)
            const uint64_t stay_state_idx = state;
            const uint64_t step_state_idx_a = state / NUM_BASES;
            const uint64_t step_trans_idx_a = state * NUM_BASES;
            float vals[NUM_TRANSITIONS];
            float fwd_max_val = vals[0] = ts_alpha_in[stay_state_idx] + kFixedStayScore;
            for (uint64_t base = 0; base < NUM_BASES; ++base) {
                // todo: this is a bandaid for indexing past the actual T dimension of scores
                // need to verify with actual MetalTxCaller impl output,
                // otherwise output remains exactly the same for this impl whether it indexes past or not
                float ts_score = ts < _T ? load_score(ts_scores, score_type, step_trans_idx_a + base) : 0.0f;

                vals[base + 1] = ts_alpha_in[step_state_idx_a + base * msb] + ts_score;
                fwd_max_val = fwd_max_val > vals[base + 1] ? fwd_max_val : vals[base + 1];
            }
            float fwd_sum = 0.0f;
            for (uint64_t i = 0; i < NUM_TRANSITIONS; ++i) {
                fwd_sum += expf(vals[i] - fwd_max_val);
            }
            ts_alpha_out[state] = fwd_max_val + logf(fwd_sum);

            // Load the forward guide value calculated in the last time step for use
            // in this time step's posterior probability calculation.
            const float fwd_val = ts_alpha_in[state];

            // Calculate fwd/bwd guide product in log space.
            const float val = fwd_val + bwd[ts_idx + state];
            out[ts_idx + state] = val;
        }
    }
}

static void softmax(const float *fwd, float *out, const uint64_t row, const uint64_t _T, const uint64_t num_states) {
    const uint64_t T = _T+1; 
    for (uint64_t ts = 0; ts < T; ++ts) {
        const uint64_t ts_idx = (row * T + ts) * num_states;

        float max_val = fwd[ts_idx];
        for (uint64_t state = 0; state < num_states; ++state) {
            max_val = max_val > fwd[ts_idx + state] ? max_val : fwd[ts_idx + state];
        }

        float exp_sum = 0;
        float exp_vals[num_states];
        for (uint64_t state = 0; state < num_states; ++state) {
            const float val = fwd[ts_idx + state];
            const float exp_val = expf(val - max_val);
            exp_vals[state] = exp_val;
            exp_sum += exp_val;
        }

        for (uint64_t state = 0; state < num_states; ++state) {
            const float exp_val = exp_vals[state];

            // Write out the posterior probability 
            out[ts_idx + state] = (float)(exp_val / exp_sum);
        }
    }
}

typedef struct {
    const openfish_opt_t *options;
    const void *scores_TNC;
    score_type_t score_type;
    float *bwd_NTC;
    float *fwd_NTC;
    float *post_NTC;
    int32_t start;
    int32_t end;
    int32_t state_len;
    int32_t T;
    int32_t N;
    int32_t C;
    state_t *states;
    uint8_t *moves;
    float *qual_data;
    float *base_probs;
    float *total_probs;
    char *sequence;
    char *qstring;
    beam_element_t *beam_vector;    // this thread's beam search scratch, one chunk
    bool chunk_rows;                // bwd/fwd/post hold every chunk (row c) or just one (row 0)
    bool scans_done;                // bwd/post given (every chunk, row c): skip the scans
} decode_thread_arg_t;

// One pass per chunk: scans, then beam search and sequence, while the chunk's guides are still in
// cache. Each thread's bwd/fwd/post/beam_vector hold a single chunk and are reused (DEBUG builds keep
// every chunk's guides, to write them out).
void *pthread_single_decode(void *voidargs) {
    decode_thread_arg_t *args = (decode_thread_arg_t *)voidargs;
    const openfish_opt_t *options = args->options;

    const int num_states = pow(NUM_BASES, args->state_len);
    const int num_state_bits = (int)log2(num_states);
    const int T = args->T;
    const int N = args->N;
    const int C = args->C;

    const float fixed_stay_score = options->blank_score;
    const float q_scale = options->q_scale;
    const float q_shift = options->q_shift;
    const size_t beam_width = options->beam_width;
    const float beam_cut = options->beam_cut;

    for (int c = args->start; c < args->end; c++) {
        const uint64_t row = args->chunk_rows ? (uint64_t)c : 0;
        if (!args->scans_done) {
            backward_scan(args->scores_TNC, args->score_type, args->bwd_NTC, c, row, T, N, num_states);
            forward_scan(args->scores_TNC, args->score_type, args->bwd_NTC, args->fwd_NTC, c, row, T, N, num_states);
            softmax(args->fwd_NTC, args->post_NTC, row, T, num_states);
        }

        const void *scores = score_ptr(args->scores_TNC, args->score_type, c * (num_states * NUM_BASES));
        float *bwd = args->bwd_NTC + row * num_states * (T+1);
        float *post = args->post_NTC + row * num_states * (T+1);
        state_t *states = args->states + c * T;
        uint8_t *moves = args->moves + c * T;
        float *qual_data = args->qual_data + c * (T * NUM_BASES);
        float *base_probs = args->base_probs + c * T;
        float *total_probs = args->total_probs + c * T;
        char *sequence = args->sequence + c * T;
        char *qstring = args->qstring + c * T;

        beam_search_cpu(scores, args->score_type, N * C, bwd, post, num_state_bits, T, beam_width, beam_cut, fixed_stay_score, states, moves, qual_data, 1.0f, 1.0f, args->beam_vector);

        size_t seq_len = 0;
        for (int i = 0; i < T; ++i) {
            seq_len += moves[i];
            total_probs[i] = 0;
            base_probs[i] = 0;
        }

        generate_sequence_cpu(moves, states, qual_data, q_shift, q_scale, T, seq_len, base_probs, total_probs, sequence, qstring);
    }

    pthread_exit(0);
}

static void decode_cpu(
    const int T,
    const int N,
    const int C,
    int nthreads,
    const void *scores_TNC,
    const score_type_t score_type,
    const int state_len,
    const openfish_opt_t *options,
    const float *bwd_in,            // precomputed guides for every chunk (both or neither): no scans
    const float *post_in,
    uint8_t **moves,
    char **sequence,
    char **qstring
) {
    const int num_states = pow(NUM_BASES, state_len);
    const bool scans_done = bwd_in != NULL && post_in != NULL;

    if (options->beam_width < 1 || options->beam_width > MAX_BEAM_WIDTH) {
        OPENFISH_ERROR("beam width must be between 1 and %d, got %zu", MAX_BEAM_WIDTH, options->beam_width);
        exit(EXIT_FAILURE);
    }

    OPENFISH_LOG_TRACE("scores tensor dim: %d, %d, %d", T, N, C);

    nthreads = N < nthreads ? N : nthreads;

    // guides: one chunk per thread, reused for each of its chunks (every chunk in DEBUG builds)
#ifdef DEBUG
    const bool chunk_rows = true;
    const size_t guide_rows = (size_t)N;
#else
    const bool chunk_rows = false;
    const size_t guide_rows = (size_t)nthreads;
#endif
    const size_t guide_len = (size_t)(T + 1) * num_states;
    float *bwd_NTC = NULL, *fwd_NTC = NULL, *post_NTC = NULL;
    if (!scans_done) {
        bwd_NTC = (float *)calloc(guide_rows * guide_len, sizeof(float));
        MALLOC_CHK(bwd_NTC);
        fwd_NTC = (float *)calloc(guide_rows * guide_len, sizeof(float));
        MALLOC_CHK(fwd_NTC);
        post_NTC = (float *)calloc(guide_rows * guide_len, sizeof(float));
        MALLOC_CHK(post_NTC);
    }

    // init results
    *moves = (uint8_t *)calloc(N * T, sizeof(uint8_t));
    MALLOC_CHK(*moves);

    *sequence = (char *)calloc(N * T, sizeof(char));
    MALLOC_CHK(*sequence);

    *qstring = (char *)calloc(N * T, sizeof(char));
    MALLOC_CHK(*qstring);

    // intermediate; the beam vector is beam search scratch, one chunk per thread
    const size_t beam_len = (size_t)MAX_BEAM_WIDTH * (T + 1);
    beam_element_t *beam_vector = (beam_element_t *)malloc((size_t)nthreads * beam_len * sizeof(beam_element_t));
    MALLOC_CHK(beam_vector);

    state_t *states = (state_t *)malloc(N * T * sizeof(state_t));
    MALLOC_CHK(states);

    float *qual_data = (float *)malloc(N * T * NUM_BASES * sizeof(float));
    MALLOC_CHK(qual_data);

    float *base_probs = (float *)malloc(N * T * sizeof(float));
    MALLOC_CHK(base_probs);

    float *total_probs = (float *)malloc(N * T * sizeof(float));
    MALLOC_CHK(total_probs);
    
    // create threads
    const int chunks_per_thread = N / nthreads;
    const int num_threads_with_one_more_chunk = N % nthreads;

    OPENFISH_LOG_TRACE("dispatching %d threads for cpu decoding", nthreads);

    pthread_t tids[nthreads];
    decode_thread_arg_t pt_args[nthreads];
    int32_t t, ret;

    // set the data structures
    for (t = 0; t < nthreads; t++) {
        int extra = t < num_threads_with_one_more_chunk ? t : num_threads_with_one_more_chunk;
        pt_args[t].start = t * chunks_per_thread + extra;
        pt_args[t].end = pt_args[t].start + chunks_per_thread + (int)(t < num_threads_with_one_more_chunk);
        pt_args[t].scores_TNC = scores_TNC;
        pt_args[t].score_type = score_type;
        pt_args[t].scans_done = scans_done;
        if (scans_done) {
            // read-only from here on: the beam search only reads the guides
            pt_args[t].chunk_rows = true;
            pt_args[t].bwd_NTC = (float *)bwd_in;
            pt_args[t].fwd_NTC = NULL;
            pt_args[t].post_NTC = (float *)post_in;
        } else {
            pt_args[t].chunk_rows = chunk_rows;
            pt_args[t].bwd_NTC = chunk_rows ? bwd_NTC : bwd_NTC + (size_t)t * guide_len;
            pt_args[t].fwd_NTC = chunk_rows ? fwd_NTC : fwd_NTC + (size_t)t * guide_len;
            pt_args[t].post_NTC = chunk_rows ? post_NTC : post_NTC + (size_t)t * guide_len;
        }
        pt_args[t].options = options;
        pt_args[t].state_len = state_len;
        pt_args[t].T = T;
        pt_args[t].N = N;
        pt_args[t].C = C;
        pt_args[t].states = states;
        pt_args[t].moves = *moves;
        pt_args[t].qual_data = qual_data;
        pt_args[t].base_probs = base_probs;
        pt_args[t].total_probs = total_probs;
        pt_args[t].sequence = *sequence;
        pt_args[t].qstring = *qstring;
        pt_args[t].beam_vector = beam_vector + (size_t)t * beam_len;
    }

    // scans, beam search and sequence, one chunk at a time per thread
    for (t = 0; t < nthreads; t++) {
        ret = pthread_create(&tids[t], NULL, pthread_single_decode, (void *)(&pt_args[t]));
        NEG_CHK(ret);
    }

    for (t = 0; t < nthreads; t++) {
        ret = pthread_join(tids[t], NULL);
        NEG_CHK(ret);
    }

#ifdef DEBUG
    if (!scans_done) {
    // write tensors
    FILE *fp;
    
    fp = fopen("bwd_NTC.blob", "w");
    F_CHK(fp, "bwd_NTC.blob");
    if (fwrite(bwd_NTC, sizeof(float), N * (T + 1) * num_states, fp) != N * (T + 1) * num_states) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("fwd_NTC.blob", "w");
    F_CHK(fp, "fwd_NTC.blob");
    if (fwrite(fwd_NTC, sizeof(float), N * (T + 1) * num_states, fp) != N * (T + 1) * num_states) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("post_NTC.blob", "w");
    F_CHK(fp, "post_NTC.blob");
    if (fwrite(post_NTC, sizeof(float), N * (T + 1) * num_states, fp) != N * (T + 1) * num_states) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    // write beam results
    fp = fopen("qual_data.blob", "w");
    F_CHK(fp, "qual_data.blob");
    if (fwrite(qual_data, sizeof(float), N * T * NUM_BASES, fp) != N * T * NUM_BASES) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("base_probs.blob", "w");
    F_CHK(fp, "base_probs.blob");
    if (fwrite(base_probs, sizeof(float), N * T, fp) != N * T) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);

    fp = fopen("total_probs.blob", "w");
    F_CHK(fp, "total_probs.blob");
    if (fwrite(total_probs, sizeof(float), N * T, fp) != N * T) {
        fprintf(stderr, "error writing sequence file: %s\n", strerror(errno));
        exit(EXIT_FAILURE);
    }
    fclose(fp);
    }
#endif

    // cleanup
    free(bwd_NTC);
    free(fwd_NTC);
    free(post_NTC);

    free(beam_vector);
    free(qual_data);
    free(states);
    free(base_probs);
    free(total_probs);
}

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
) {
    decode_cpu(T, N, C, nthreads, scores_TNC, SCORES_F32, state_len, options, NULL, NULL, moves, sequence, qstring);
}

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
) {
    decode_cpu(T, N, C, nthreads, scores_TNC, SCORES_F16, state_len, options, NULL, NULL, moves, sequence, qstring);
}

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
) {
    if (bwd_NTC == NULL || post_NTC == NULL) {
        OPENFISH_ERROR("%s", "openfish_beam_cpu_fp16 needs the backward and posterior guides");
        exit(EXIT_FAILURE);
    }
    decode_cpu(T, N, C, nthreads, scores_TNC, SCORES_F16, state_len, options, bwd_NTC, post_NTC, moves, sequence, qstring);
}
