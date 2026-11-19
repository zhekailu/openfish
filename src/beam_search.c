#include "beam_search.h"
#include "decode.h"
#include "scores.h"

#include <math.h>
#include <float.h>

#ifndef BS_PREFETCH
#define BS_PREFETCH 4   // timesteps of scores to prefetch ahead in beam_search_cpu
#endif

static void swapf(float *a, float *b) {
    float temp = *a;
    *a = *b;
    *b = temp;
}

static int partitionf(float *nums, int left, int right) {
	float pivot = nums[left];
    int l = left+1;
    int r = right;
	while (l <= r) {
		if (nums[l] < pivot && nums[r] > pivot) {
            swapf(&nums[l], &nums[r]);
            l += 1;
            r -= 1;
        }
		if (nums[l] >= pivot) ++l;
		if (nums[r] <= pivot) --r;
	}
	swapf(&nums[left], &nums[r]);
	return r;
}

static float kth_largestf(float *nums, int k, int n) {
	int left = 0;
    int right = n-1;
    int idx = 0;
	while (true) {
		idx = partitionf(nums, left, right);
		if (idx == k) break;
		else if (idx < k) left = idx+1;
		else right = idx-1;
	}
	return nums[idx];
}

static float log_sum_exp(float x, float y) {
    float abs_diff = fabs(x - y);
    float m = x > y ? x : y;
    return m + ((abs_diff < 17.0f) ? (log( 1.0 + exp(-abs_diff))) : 0.0f);
}

void generate_sequence_cpu(
    const uint8_t *moves,
    const state_t *states,
    const float *qual_data,
    const float shift,
    const float scale,
    const size_t num_ts,
    const size_t seq_len,
    float *base_probs,
    float *total_probs,
    char *sequence,
    char *qstring
) {
    size_t seq_pos = 0;

    const char alphabet[4] = {'A', 'C', 'G', 'T'};

    for (size_t blk = 0; blk < num_ts; ++blk) {
        int state = states[blk];
        int move = (int)moves[blk];
        int base = state & 3;
        int offset = (blk == 0) ? 0 : move - 1;
        int probPos = (int)(seq_pos + offset);

        // get the probability for the called base.
        base_probs[probPos] += qual_data[blk * NUM_BASES + base];

        // accumulate the total probability for all possible bases at this position, for normalization.
        for (size_t k = 0; k < NUM_BASES; ++k) {
            total_probs[probPos] += qual_data[blk * NUM_BASES + k];
        }

        if (blk == 0) {
            sequence[seq_pos++] = (char)base;
        } else {
            for (int j = 0; j < move; ++j) {
                sequence[seq_pos++] = (char)base;
            }
        }
    }

    for (size_t i = 0; i < seq_len; ++i) {
        sequence[i] = alphabet[(int)sequence[i]];
        base_probs[i] = 1.0f - (base_probs[i] / total_probs[i]);
        base_probs[i] = -10.0f * log10f(base_probs[i]);
        float qscore = base_probs[i] * scale + shift;
        if (qscore > 50.0f) qscore = 50.0f;
        if (qscore < 1.0f) qscore = 1.0f;
        qstring[i] = (char)(33.5f + qscore);
    }
}

// Incorporates NUM_NEW_BITS into a Castagnoli CRC32, aka CRC32C
// (not the same polynomial as CRC32 as used in zip/ethernet).
static uint32_t crc32c(uint32_t crc, uint32_t new_bits, int num_new_bits) {
    // Note that this is the reversed polynomial.
    const uint32_t POLYNOMIAL = 0x82f63b78u;
    for (int i = 0; i < num_new_bits; ++i) {
        uint32_t b = (new_bits ^ crc) & 1;
        crc >>= 1;
        if (b) {
            crc ^= POLYNOMIAL;
        }
        new_bits >>= 1;
    }
    return crc;
}

// beam_search_cpu, restructured for speed with exactly the arithmetic of the original:
//  * candidates in separate arrays (hash, state, score) instead of structs, at the original
//    indices: the steps of beam element j at 4j + b, its stay at 4W + j (W = current beam width).
//    A candidate's beam element and stay flag follow from its index.
//  * the 4 steps of one beam element are computed together (NEON): scores (one table lookup for
//    the 4 strided transition scores), backward guides, states and hashes. The 2-bit CRC32C update
//    is a 4-entry table: CRC is linear, so crc32c(h, b, 2) == (h >> 2) ^ CRC2[(h ^ b) & 3].
//  * stay scores 4 at a time; the merges then run stay by stay, in the original order.
//  * finding the steps a stay merges with: for a fixed base, the 2-bit CRC update is a bijection
//    of the hash, so the step 4j + b has the stay's hash h exactly when beam element j's hash is
//    the unique preimage of h under base b. So a stay looks that one value up among the W beam
//    hashes (a 64-bucket table built once per timestep, chains in ascending j) instead of
//    comparing W step hashes behind a Bloom filter, and merges with the matches in ascending j,
//    as before (each merge changes scores the next one reads). If the CRC table
//    does not give a bijection (it does for this polynomial; checked at runtime), the original
//    scan over step hashes is used.
//  * the beam cut counts candidates >= cutoff 4 at a time; the binary search is unchanged; the
//    survivors are packed 4 at a time (NEON compress), in order, stopping once the beam is full,
//    and copied to the beam vector 4 elements per store
//  * scores for a later timestep are prefetched: one chunk's scores for consecutive timesteps are
//    N * C values apart in the TNC layout, so the loop otherwise waits on memory
//  * per-base qualities: each state's list of distinct neighbour states is built once per call,
//    in the order the original visits them, so block_prob sums the same terms in the same order
// Every float operation (each add, mul, compare, log_sum_exp, powf) is the original one, in the
// original order, so the output is identical. (A max over candidates is order independent for
// non-NaN scores.)
#ifndef BS_NEON   // -DBS_NEON=0 selects the portable path on aarch64 too
#if defined(__aarch64__) && defined(__ARM_NEON)
#define BS_NEON 1
#else
#define BS_NEON 0
#endif
#endif
#if BS_NEON
#include <arm_neon.h>
#endif

// number of candidates whose score is >= cutoff
static inline size_t count_ge(const float *scores, size_t n, float cutoff) {
    size_t i = 0, count = 0;
#if BS_NEON
    const float32x4_t c = vdupq_n_f32(cutoff);
    uint32x4_t acc = vdupq_n_u32(0);
    for (; i + 4 <= n; i += 4) {
        // a true lane is all ones (-1), so subtracting the mask adds 1
        acc = vsubq_u32(acc, vcgeq_f32(vld1q_f32(scores + i), c));
    }
    count = vaddvq_u32(acc);
#endif
    for (; i < n; ++i) {
        if (scores[i] >= cutoff) ++count;
    }
    return count;
}

// merge stay candidate s with step candidate t, as the original does
static inline void merge_stay_step(float *score, size_t s, size_t t, float *max_score) {
    const float folded_score = log_sum_exp(score[s], score[t]);
    // the better scoring one gets the folded score, the other is discarded
    if (score[s] > score[t]) {
        score[s] = folded_score;
        score[t] = -FLT_MAX;
    } else {
        score[t] = folded_score;
        score[s] = -FLT_MAX;
    }
    *max_score = folded_score > *max_score ? folded_score : *max_score;
}

void beam_search_cpu(
    const void *const scores_TNC,
    const score_type_t score_type,
    size_t scores_block_stride,
    const float *const bwd_NTC,
    const float *const post_NTC,
    const int num_state_bits,
    const size_t T,
    const size_t beam_width,
    const float beam_cut,
    const float fixed_stay_score,
    state_t *states,
    uint8_t *moves,
    float *qual_data,
    float score_scale,
    float posts_scale,
    beam_element_t *beam_vector
) {
    const size_t num_states = 1ull << num_state_bits;
    const state_t states_mask = (state_t)(num_states - 1);
    const float log_beam_cut = (beam_cut > 0.0f) ? logf(beam_cut) : FLT_MAX;
    const size_t score_bytes = (num_states * NUM_BASES) * score_elem_size(score_type);

    // crc32c(h, b, 2) == (h >> 2) ^ crc2[(h ^ b) & 3]; crc2_perm[x][b] = crc2[x ^ b]
    uint32_t crc2[NUM_BASES];
    for (uint32_t v = 0; v < NUM_BASES; ++v) crc2[v] = crc32c(v, 0, NUM_BASE_BITS);
    uint32_t crc2_perm[NUM_BASES][NUM_BASES];
    for (int x = 0; x < NUM_BASES; ++x)
        for (int b = 0; b < NUM_BASES; ++b) crc2_perm[x][b] = crc2[x ^ b];
    // Preimage: h' = (h >> 2) ^ crc2[y] with y = (h ^ b) & 3. (h >> 2) has its top 2 bits clear, so
    // y is the entry whose top 2 bits equal h's; if the 4 entries' top bits are all different, y and
    // so h = ((h' ^ crc2[y]) << 2) | (y ^ b) are unique.
    int crc2_by_top[4] = {-1, -1, -1, -1};
    bool crc2_invertible = true;
    for (int y = 0; y < NUM_BASES; ++y) {
        const int top = (int)(crc2[y] >> 30);
        if (crc2_by_top[top] >= 0) crc2_invertible = false;
        crc2_by_top[top] = y;
    }

    // candidates; scratch room past 5 * MAX_BEAM_WIDTH for 4-wide writes and loads
    enum { CAND_CAP = MAX_BEAM_CANDIDATES + 8 };
    uint32_t cand_hash[CAND_CAP];
    state_t cand_state[CAND_CAP];
    float cand_score[CAND_CAP];

    // the beam after the cut (4-wide packing may write a few slots past beam_width)
    uint32_t prev_hash[CAND_CAP];
    state_t prev_state[CAND_CAP];
    uint32_t prev_prev[CAND_CAP];
    uint32_t prev_stay[CAND_CAP];
    float prev_scores[CAND_CAP];

    // Find the score an initial element needs in order to make it into the beam
    float beam_init_threshold = -FLT_MAX;
    if (beam_width < num_states) {
        // copy the first set of back guides and sort to extract max_beam_width highest elements
        float sorted_bwd_NTCs[MAX_STATES];
        for (size_t i = 0; i < num_states; ++i) {
            sorted_bwd_NTCs[i] = bwd_NTC[i];
        }
        // note that we don't need a full sort here to get the max_beam_width highest values
        beam_init_threshold = kth_largestf(sorted_bwd_NTCs, beam_width-1, num_states);
    }

    // initialise all beams from the first block of the back guide
    for (size_t state = 0, beam_element = 0; state < num_states && beam_element < beam_width; state++) {
        if (bwd_NTC[state] >= beam_init_threshold) {
            // note that this first element has a prev_element_index of 0
            prev_hash[beam_element] = crc32c(CRC_SEED, (uint32_t)state, 32);
            prev_state[beam_element] = (state_t)state;
            prev_prev[beam_element] = 0;
            prev_stay[beam_element] = 0;
            prev_scores[beam_element] = 0.0f;
            ++beam_element;
        }
    }

    // copy all beam fronts into the beam persistent state
    size_t current_beam_width = beam_width < num_states ? beam_width : num_states;
    for (size_t element_idx = 0; element_idx < current_beam_width; ++element_idx) {
        beam_vector[element_idx].state = prev_state[element_idx];
        beam_vector[element_idx].prev_element_index = (uint8_t)prev_prev[element_idx];
        beam_vector[element_idx].stay = prev_stay[element_idx] != 0;
    }

#if BS_NEON
    // byte indices gathering scores 4b + top (b = 0..3) out of 16 consecutive ones, per top
    uint8x8_t gather16[NUM_BASES];
    uint8x16_t gather32[NUM_BASES];
    for (int t = 0; t < NUM_BASES; ++t) {
        uint8_t i16[8], i32[16];
        for (int b = 0; b < NUM_BASES; ++b) {
            for (int k = 0; k < 2; ++k) i16[2 * b + k] = (uint8_t)(2 * (4 * b + t) + k);
            for (int k = 0; k < 4; ++k) i32[4 * b + k] = (uint8_t)(4 * (4 * b + t) + k);
        }
        gather16[t] = vld1_u8(i16);
        gather32[t] = vld1q_u8(i32);
    }
    // compress: for a 4-bit mask of passing lanes, the byte shuffle packing them to the front
    uint8x16_t pack[16];
    for (int m = 0; m < 16; ++m) {
        uint8_t idx[16];
        int o = 0;
        for (int l = 0; l < 4; ++l) {
            if (!(m & (1 << l))) continue;
            for (int k = 0; k < 4; ++k) idx[4 * o + k] = (uint8_t)(4 * l + k);
            ++o;
        }
        for (int q = o; q < 4; ++q)
            for (int k = 0; k < 4; ++k) idx[4 * q + k] = 0xff;   // zero fill (never kept)
        pack[m] = vld1q_u8(idx);
    }
    const uint32x4_t lane_bits = {1, 2, 4, 8};
#endif

    // beam hashes by bucket (low 6 bits), chained in ascending element order; valid for the
    // current timestep only when the stamp matches, so nothing is cleared per timestep
    enum { PH_BUCKETS = 64 };
    uint32_t ph_stamp[PH_BUCKETS];
    int8_t ph_head[PH_BUCKETS], ph_tail[PH_BUCKETS], ph_next[MAX_BEAM_WIDTH];
    memset(ph_stamp, 0, sizeof(ph_stamp));

    for (size_t pf = 0; pf < BS_PREFETCH && pf < T; ++pf) {
        const char *p = (const char *)score_ptr(scores_TNC, score_type, pf * scores_block_stride);
        for (size_t off = 0; off < score_bytes; off += 64) __builtin_prefetch(p + off, 0, 0);
    }

    // iterate through blocks, extending beam
    for (size_t block_idx = 0; block_idx < T; ++block_idx) {
        const void *const block_scores = score_ptr(scores_TNC, score_type, block_idx * scores_block_stride);
        const float *const block_back_scores = bwd_NTC + ((block_idx + 1) << num_state_bits);
        const size_t W = current_beam_width;
        const size_t steps_end = W * NUM_BASES;

        if (block_idx + BS_PREFETCH < T) {
            const char *p = (const char *)score_ptr(scores_TNC, score_type, (block_idx + BS_PREFETCH) * scores_block_stride);
            for (size_t off = 0; off < score_bytes; off += 64) __builtin_prefetch(p + off, 0, 0);
        }

        // ---- steps: element j extended by base b -> candidate 4j + b
        float max_score = -FLT_MAX;
#if BS_NEON
        float32x4_t vmax = vdupq_n_f32(-FLT_MAX);
        const int32x4_t lane_base = {0, 1, 2, 3};
        const float32x4_t vscale = vdupq_n_f32(score_scale);
#endif
        for (size_t j = 0; j < W; ++j) {
            const state_t s = prev_state[j];
            // shift the state (k-mer) left and append the new base: new_state = ns_base | b
            const state_t ns_base = (state_t)((s << NUM_BASE_BITS) & states_mask);
            // transition index = (new_state << 2) + base that leaves = 4 * ns_base + 4 * b + top
            const state_t top = (state_t)((s << NUM_BASE_BITS) >> num_state_bits);
            const size_t mi0 = (size_t)(ns_base << NUM_BASE_BITS) + (size_t)top;
            const uint32_t h = prev_hash[j];
            const size_t c0 = j * NUM_BASES;
#if BS_NEON
            float32x4_t block_score;
            if (score_type == SCORES_F16) {
                const uint8x16x2_t row = vld1q_u8_x2((const uint8_t *)((const uint16_t *)block_scores + (mi0 - (size_t)top)));
                block_score = vcvt_f32_f16(vreinterpret_f16_u8(vqtbl2_u8(row, gather16[top])));
            } else {
                const uint8x16x4_t row = vld1q_u8_x4((const uint8_t *)((const float *)block_scores + (mi0 - (size_t)top)));
                block_score = vreinterpretq_f32_u8(vqtbl4q_u8(row, gather32[top]));
            }
            block_score = vmulq_f32(block_score, vscale);
            // (prev score + block score) + back guide, the original order
            const float32x4_t score = vaddq_f32(vaddq_f32(vdupq_n_f32(prev_scores[j]), block_score),
                                                vld1q_f32(block_back_scores + ns_base));
            vst1q_f32(cand_score + c0, score);
            vst1q_u32(cand_hash + c0, veorq_u32(vdupq_n_u32(h >> NUM_BASE_BITS), vld1q_u32(crc2_perm[h & 3])));
            vst1q_s32(cand_state + c0, vorrq_s32(vdupq_n_s32(ns_base), lane_base));
            vmax = vmaxq_f32(vmax, score);
#else
            for (int b = 0; b < NUM_BASES; ++b) {
                const float block_score = load_score(block_scores, score_type, mi0 + (size_t)(b << NUM_BASE_BITS)) * score_scale;
                const float new_score = prev_scores[j] + block_score + (float)block_back_scores[ns_base | b];
                cand_score[c0 + b] = new_score;
                cand_hash[c0 + b] = (h >> NUM_BASE_BITS) ^ crc2_perm[h & 3][b];
                cand_state[c0 + b] = ns_base | (state_t)b;
                max_score = max_score > new_score ? max_score : new_score;
            }
#endif
        }

        // ---- stays: candidate 4W + i is beam element i, unchanged
        {
            size_t i = 0;
#if BS_NEON
            const float32x4_t vfixed = vdupq_n_f32(fixed_stay_score);
            for (; i + 4 <= W; i += 4) {
                const float g[4] = {block_back_scores[prev_state[i]], block_back_scores[prev_state[i + 1]],
                                    block_back_scores[prev_state[i + 2]], block_back_scores[prev_state[i + 3]]};
                // (prev score + fixed stay score) + back guide, the original order
                const float32x4_t stay = vaddq_f32(vaddq_f32(vld1q_f32(prev_scores + i), vfixed), vld1q_f32(g));
                vst1q_f32(cand_score + steps_end + i, stay);
                vst1q_u32(cand_hash + steps_end + i, vld1q_u32(prev_hash + i));
                vst1q_s32(cand_state + steps_end + i, vld1q_s32(prev_state + i));
                vmax = vmaxq_f32(vmax, stay);
            }
            if (W > 0) max_score = vmaxvq_f32(vmax);
#endif
            for (; i < W; ++i) {
                const float stay_score = prev_scores[i] + fixed_stay_score + (float)block_back_scores[prev_state[i]];
                cand_score[steps_end + i] = stay_score;
                cand_hash[steps_end + i] = prev_hash[i];
                cand_state[steps_end + i] = prev_state[i];
                max_score = max_score > stay_score ? max_score : stay_score;
            }
        }

        // ---- merges: each stay with the steps of the same sequence, in the original order
        const uint32_t stamp = (uint32_t)block_idx + 1;
        if (crc2_invertible) {
            for (size_t j = 0; j < W; ++j) {
                const uint32_t bucket = prev_hash[j] & (PH_BUCKETS - 1);
                if (ph_stamp[bucket] != stamp) {
                    ph_stamp[bucket] = stamp;
                    ph_head[bucket] = (int8_t)j;
                } else {
                    ph_next[ph_tail[bucket]] = (int8_t)j;
                }
                ph_tail[bucket] = (int8_t)j;
                ph_next[j] = -1;
            }
        }
        for (size_t i = 0; i < W; ++i) {
            const size_t stay_elem_idx = steps_end + i;
            const uint32_t h = prev_hash[i];
            const int base = (int)(prev_state[i] & 3);
            if (crc2_invertible) {
                // the beam elements j whose step with this base has hash h: those with hash == target
                const int y = crc2_by_top[h >> 30];
                const uint32_t target = ((h ^ crc2[y]) << NUM_BASE_BITS) | (uint32_t)(y ^ base);
                const uint32_t bucket = target & (PH_BUCKETS - 1);
                if (ph_stamp[bucket] == stamp) {
                    for (int j = ph_head[bucket]; j >= 0; j = ph_next[j])
                        if (prev_hash[j] == target)
                            merge_stay_step(cand_score, stay_elem_idx, ((size_t)j << NUM_BASE_BITS) | (size_t)base, &max_score);
                }
            } else {
                for (size_t j = 0; j < W; ++j) {
                    const size_t step_elem_idx = (j << NUM_BASE_BITS) | (size_t)base;
                    if (cand_hash[stay_elem_idx] == cand_hash[step_elem_idx])
                        merge_stay_step(cand_score, stay_elem_idx, step_elem_idx, &max_score);
                }
            }
        }
        const size_t new_elem_count = steps_end + W;

        // ---- cut off to fit beam width, starting from the beam cut score
        float beam_cutoff_score = max_score - log_beam_cut;
        size_t elem_count = count_ge(cand_score, new_elem_count, beam_cutoff_score);

        // binary search for a score that keeps between 80% and 100% of the beam width
        if (elem_count > beam_width) {
            size_t min_beam_width = (beam_width * 8) / 10;
            float low_score = beam_cutoff_score;
            float hi_score = max_score;
            int num_guesses = 1;
            const int MAX_GUESSES = 10;

            while ((elem_count > beam_width || elem_count < min_beam_width) && num_guesses < MAX_GUESSES) {
                if (elem_count > beam_width) {
                    // make a higher guess
                    low_score = beam_cutoff_score;
                    beam_cutoff_score = (beam_cutoff_score + hi_score) / 2.0f;
                } else {
                    // make a lower guess
                    hi_score = beam_cutoff_score;
                    beam_cutoff_score = (beam_cutoff_score + low_score) / 2.0f;
                }
                elem_count = count_ge(cand_score, new_elem_count, beam_cutoff_score);
                ++num_guesses;
            }

            // no suitable score after MAX_GUESSES: take the higher limit (see the original comment)
            if (num_guesses == MAX_GUESSES) {
                beam_cutoff_score = hi_score;
                elem_count = count_ge(cand_score, new_elem_count, beam_cutoff_score);
            }

            // clamp the element count to the max beam width
            elem_count = elem_count < beam_width ? elem_count : beam_width;
        }

        // ---- keep the candidates that pass the cut, in order, until beam_width are kept
        {
            size_t write_idx = 0, read_idx = 0;
#if BS_NEON
            const float32x4_t vcut = vdupq_n_f32(beam_cutoff_score);
            const uint32x4_t vsteps_end = vdupq_n_u32((uint32_t)steps_end);
            uint32x4_t vidx = {0, 1, 2, 3};
            for (; read_idx + 4 <= new_elem_count && write_idx < beam_width; read_idx += 4) {
                const float32x4_t sc = vld1q_f32(cand_score + read_idx);
                const uint32x4_t pass = vcgeq_f32(sc, vcut);
                const unsigned m = vaddvq_u32(vandq_u32(pass, lane_bits));
                if (m) {
                    const uint8x16_t pk = pack[m];
                    // beam element and stay flag of each candidate, from its index
                    const uint32x4_t is_stay = vcgeq_u32(vidx, vsteps_end);
                    const uint32x4_t prev = vbslq_u32(is_stay, vsubq_u32(vidx, vsteps_end), vshrq_n_u32(vidx, NUM_BASE_BITS));
                    vst1q_u8((uint8_t *)(prev_hash + write_idx), vqtbl1q_u8(vreinterpretq_u8_u32(vld1q_u32(cand_hash + read_idx)), pk));
                    vst1q_u8((uint8_t *)(prev_state + write_idx), vqtbl1q_u8(vreinterpretq_u8_s32(vld1q_s32(cand_state + read_idx)), pk));
                    vst1q_u8((uint8_t *)(prev_scores + write_idx), vqtbl1q_u8(vreinterpretq_u8_f32(sc), pk));
                    vst1q_u8((uint8_t *)(prev_prev + write_idx), vqtbl1q_u8(vreinterpretq_u8_u32(prev), pk));
                    vst1q_u8((uint8_t *)(prev_stay + write_idx), vqtbl1q_u8(vreinterpretq_u8_u32(vshrq_n_u32(is_stay, 31)), pk));
                    write_idx += (size_t)__builtin_popcount(m);
                }
                vidx = vaddq_u32(vidx, vdupq_n_u32(4));
            }
#endif
            // the rest one at a time (stops at beam_width like the original); entries a 4-wide
            // write put past beam_width are ignored
            for (; read_idx < new_elem_count && write_idx < beam_width; ++read_idx) {
                if (cand_score[read_idx] >= beam_cutoff_score) {
                    const bool stay = read_idx >= steps_end;
                    prev_hash[write_idx] = cand_hash[read_idx];
                    prev_state[write_idx] = cand_state[read_idx];
                    prev_prev[write_idx] = (uint32_t)(stay ? read_idx - steps_end : read_idx >> NUM_BASE_BITS);
                    prev_stay[write_idx] = stay;
                    prev_scores[write_idx] = cand_score[read_idx];
                    ++write_idx;
                }
            }
        }

        // at the last timestep, the best path must be element 0
        if (block_idx == T - 1) {
            float best_score = -FLT_MAX;
            size_t best = 0;
            for (size_t i = 0; i < elem_count; i++) {
                if (prev_scores[i] > best_score) {
                    best_score = prev_scores[i];
                    best = i;
                }
            }
            { uint32_t t = prev_hash[0]; prev_hash[0] = prev_hash[best]; prev_hash[best] = t; }
            { state_t t = prev_state[0]; prev_state[0] = prev_state[best]; prev_state[best] = t; }
            { uint32_t t = prev_prev[0]; prev_prev[0] = prev_prev[best]; prev_prev[best] = t; }
            { uint32_t t = prev_stay[0]; prev_stay[0] = prev_stay[best]; prev_stay[best] = t; }
            { float t = prev_scores[0]; prev_scores[0] = prev_scores[best]; prev_scores[best] = t; }
        }

        // copy this new beam front into the beam persistent state
        size_t beam_offset = (block_idx + 1) * MAX_BEAM_WIDTH;
        size_t i = 0;
#if BS_NEON
        // beam_element_t is {int32 state; uint8 prev_element_index; bool stay; 2 pad}: as two words,
        // state and prev | stay << 8, so 4 elements are one interleaving store
        _Static_assert(sizeof(beam_element_t) == 8, "beam_element_t layout");
        for (; i + 4 <= elem_count; i += 4) {
            const float g[4] = {block_back_scores[prev_state[i]], block_back_scores[prev_state[i + 1]],
                                block_back_scores[prev_state[i + 2]], block_back_scores[prev_state[i + 3]]};
            // remove backwards contribution from score
            vst1q_f32(prev_scores + i, vsubq_f32(vld1q_f32(prev_scores + i), vld1q_f32(g)));
            uint32x4x2_t e;
            e.val[0] = vreinterpretq_u32_s32(vld1q_s32(prev_state + i));
            e.val[1] = vorrq_u32(vld1q_u32(prev_prev + i), vshlq_n_u32(vld1q_u32(prev_stay + i), 8));
            vst2q_u32((uint32_t *)(beam_vector + beam_offset + i), e);
        }
#endif
        for (; i < elem_count; ++i) {
            // remove backwards contribution from score
            prev_scores[i] -= (float)block_back_scores[prev_state[i]];

            beam_vector[beam_offset + i].state = prev_state[i];
            beam_vector[beam_offset + i].prev_element_index = (uint8_t)prev_prev[i];
            beam_vector[beam_offset + i].stay = prev_stay[i] != 0;
        }
        current_beam_width = elem_count;
    }

    // write out sequence bases and move table
    // note that we don't emit the seed state at the front of the beam, hence the -1 offset when copying the path
    uint8_t element_index = 0;
    for (size_t beam_idx = T; beam_idx != 0; --beam_idx) {
        size_t beam_addr = beam_idx * MAX_BEAM_WIDTH + element_index;
        states[beam_idx - 1] = (int32_t)beam_vector[beam_addr].state;
        moves[beam_idx - 1] = beam_vector[beam_addr].stay ? 0 : 1;
        element_index = beam_vector[beam_addr].prev_element_index;
    }
    moves[0] = 1;  // always step in the first event

    // per-base qual data. For each state, the distinct left- and right-shifted states other than
    // itself, in the order the original loop visits (and counts) them.
    int8_t nbr_count[MAX_STATES];
    int16_t nbr[MAX_STATES][2 * NUM_BASES];
    {
        const int msb = ((int)num_states) >> NUM_BASE_BITS;
        for (int state = 0; state < (int)num_states; ++state) {
            int shifted_states[2 * NUM_BASES];
            const int l_shift_idx = state >> NUM_BASE_BITS;
            const int r_shift_idx = (state << NUM_BASE_BITS) % (int)num_states;
            for (int shift_base = 0; shift_base < NUM_BASES; ++shift_base) {
                shifted_states[2 * shift_base] = l_shift_idx + msb * shift_base;
                shifted_states[2 * shift_base + 1] = r_shift_idx + shift_base;
            }
            int n = 0;
            for (int state_idx = 0; state_idx < 2 * NUM_BASES; ++state_idx) {
                const int candidate_state = shifted_states[state_idx];
                bool count_state = (candidate_state != state);
                for (int inner = 0; count_state && inner < state_idx; ++inner) {
                    if (shifted_states[inner] == candidate_state) count_state = false;
                }
                if (count_state) nbr[state][n++] = (int16_t)candidate_state;
            }
            nbr_count[state] = (int8_t)n;
        }
    }
    for (size_t block_idx = 0; block_idx < T; ++block_idx) {
        const int state = states[block_idx];
        const int base_to_emit = state % NUM_BASES;
        states[block_idx] = base_to_emit;

        // compute a probability for this block, based on the path kmer
        const float *const timestep_posts = post_NTC + ((block_idx + 1) << num_state_bits);
        float block_prob = (float)(timestep_posts[state]) * posts_scale;
        for (int k = 0; k < nbr_count[state]; ++k) {
            block_prob += (float)(timestep_posts[nbr[state][k]]) * posts_scale;
        }
        if (block_prob > 1.0f) block_prob = 1.0f;
        else if (block_prob < 0.0f) block_prob = 0.0f;
        block_prob = powf(block_prob, 0.4f); // power fudge factor

        // calculate a placeholder qscore for the "wrong" bases
        const float wrong_base_prob = (1.0f - block_prob) / 3.0f;

        for (size_t base = 0; base < NUM_BASES; base++) {
            qual_data[block_idx * NUM_BASES + base] = ((int)base == base_to_emit ? block_prob : wrong_base_prob);
        }
    }
}
