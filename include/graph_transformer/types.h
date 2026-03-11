#ifndef GRAPH_TRANSFORMER_TYPES_H
#define GRAPH_TRANSFORMER_TYPES_H

#include "ggml.h"
#include <stdint.h>

typedef struct {
    struct ggml_tensor *W_q;
    struct ggml_tensor *W_k;
    struct ggml_tensor *W_v;
    struct ggml_tensor *W_o;
} gt_attn_weights;

typedef struct {
    struct ggml_tensor *W1;
    struct ggml_tensor *W2;
} gt_ffn_weights;

typedef struct {
    struct ggml_tensor *gamma;
    struct ggml_tensor *beta;
    float eps;
} gt_ln_weights;

typedef struct {
    gt_attn_weights attn;
    gt_ffn_weights ffn;
    gt_ln_weights ln1;
    gt_ln_weights ln2;
} gt_block_weights;

typedef struct {
    int32_t num_nodes;
    int32_t num_edges;
    const int32_t *src;
    const int32_t *dst;
} gt_edge_data;

#endif
