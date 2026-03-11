#ifndef GRAPH_TRANSFORMER_ATTENTION_H
#define GRAPH_TRANSFORMER_ATTENTION_H

#include "graph_transformer/types.h"

struct ggml_tensor *gt_attention_forward(
    struct ggml_context *ctx,
    struct ggml_tensor *x,
    const gt_attn_weights *weights,
    int n_heads,
    int d_head,
    const gt_edge_data *edges);

#endif
