#ifndef GRAPH_TRANSFORMER_BLOCK_H
#define GRAPH_TRANSFORMER_BLOCK_H

#include "graph_transformer/types.h"

struct ggml_tensor *gt_block_forward(
    struct ggml_context *ctx,
    struct ggml_tensor *x,
    const gt_block_weights *weights,
    int n_heads,
    int d_head,
    const gt_edge_data *edges);

#endif
