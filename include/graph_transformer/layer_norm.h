#ifndef GRAPH_TRANSFORMER_LAYER_NORM_H
#define GRAPH_TRANSFORMER_LAYER_NORM_H

#include "graph_transformer/types.h"

struct ggml_tensor *gt_ln_forward(
    struct ggml_context *ctx,
    struct ggml_tensor *x,
    const gt_ln_weights *weights);

#endif
