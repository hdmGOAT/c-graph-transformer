#ifndef GRAPH_TRANSFORMER_FFN_H
#define GRAPH_TRANSFORMER_FFN_H

#include "graph_transformer/types.h"

struct ggml_tensor *gt_ffn_forward(
    struct ggml_context *ctx,
    struct ggml_tensor *x,
    const gt_ffn_weights *weights);

#endif
