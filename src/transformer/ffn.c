#include "graph_transformer/ffn.h"

struct ggml_tensor *gt_ffn_forward(
    struct ggml_context *ctx,
    struct ggml_tensor *x,
    const gt_ffn_weights *weights) {
    struct ggml_tensor *hidden = ggml_mul_mat(ctx, weights->W1, x);
    hidden = ggml_gelu(ctx, hidden);
    return ggml_mul_mat(ctx, weights->W2, hidden);
}
