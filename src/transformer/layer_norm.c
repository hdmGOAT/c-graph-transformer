#include "graph_transformer/layer_norm.h"

struct ggml_tensor *gt_ln_forward(
    struct ggml_context *ctx,
    struct ggml_tensor *x,
    const gt_ln_weights *weights) {
    struct ggml_tensor *norm = ggml_norm(ctx, x, weights->eps);
    struct ggml_tensor *gamma = ggml_repeat(ctx, weights->gamma, norm);
    struct ggml_tensor *beta = ggml_repeat(ctx, weights->beta, norm);
    struct ggml_tensor *scaled = ggml_mul(ctx, norm, gamma);
    return ggml_add(ctx, scaled, beta);
}
