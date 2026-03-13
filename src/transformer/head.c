#include "graph_transformer/head.h"
#include "graph_transformer/types.h"

struct ggml_tensor *gt_node_head_forward(
    struct ggml_context *ctx,
    struct ggml_tensor *x,
    const gt_task_head_weights *weights) {
    struct ggml_tensor *xW = ggml_mul_mat(ctx, weights->W, x);
    struct ggml_tensor *broadcast_bias = ggml_repeat(ctx, weights->b, xW);
    return ggml_add(ctx, xW, broadcast_bias);
}


struct ggml_tensor *gt_graph_head_forward(
    struct ggml_context *ctx,
    struct ggml_tensor *x,
    const gt_task_head_weights *weights
){
    struct ggml_tensor *x_t = ggml_cont(ctx, ggml_transpose(ctx, x));
    struct ggml_tensor *pooled_t = ggml_mean(ctx, x_t);
    struct ggml_tensor *pooled = ggml_transpose(ctx, pooled_t);

    struct ggml_tensor *out = ggml_mul_mat(ctx, weights->W, pooled);
    struct ggml_tensor *broadcast_bias = ggml_repeat(ctx, weights->b, out);
    out = ggml_add(ctx, out, broadcast_bias);
    return out;
}
