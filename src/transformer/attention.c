#include "graph_transformer/attention.h"

#include <math.h>

static struct ggml_tensor *gt_build_attention_mask(
    struct ggml_context *ctx,
    int64_t width,
    int64_t height,
    const gt_edge_data *edges) {
    struct ggml_tensor *mask = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, width, height);
    float *data = (float *)mask->data;
    const int64_t total = width * height;

    for (int64_t index = 0; index < total; ++index) {
        data[index] = -1e9f;
    }

    const int64_t diag = width < height ? width : height;
    for (int64_t node = 0; node < diag; ++node) {
        data[node * width + node] = 0.0f;
    }

    if (edges != NULL && edges->src != NULL && edges->dst != NULL) {
        for (int32_t edge = 0; edge < edges->num_edges; ++edge) {
            const int32_t src = edges->src[edge];
            const int32_t dst = edges->dst[edge];
            if (src >= 0 && src < width && dst >= 0 && dst < height) {
                data[dst * width + src] = 0.0f;
            }
        }
    }

    return mask;
}

static struct ggml_tensor *gt_graph_attention_kernel(
    struct ggml_context *ctx,
    struct ggml_tensor *Q,
    struct ggml_tensor *K,
    struct ggml_tensor *V,
    const gt_edge_data *edges) {
    const float scale = 1.0f / sqrtf((float)Q->ne[0]);
    struct ggml_tensor *scores = ggml_mul_mat(ctx, K, Q);
    struct ggml_tensor *mask_2d = gt_build_attention_mask(ctx, scores->ne[0], scores->ne[1], edges);
    struct ggml_tensor *probs = ggml_soft_max_ext(ctx, scores, mask_2d, scale, 0.0f);
    struct ggml_tensor *V_t = ggml_cont(ctx, ggml_transpose(ctx, V));
    return ggml_mul_mat(ctx, V_t, probs);
}

struct ggml_tensor *gt_attention_forward(
    struct ggml_context *ctx,
    struct ggml_tensor *x,
    const gt_attn_weights *weights,
    int n_heads,
    int d_head,
    const gt_edge_data *edges) {
    (void)n_heads;
    (void)d_head;

    struct ggml_tensor *Q = ggml_mul_mat(ctx, weights->W_q, x);
    struct ggml_tensor *K = ggml_mul_mat(ctx, weights->W_k, x);
    struct ggml_tensor *V = ggml_mul_mat(ctx, weights->W_v, x);

    struct ggml_tensor *context = gt_graph_attention_kernel(ctx, Q, K, V, edges);
    return ggml_mul_mat(ctx, weights->W_o, context);
}
