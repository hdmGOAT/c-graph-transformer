#include "graph_transformer/block.h"

#include "graph_transformer/attention.h"
#include "graph_transformer/ffn.h"
#include "graph_transformer/layer_norm.h"

struct ggml_tensor *gt_block_forward(
    struct ggml_context *ctx,
    struct ggml_tensor *x,
    const gt_block_weights *weights,
    int n_heads,
    int d_head,
    const gt_edge_data *edges) {
    struct ggml_tensor *x_norm_1 = gt_ln_forward(ctx, x, &weights->ln1);
    struct ggml_tensor *attn_out = gt_attention_forward(ctx, x_norm_1, &weights->attn, n_heads, d_head, edges);
    struct ggml_tensor *x_res_1 = ggml_add(ctx, x, attn_out);

    struct ggml_tensor *x_norm_2 = gt_ln_forward(ctx, x_res_1, &weights->ln2);
    struct ggml_tensor *ffn_out = gt_ffn_forward(ctx, x_norm_2, &weights->ffn);
    return ggml_add(ctx, x_res_1, ffn_out);
}
