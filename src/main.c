#include "graph_transformer/block.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <stdbool.h>
#include <stdint.h>
#include <stdio.h>
#include <string.h>

enum {
    GT_NODES = 3,
    GT_D_MODEL = 8,
    GT_D_FF = 16,
    GT_N_HEADS = 2,
    GT_D_HEAD = GT_D_MODEL / GT_N_HEADS
};

static void fill_tensor_linear(struct ggml_tensor *tensor, float base, float step) {
    float *data = (float *)tensor->data;
    const int64_t total = ggml_nelements(tensor);
    for (int64_t idx = 0; idx < total; ++idx) {
        data[idx] = base + step * (float)idx;
    }
}

static void fill_tensor_constant(struct ggml_tensor *tensor, float value) {
    float *data = (float *)tensor->data;
    const int64_t total = ggml_nelements(tensor);
    for (int64_t idx = 0; idx < total; ++idx) {
        data[idx] = value;
    }
}

static void print_tensor_2d(const struct ggml_tensor *tensor, const char *name, int max_rows) {
    const float *data = (const float *)tensor->data;
    const int cols = (int)tensor->ne[0];
    const int rows = (int)tensor->ne[1];
    const int shown_rows = rows < max_rows ? rows : max_rows;

    printf("%s shape=[%d, %d]\n", name, cols, rows);
    for (int row = 0; row < shown_rows; ++row) {
        printf("row %d: ", row);
        for (int col = 0; col < cols; ++col) {
            printf("% .4f ", data[row * cols + col]);
        }
        printf("\n");
    }
}

int main(void) {
    const int32_t src_edges[] = {0, 1, 2, 2};
    const int32_t dst_edges[] = {1, 2, 0, 1};
    const gt_edge_data edges = {
        .num_nodes = GT_NODES,
        .num_edges = (int32_t)(sizeof(src_edges) / sizeof(src_edges[0])),
        .src = src_edges,
        .dst = dst_edges,
    };

    const size_t ctx_size = 64 * 1024 * 1024;
    struct ggml_init_params params = {
        .mem_size = ctx_size,
        .mem_buffer = NULL,
        .no_alloc = false,
    };

    struct ggml_context *ctx = ggml_init(params);
    if (ctx == NULL) {
        fprintf(stderr, "failed to create ggml context\n");
        return 1;
    }

    struct ggml_tensor *x = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GT_D_MODEL, GT_NODES);
    fill_tensor_linear(x, -0.25f, 0.03f);

    gt_block_weights weights;
    weights.attn.W_q = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GT_D_MODEL, GT_D_MODEL);
    weights.attn.W_k = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GT_D_MODEL, GT_D_MODEL);
    weights.attn.W_v = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GT_D_MODEL, GT_D_MODEL);
    weights.attn.W_o = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GT_D_MODEL, GT_D_MODEL);
    weights.ffn.W1 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GT_D_MODEL, GT_D_FF);
    weights.ffn.W2 = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GT_D_FF, GT_D_MODEL);
    weights.ln1.gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, GT_D_MODEL);
    weights.ln1.beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, GT_D_MODEL);
    weights.ln2.gamma = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, GT_D_MODEL);
    weights.ln2.beta = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, GT_D_MODEL);
    weights.ln1.eps = 1e-5f;
    weights.ln2.eps = 1e-5f;

    fill_tensor_linear(weights.attn.W_q, -0.05f, 0.002f);
    fill_tensor_linear(weights.attn.W_k, 0.03f, -0.001f);
    fill_tensor_linear(weights.attn.W_v, 0.01f, 0.0015f);
    fill_tensor_linear(weights.attn.W_o, -0.02f, 0.0008f);
    fill_tensor_linear(weights.ffn.W1, 0.00f, 0.001f);
    fill_tensor_linear(weights.ffn.W2, 0.00f, -0.0007f);
    fill_tensor_constant(weights.ln1.gamma, 1.0f);
    fill_tensor_constant(weights.ln2.gamma, 1.0f);
    fill_tensor_constant(weights.ln1.beta, 0.0f);
    fill_tensor_constant(weights.ln2.beta, 0.0f);

    struct ggml_tensor *out = gt_block_forward(ctx, x, &weights, GT_N_HEADS, GT_D_HEAD, &edges);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    print_tensor_2d(out, "block_out", GT_NODES);

    ggml_free(ctx);
    return 0;
}
