#include "graph_transformer/block.h"
#include "graph_transformer/head.h"

#include "ggml-cpu.h"
#include "ggml.h"

#include <stdbool.h>
#include <ctype.h>
#include <stdint.h>
#include <stdlib.h>
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

static bool load_citeseer_first_label(const char *path, char *label_out, size_t label_out_size) {
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        return false;
    }

    char line[65536];
    if (fgets(line, (int)sizeof(line), file) == NULL) {
        fclose(file);
        return false;
    }
    fclose(file);

    char *newline = strpbrk(line, "\r\n");
    if (newline != NULL) {
        *newline = '\0';
    }

    size_t len = strlen(line);
    if (len == 0) {
        return false;
    }

    size_t end = len;
    while (end > 0 && isspace((unsigned char)line[end - 1])) {
        --end;
    }
    if (end == 0) {
        return false;
    }

    size_t start = end;
    while (start > 0 && !isspace((unsigned char)line[start - 1])) {
        --start;
    }
    if (start == end) {
        return false;
    }

    const size_t token_len = end - start;
    if (token_len + 1 > label_out_size) {
        return false;
    }

    memcpy(label_out, line + start, token_len);
    label_out[token_len] = '\0';
    return true;
}

static bool load_mutag_first_label(const char *path, int *label_out) {
    FILE *file = fopen(path, "r");
    if (file == NULL) {
        return false;
    }

    char line[65536];
    if (fgets(line, (int)sizeof(line), file) == NULL) {
        fclose(file);
        return false;
    }
    fclose(file);

    char *marker = strstr(line, "\"y\": [");
    if (marker == NULL) {
        return false;
    }

    marker += strlen("\"y\": [");
    *label_out = (int)strtol(marker, NULL, 10);
    return true;
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

    gt_task_head_weights node_head;
    node_head.W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GT_D_MODEL, 6);
    node_head.b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 6);
    fill_tensor_linear(node_head.W, -0.02f, 0.001f);
    fill_tensor_constant(node_head.b, 0.01f);

    gt_task_head_weights graph_head;
    graph_head.W = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, GT_D_MODEL, 2);
    graph_head.b = ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 2);
    fill_tensor_linear(graph_head.W, 0.03f, -0.001f);
    fill_tensor_constant(graph_head.b, -0.02f);

    struct ggml_tensor *node_logits = gt_node_head_forward(ctx, out, &node_head);
    struct ggml_tensor *graph_logits = gt_graph_head_forward(ctx, out, &graph_head);

    struct ggml_cgraph *gf = ggml_new_graph(ctx);
    ggml_build_forward_expand(gf, out);
    ggml_build_forward_expand(gf, node_logits);
    ggml_build_forward_expand(gf, graph_logits);
    ggml_graph_compute_with_ctx(ctx, gf, 1);

    print_tensor_2d(out, "block_out", GT_NODES);
    print_tensor_2d(node_logits, "node_logits (CiteSeer-like, 6 classes)", GT_NODES);
    print_tensor_2d(graph_logits, "graph_logits (MUTAG-like, 2 classes)", 1);

    char citeseer_label[64] = {0};
    int mutag_label = -1;

    const bool citeseer_ok = load_citeseer_first_label(
        "data/raw/citeseer/citeseer/citeseer.content",
        citeseer_label,
        sizeof(citeseer_label));
    const bool mutag_ok = load_mutag_first_label("data/raw/MUTAG/full.jsonl", &mutag_label);

    if (citeseer_ok) {
        printf("CiteSeer first paper label: %s\n", citeseer_label);
    } else {
        printf("CiteSeer first paper label: unavailable (dataset missing?)\n");
    }

    if (mutag_ok) {
        printf("MUTAG first graph label: %d\n", mutag_label);
    } else {
        printf("MUTAG first graph label: unavailable (dataset missing?)\n");
    }

    ggml_free(ctx);
    return 0;
}
