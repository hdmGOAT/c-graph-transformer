#ifndef GRAPH_TRANSFORMER_HEAD_H
#define GRAPH_TRANSFORMER_HEAD_H

#include "graph_transformer/types.h"

struct ggml_tensor *gt_node_head_forward(
    struct ggml_context *ctx,
    struct ggml_tensor *x,
    const gt_task_head_weights *weights);

struct ggml_tensor *gt_graph_head_forward(
	struct ggml_context *ctx,
	struct ggml_tensor *x,
	const gt_task_head_weights *weights
);

#endif
