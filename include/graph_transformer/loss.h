#ifndef GRAPH_TRANSFORMER_LOSS_H
#define GRAPH_TRANSFORMER_LOSS_H

#include "ggml.h"

float gt_cross_entropy_loss_row(
    const struct ggml_tensor *logits,
    int sample_index,
    int target_class);

float gt_cross_entropy_loss_mean(
    const struct ggml_tensor *logits,
    const int *targets,
    int n_targets);

#endif
