#include "graph_transformer/loss.h"

#include <math.h>

float gt_cross_entropy_loss_row(
    const struct ggml_tensor *logits,
    int sample_index,
    int target_class) {
    if (logits == NULL || logits->type != GGML_TYPE_F32) {
        return -1.0f;
    }

    const int n_classes = (int)logits->ne[0];
    const int n_samples = (int)logits->ne[1];

    if (sample_index < 0 || sample_index >= n_samples) {
        return -1.0f;
    }
    if (target_class < 0 || target_class >= n_classes) {
        return -1.0f;
    }

    const float *data = (const float *)logits->data;
    const float *sample_logits = data + sample_index * n_classes;

    float max_logit = sample_logits[0];
    for (int class_idx = 1; class_idx < n_classes; ++class_idx) {
        if (sample_logits[class_idx] > max_logit) {
            max_logit = sample_logits[class_idx];
        }
    }

    float sum_exp = 0.0f;
    for (int class_idx = 0; class_idx < n_classes; ++class_idx) {
        sum_exp += expf(sample_logits[class_idx] - max_logit);
    }

    const float log_prob_target =
        (sample_logits[target_class] - max_logit) - logf(sum_exp);
    return -log_prob_target;
}

float gt_cross_entropy_loss_mean(
    const struct ggml_tensor *logits,
    const int *targets,
    int n_targets) {
    if (targets == NULL || n_targets <= 0) {
        return -1.0f;
    }

    float total = 0.0f;
    for (int sample_idx = 0; sample_idx < n_targets; ++sample_idx) {
        const float sample_loss =
            gt_cross_entropy_loss_row(logits, sample_idx, targets[sample_idx]);
        if (sample_loss < 0.0f) {
            return -1.0f;
        }
        total += sample_loss;
    }

    return total / (float)n_targets;
}
