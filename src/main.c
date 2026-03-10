#include "ggml.h"
#include "ggml-cpu.h"
typedef struct {
	struct ggml_tensor *W_q; // [d_model, d_model]
	struct ggml_tensor *W_k;
	struct ggml_tensor *W_v;
	struct ggml_tensor *W_o;
} gt_attn_weights ;

typedef struct {
    struct ggml_tensor *W1;  // [d_ff, d_model]
    struct ggml_tensor *W2;  // [d_model, d_ff]
} gt_ffn_weights;

typedef struct {
    struct ggml_tensor *gamma; // [d_model]
    struct ggml_tensor *beta;  // [d_model]
    float eps;
} gt_ln_weights;

typedef struct {
    gt_attn_weights attn;
    gt_ffn_weights  ffn;
    gt_ln_weights   ln1;
    gt_ln_weights   ln2;
} gt_block_weights;

typedef struct {
    int n_layers;
    int n_heads;
    int d_model;
    int d_head;   // d_model / n_heads

    gt_block_weights *blocks;
} gt_model;

static struct ggml_tensor * gt_ffn_forward(
    struct ggml_context * ctx,
    struct ggml_tensor  * x,   // [d_model, N]
    const gt_ffn_weights * w   // W1, W2
) {
    struct ggml_tensor * h =
        ggml_mul_mat(ctx, w->W1, x);

    h = ggml_gelu(ctx, h);

    struct ggml_tensor * y =
        ggml_mul_mat(ctx, w->W2, h);

    return y;
}

static struct ggml_tensor * gt_ln_forward(
    struct ggml_context * ctx,
    struct ggml_tensor * x,
    const gt_ln_weights * w
) {
    struct ggml_tensor * mean =
        ggml_mean(ctx, x);

    struct ggml_tensor * x_centered =
        ggml_sub(ctx, x, mean);

    struct ggml_tensor * sq =
        ggml_sqr(ctx, x_centered);

    struct ggml_tensor * var =
        ggml_mean(ctx, sq);

    struct ggml_tensor * eps =
        ggml_new_tensor_1d(ctx, GGML_TYPE_F32, 1);
    ggml_set_f32_1d(eps, 0, w->eps);

    struct ggml_tensor * var_eps =
        ggml_add(ctx, var, eps);

    struct ggml_tensor * std =
        ggml_sqrt(ctx, var_eps);

    struct ggml_tensor * norm =
        ggml_div(ctx, x_centered, std);

    struct ggml_tensor * scaled =
        ggml_mul(ctx, norm, w->gamma);

    struct ggml_tensor * out =
        ggml_add(ctx, scaled, w->beta);

    return out;
}

typedef struct {
    int32_t num_nodes;
    int32_t num_edges;

    const int32_t * src;   // [E]
    const int32_t * dst;   // [E]
} edge_data;

static struct ggml_tensor * gt_attention(
    struct ggml_context        * ctx,
    struct ggml_tensor         * x,        // [d_model, N]
    const gt_attn_weights      * attn,     // W_q, W_k, W_v, W_o
    int                          n_heads,
    int                          d_head,
    const edge_data            * edges
) {
}
int main(void) {

	return 0;
}
