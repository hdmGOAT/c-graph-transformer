#include "ggml.h"
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



int main(void) {

	return 0;
}
