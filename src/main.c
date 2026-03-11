#include "ggml.h"
#include "ggml-cpu.h"
#include <math.h>
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

typedef struct {
    int32_t num_nodes;
    int32_t num_edges;

    const int32_t * src;   // [E]
    const int32_t * dst;   // [E]
} edge_data;

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
    struct ggml_tensor * norm =
        ggml_norm(ctx, x, w->eps);

    struct ggml_tensor * gamma =
        ggml_repeat(ctx, w->gamma, norm);
    struct ggml_tensor * beta =
        ggml_repeat(ctx, w->beta, norm);

    struct ggml_tensor * scaled =
        ggml_mul(ctx, norm, gamma);

    struct ggml_tensor * out =
        ggml_add(ctx, scaled, beta);

    return out;
}

static struct ggml_tensor * gt_build_attention_mask(
    struct ggml_context * ctx,
    int64_t n_nodes,
    const edge_data * edges
) {
    struct ggml_tensor * mask =
        ggml_new_tensor_2d(ctx, GGML_TYPE_F32, n_nodes, n_nodes);

    float * data = (float *) mask->data;
    const int64_t total = n_nodes * n_nodes;

    for (int64_t index = 0; index < total; ++index) {
        data[index] = -1e9f;
    }

    for (int64_t node = 0; node < n_nodes; ++node) {
        data[node * n_nodes + node] = 0.0f;
    }

    if (edges != NULL && edges->src != NULL && edges->dst != NULL) {
        for (int32_t edge = 0; edge < edges->num_edges; ++edge) {
            const int32_t src = edges->src[edge];
            const int32_t dst = edges->dst[edge];

            if (src >= 0 && src < n_nodes && dst >= 0 && dst < n_nodes) {
                data[dst * n_nodes + src] = 0.0f;
            }
        }
    }

    return mask;
}

static struct ggml_tensor * gt_graph_attention_kernel(
    struct ggml_context *ctx,
    struct ggml_tensor *Qh, // [d_head, n_heads, N]
    struct ggml_tensor *Kh, // [d_head, n_heads, N]
    struct ggml_tensor *Vh, // [d_head, n_heads, N]
    const edge_data *edges  
){
    const float d_head = (float) Qh->ne[0];
    struct ggml_tensor * scores = ggml_mul_mat(ctx, Kh, Qh);
    struct ggml_tensor * mask_2d =
        gt_build_attention_mask(ctx, Qh->ne[2], edges);
    struct ggml_tensor * probs =
        ggml_soft_max_ext(ctx, scores, mask_2d, 1.0f / sqrtf(d_head), 0.0f);

    struct ggml_tensor * V_weighted = ggml_mul_mat(ctx, Vh, probs);

    return V_weighted;
}

static struct ggml_tensor * gt_attention(
    struct ggml_context        * ctx,
    struct ggml_tensor         * x,        // [d_model, N]
    const gt_attn_weights      * attn,     // W_q, W_k, W_v, W_o
    int                          n_heads,
    int                          d_head,
    const edge_data            * edges
) {
	struct ggml_tensor *Q = ggml_mul_mat(ctx, attn->W_q, x);
	struct ggml_tensor *K = ggml_mul_mat(ctx, attn->W_k, x);
	struct ggml_tensor *V = ggml_mul_mat(ctx, attn->W_v, x);

	struct ggml_tensor * Qh =
	    ggml_reshape_3d(ctx, Q, d_head, n_heads, x->ne[1]);
	struct ggml_tensor * Kh =
	    ggml_reshape_3d(ctx, K, d_head, n_heads, x->ne[1]);
	struct ggml_tensor * Vh =
	    ggml_reshape_3d(ctx, V, d_head, n_heads, x->ne[1]);

    struct ggml_tensor * context =
        gt_graph_attention_kernel(ctx, Qh, Kh, Vh, edges);

    struct ggml_tensor * context_2d =
        ggml_reshape_2d(ctx, context, d_head * n_heads, x->ne[1]);

    struct ggml_tensor * out =
        ggml_mul_mat(ctx, attn->W_o, context_2d);

    return out;
}

int main(void) {

	return 0;
}
