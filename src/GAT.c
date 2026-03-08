
#include "ggml-cpu.h"
#include "ggml.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <stdio.h>
#define E 7
#define N 3
#define FN 2
#define EMBEDDINGS_DEPTH 16
static void init_random_tensor_f32(struct ggml_tensor *tensor, float scale) {
	float *data = (float *)tensor->data;
	const int64_t ne0 = tensor->ne[0];
	const int64_t ne1 = tensor->ne[1];
	const int64_t ne2 = tensor->ne[2];
	const int64_t ne3 = tensor->ne[3];
	const int64_t total = ne0 * ne1 * ne2 * ne3;
	for (int64_t i = 0; i < total; i++) {
		data[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
	}
}
int main(void) {
    	// Simple Graph for now
	float features[N * FN] = {1.0f, 0.0f, 
				0.0f, 1.0f, 
				1.1f, 1.1f};
    
	// Adjacency matrix including self-loops
	float adjacency[N * N] = {
	    1.0f, 1.0f, 0.0f,
	    1.0f, 1.0f, 1.0f,
	    0.0f, 1.0f, 1.0f
	};

	size_t ctx_size = 8 * 1024 * 1024; 
	
	struct ggml_init_params params = {
		ctx_size, NULL, false
	};	
	
	struct ggml_context *ctx = ggml_init(params);

	
	struct ggml_tensor *adj_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);	
	memcpy(adj_mat->data, adjacency, ggml_nbytes(adj_mat));
	struct ggml_tensor *feature_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, FN, N);	
	memcpy(feature_mat->data, features, ggml_nbytes(feature_mat));
	struct ggml_tensor *weight_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, FN, EMBEDDINGS_DEPTH);	

	
	srand((unsigned int)time(NULL));
	init_random_tensor_f32(weight_mat, 0.1f);


	struct ggml_cgraph *gf = ggml_new_graph(ctx);


	struct ggml_tensor *h =
	    ggml_mul_mat(ctx, weight_mat, feature_mat);

	struct ggml_tensor *att_l =
	    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, EMBEDDINGS_DEPTH);

	struct ggml_tensor *att_r =
	    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, EMBEDDINGS_DEPTH);

	init_random_tensor_f32(att_l, 0.1f);
	init_random_tensor_f32(att_r, 0.1f);
	
	struct ggml_tensor *f_i =
	    ggml_mul_mat(ctx, att_l, h);

	struct ggml_tensor *f_j =
	    ggml_mul_mat(ctx, att_r, h);

	struct ggml_tensor *f_i_exp =
	    ggml_repeat(ctx, f_i, adj_mat);

	struct ggml_tensor *f_j_exp =
	    ggml_repeat(ctx, f_j, adj_mat);

	struct ggml_tensor *e =
	    ggml_add(ctx, f_i_exp, f_j_exp);

	struct ggml_tensor *masked = 
	    ggml_mul_mat(ctx, e, adj_mat);

	struct ggml_tensor *alpha =
	    ggml_soft_max(ctx, masked);

	struct ggml_tensor *h_t =
	    ggml_transpose(ctx, h);
	struct ggml_tensor *h_t_cont =
	    ggml_cont(ctx, h_t);

	struct ggml_tensor *agg =
	    ggml_mul_mat(ctx, alpha, h_t_cont);

	struct ggml_tensor *out =
	    ggml_leaky_relu(ctx, agg, 0.2f, false);


	ggml_build_forward_expand(gf, out);
	ggml_graph_compute_with_ctx(ctx, gf, 1);

	float *out_data = (float *) out->data;

	printf("Node embeddings:\n");
	for (int i = 0; i < N; i++) {
	    printf("Node %d: ", i);
	    for (int j = 0; j < EMBEDDINGS_DEPTH; j++) {
		printf("%7.4f ", out_data[j * N + i]);
	    }
	    printf("\n");
	}

	return 0;
}
