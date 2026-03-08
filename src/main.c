
#include "ggml-cpu.h"
#include "ggml.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#define E 7
#define N 3
#define FN 2
#define EMBEDDINGS_DEPTH 16
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

	
	struct ggml_tensor *feature_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, FN, N);	
	memcpy(feature_mat->data, features, ggml_nbytes(feature_mat));
	struct ggml_tensor *weight_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, FN, EMBEDDINGS_DEPTH);	

	
	srand((unsigned int)time(NULL));

	float scale = 0.1f;
	float *wdata = (float *)weight_mat->data;
	for (int i = 0; i < FN * EMBEDDINGS_DEPTH; i++) {
	    wdata[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
	}


	struct ggml_cgraph *gf = ggml_new_graph(ctx);


	struct ggml_tensor *h =
	    ggml_mul_mat(ctx, weight_mat, feature_mat);

	struct ggml_tensor *att_l =
	    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, EMBEDDINGS_DEPTH);

	struct ggml_tensor *att_r =
	    ggml_new_tensor_1d(ctx, GGML_TYPE_F32, EMBEDDINGS_DEPTH);

	// Add random init
	
	return 0;
}
