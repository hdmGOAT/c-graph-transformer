#include "ggml-cpu.h"
#include "ggml.h"
#include <string.h>
#include <stdlib.h>
#include <time.h>
#include <math.h>
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

    	// Okay so i have 3 nodes with features being represented by two floating point values.
    	// We wanna matmul our FEATURE_NUM * N matrix (features by node) to some learnable weight matrix,
	// learnable weight matrix will be of FEATURE_NUM by Embedding length (lets say 16), BUT we should only
	// do that operation towards neighbors, this is apparantly applied by multiplying against ajacency matrix
	// then finally feed into an activation function, this output then becomes a node embedding    
	// Use large context for easyness	
	size_t ctx_size = 8 * 1024 * 1024; 
	// STEP 0: set up ggml	
	
	struct ggml_init_params params = {
		ctx_size, NULL, false
	};	
	
	struct ggml_context *ctx = ggml_init(params);

	// Step 0,5 : allocate tensors
	struct ggml_tensor *feature_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, FN);	
	memcpy(feature_mat->data, features, ggml_nbytes(feature_mat));
	struct ggml_tensor *weight_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, FN, EMBEDDINGS_DEPTH);	

	
	srand((unsigned int)time(NULL));

	float scale = 0.1f;
	float *wdata = (float *)weight_mat->data;
	for (int i = 0; i < FN * EMBEDDINGS_DEPTH; i++) {
	    wdata[i] = ((float)rand() / RAND_MAX * 2.0f - 1.0f) * scale;
	}

	struct ggml_tensor *adj_mat = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, N, N);	
	memcpy(adj_mat->data, adjacency, ggml_nbytes(adj_mat));


	//  byuild graph
	
	struct ggml_cgraph *gf = ggml_new_graph(ctx);

	// AX = A · X
	struct ggml_tensor *ax =
	    ggml_mul_mat(ctx, feature_mat, adj_mat);

	// AXW = (A · X) · W
	struct ggml_tensor *axw =
	    ggml_mul_mat(ctx, weight_mat, ax);

	// activation
	struct ggml_tensor *out =
	    ggml_relu(ctx, axw);

	// build & run graph
	ggml_build_forward_expand(gf, out);
	ggml_graph_compute_with_ctx(ctx, gf, 1);

	// print output (ggml column-major layout)
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
