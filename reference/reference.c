#include "ggml-cpu.h"
#include "ggml.h"
#include <stdbool.h>
#include <stdio.h>
#include <string.h>

enum {
    ROWS_A = 4,
    COLS_A = 2,
    ROWS_B = 4,
    COLS_B = 2
};

int main(void) {

	float matrix_A[ROWS_A * COLS_A] = {
		2, 8,
		5, 1,
		4, 2,
		8, 6
	};


	float matrix_B[ROWS_B * COLS_B] = {
		2, 8,
		5, 1,
		4, 2,
		8, 6
	};
	
	size_t ctx_size = 0;

	ctx_size += ROWS_A * COLS_A * ggml_type_size(GGML_TYPE_F32);

	ctx_size += ROWS_B * COLS_B * ggml_type_size(GGML_TYPE_F32);

	ctx_size += ROWS_A * ROWS_B * ggml_type_size(GGML_TYPE_F32);

	ctx_size += 3 * ggml_tensor_overhead();
	ctx_size += ggml_graph_overhead();
	ctx_size += 1024;

	struct ggml_init_params params = {
		ctx_size, NULL, false
	};
	struct ggml_context *ctx = ggml_init(params);
	struct ggml_tensor *tensor_a = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, COLS_A, ROWS_A);

	struct ggml_tensor *tensor_b = ggml_new_tensor_2d(ctx, GGML_TYPE_F32, COLS_B, ROWS_B);

	memcpy(tensor_a->data, matrix_A, ggml_nbytes(tensor_a));
	memcpy(tensor_b->data, matrix_B, ggml_nbytes(tensor_b));

	struct ggml_cgraph *gf = ggml_new_graph(ctx);

	struct ggml_tensor *result = ggml_mul_mat(ctx, tensor_a, tensor_b);

	ggml_build_forward_expand(gf, result);

	ggml_graph_compute_with_ctx(ctx, gf, 1);

	float * result_data = (float *) result->data;
    	printf("mul mat (%d x %d) (transposed result):\n[", (int) result->ne[0], (int) result->ne[1]);
    	for (int j = 0; j < result->ne[1]; j++) {
        	if (j > 0) {
            	printf("\n");
        }

        for (int i = 0; i < result->ne[0]; i++) {
        	printf(" %.2f", result_data[j * result->ne[0] + i]);
        }
    }
    printf(" ]\n");

    ggml_free(ctx);
    return 0;
}
