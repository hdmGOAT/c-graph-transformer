cmake_minimum_required(VERSION 3.16)
project(c_graph_transformer C)

set(CMAKE_C_STANDARD 11)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

add_subdirectory(third_party/ggml)

add_executable(c_graph_transformer
	src/main.c
	src/transformer/attention.c
	src/transformer/head.c
	src/transformer/ffn.c
	src/transformer/layer_norm.c
	src/transformer/block.c
)
target_include_directories(c_graph_transformer PRIVATE
	third_party/ggml/include
	include
)
target_link_libraries(c_graph_transformer PRIVATE ggml m)
