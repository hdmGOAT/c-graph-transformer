cmake_minimum_required(VERSION 3.16)
project(c_graph_transformer C)

add_subdirectory(third_party/ggml)

add_executable(c_graph_transformer src/main.c)
target_include_directories(c_graph_transformer PRIVATE third_party/ggml/include)
target_link_libraries(c_graph_transformer PRIVATE ggml)
