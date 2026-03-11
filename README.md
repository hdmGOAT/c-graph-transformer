# c-graph-transformer

Lightweight graph-transformer experiments in C with `ggml`.

## Repository layout

- `src/main.c`: runnable demo wiring a single transformer block.
- `src/transformer/`: modular transformer implementation pieces.
- `include/graph_transformer/`: public headers for model components.
- `reference/`: older standalone reference experiments (`GCN`, `GAT`, matrix ops).
- `third_party/ggml/`: vendored `ggml` dependency.

## Build and run

```bash
cmake -S . -B build
cmake --build build
./build/c_graph_transformer
```
