default: run

build_dir:
	@mkdir -p build

compile: build_dir
	@cmake -B build -S .
	@cmake --build build

run: compile
	@./build/c_graph_transformer

clean:
	@rm -rf build
	@echo "Build directory nuked."
