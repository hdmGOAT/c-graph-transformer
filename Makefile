default: run

build_dir:
	@mkdir -p build

compile: build_dir
	@cmake -B build -S . -DCMAKE_BUILD_TYPE=Debug -DCMAKE_EXPORT_COMPILE_COMMANDS=ON
	@cmake --build build
	@cp -f build/compile_commands.json compile_commands.json

run: compile
	@./build/c_graph_transformer

clean:
	@rm -rf build
	@rm -f compile_commands.json
	@echo "Build directory nuked."

clangd: compile
	@echo "clangd compilation database refreshed."
