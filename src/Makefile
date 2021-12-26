CXX = clang++
CXXFLAGS = -std=c++20 -Wall
SYCLFLAGS = -fsycl
SYCLCUDAFLAGS = -fsycl-targets=nvptx64-nvidia-cuda
SYCLAOTFLAGS = -fsycl-default-sub-group-size 32
INCLUDES = -I./include
PROG = run
DFLAGS = -D$(shell echo $(or $(DEVICE),default) | tr a-z A-Z)

$(PROG): main.o utils.o bench_p64_ctbn.o bench_p254_ctbn.o
	$(CXX) $(SYCLFLAGS) $^ -o $@

main.o: main.cpp include/bench_p64_ctbn.hpp include/bench_p254_ctbn.hpp include/types.hpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

utils.o: utils.cpp include/utils.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

bench_p64_ctbn.o: bench_p64_ctbn.cpp include/bench_p64_ctbn.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

bench_p254_ctbn.o: bench_p254_ctbn.cpp include/bench_p254_ctbn.hpp
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(DFLAGS) -c $^ $(INCLUDES)

clean:
	find . -name '*.o' -o -name 'run' -o -name 'a.out' -o -name '*.gch' -o -name 'test' -o  -name '__pycache__' | xargs rm -rf

format:
	find . -name '*.cpp' -o -name '*.hpp' | xargs clang-format -i

aot_cpu:
	$(CXX) $(CXXFLAGS) $(DFLAGS) $(SYCLFLAGS) -c main.cpp -o main.o $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) -c utils.cpp -o utils.o $(INCLUDES)
	@if lscpu | grep -q 'avx512'; then \
		echo "Using avx512"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLAOTFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx512" bench_p64_ctbn.cpp bench_p254_ctbn.cpp utils.o main.o; \
	elif lscpu | grep -q 'avx2'; then \
		echo "Using avx2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLAOTFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx2" bench_p64_ctbn.cpp bench_p254_ctbn.cpp utils.o main.o; \
	elif lscpu | grep -q 'avx'; then \
		echo "Using avx"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLAOTFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=avx" bench_p64_ctbn.cpp bench_p254_ctbn.cpp utils.o main.o; \
	elif lscpu | grep -q 'sse4.2'; then \
		echo "Using sse4.2"; \
		$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLAOTFLAGS) $(INCLUDES) -fsycl-targets=spir64_x86_64-unknown-unknown-sycldevice -Xs "-march=sse4.2" bench_p64_ctbn.cpp bench_p254_ctbn.cpp utils.o main.o; \
	else \
		echo "Can't AOT compile using avx, avx2, avx512 or sse4.2"; \
	fi

cuda:
	# make sure you've built `clang++` with CUDA support
	# check https://intel.github.io/llvm-docs/GetStartedGuide.html#build-dpc-toolchain-with-support-for-nvidia-cuda
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLCUDAFLAGS) $(DFLAGS) -c main.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLCUDAFLAGS) $(DFLAGS) -c utils.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLCUDAFLAGS) $(DFLAGS) -c bench_p64_ctbn.cpp $(INCLUDES)
	$(CXX) $(CXXFLAGS) $(SYCLFLAGS) $(SYCLCUDAFLAGS) $(DFLAGS) -c bench_p254_ctbn.cpp $(INCLUDES)
	$(CXX) $(SYCLFLAGS) *.o -o $(PROG)
