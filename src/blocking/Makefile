CXX_FLAGS = -O3 -march=native -mtune=native -flto -fuse-linker-plugin --std=c++2a
LINKER_FLAGS = -lbenchmark -lpthread

benchmark : blocking_gemm.o blocking_parallel_gemm.o chunked_gemm.o
	g++ benchmark.cpp blocking_gemm.o blocking_parallel_gemm.o chunked_gemm.o ${CXX_FLAGS} ${LINKER_FLAGS} -o gemm_bench.out

blocking_parallel_gemm.o : blocking_parallel_gemm.cpp
	g++ -c blocking_parallel_gemm.cpp ${CXX_FLAGS}

blocking_gemm.o : blocking_gemm.cpp
	g++ -c blocking_gemm.cpp ${CXX_FLAGS}

chunked_gemm.o : chunked_gemm.cpp
	g++ -c chunked_gemm.cpp ${CXX_FLAGS}

clean :
	rm -rf gemm_bench.out
	rm -rf *.o