CXX_FLAGS = -O3 -march=native -mtune=native -flto -fuse-linker-plugin --std=c++2a
LINKER_FLAGS = -lbenchmark -lpthread

benchmark : serial_gemm.o parallel_gemm.o 
	g++ benchmark.cpp serial_gemm.o parallel_gemm.o ${CXX_FLAGS} ${LINKER_FLAGS} -o gemm_bench.out

parallel_gemm.o : parallel_gemm.cpp
	g++ -c parallel_gemm.cpp ${CXX_FLAGS}

serial_gemm.o : serial_gemm.cpp
	g++ -c serial_gemm.cpp ${CXX_FLAGS}

clean :
	rm -rf gemm_bench.out
	rm -rf *.o