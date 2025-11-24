CHECK_TARGET = cuda_healthcheck
CHECK_SRC    = cuda_healthcheck.cu

NAIVE_TARGET = attn_naive
NAIVE_SRC    = src/attn_naive.cu src/quantization_utils.cu

all: $(CHECK_TARGET) $(NAIVE_TARGET)

$(CHECK_TARGET): $(CHECK_SRC)
	nvcc -o $(CHECK_TARGET) $(CHECK_SRC)

$(NAIVE_TARGET): $(NAIVE_SRC)
	nvcc -o $(NAIVE_TARGET) $(NAIVE_SRC) -Isrc

run_check: $(CHECK_TARGET)
	CUDA_VISIBLE_DEVICES=0 ./$(CHECK_TARGET)

run_naive: $(NAIVE_TARGET)
	CUDA_VISIBLE_DEVICES=0 ./$(NAIVE_TARGET)

clean:
	rm -f $(CHECK_TARGET) $(NAIVE_TARGET)