CHECK_TARGET = cuda_healthcheck
CHECK_SRC    = cuda_healthcheck.cu

EVAL_TARGET = eval
EVAL_SRC    = src/eval.cu src/attn_naive.cu src/attn_tiled.cu src/attn_flash.cu src/attn_ours.cu src/attn_full.cu src/quantization_utils.cu

all: $(CHECK_TARGET) $(EVAL_TARGET)

$(CHECK_TARGET): $(CHECK_SRC)
	nvcc -o $(CHECK_TARGET) $(CHECK_SRC)

$(EVAL_TARGET): $(EVAL_SRC)
	nvcc -o $(EVAL_TARGET) $(EVAL_SRC) -Isrc
	
run_check: $(CHECK_TARGET)
	CUDA_VISIBLE_DEVICES=0 ./$(CHECK_TARGET)

run_eval: $(EVAL_TARGET)
	CUDA_VISIBLE_DEVICES=0 ./$(EVAL_TARGET)

clean:
	rm -f $(CHECK_TARGET) $(EVAL_TARGET)