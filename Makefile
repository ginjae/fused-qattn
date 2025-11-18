TARGET = cuda_healthcheck
SRC    = cuda_healthcheck.cu

all: $(TARGET)

$(TARGET): $(SRC)
	nvcc -o $(TARGET) $(SRC)

run: $(TARGET)
	CUDA_VISIBLE_DEVICES=0 ./$(TARGET)

clean:
	rm -f $(TARGET)
