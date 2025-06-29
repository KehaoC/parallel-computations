CUDNN_INCLUDE = $(CONDA_PREFIX)/include
CUDNN_LIB = $(CONDA_PREFIX)/lib

CC = nvcc
CFLAGS = -O3 -std=c++11 -I$(CUDNN_INCLUDE)
LDFLAGS = -L$(CUDNN_LIB) -lcudnn

TARGET = conv_test
SRCS = main.cu direct_conv.cu im2col_conv.cu cudnn_conv.cu

all: $(TARGET)

$(TARGET): $(SRCS)
	$(CC) $(CFLAGS) -o $@ $^ $(LDFLAGS) -arch=native

clean:
	rm -f $(TARGET)

.PHONY: all clean