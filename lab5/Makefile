CC = gcc
CFLAGS = -O3 -Wall
LDFLAGS = -pthread
TARGETS = matrix_mult_pthread
SRC_DIR = src
BUILD_DIR = build

all: $(TARGETS)

matrix_mult_pthread: $(SRC_DIR)/matrix_mult_pthread.c $(SRC_DIR)/parallel_for.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $^ -o $(BUILD_DIR)/$@ $(LDFLAGS)

.PHONY: clean

clean:
	rm -rf $(BUILD_DIR) 