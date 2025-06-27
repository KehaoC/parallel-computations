CC = mpicc
CFLAGS = -O3 -Wall
TARGET = matrix_mult
SRC_DIR = src
BUILD_DIR = build

$(BUILD_DIR)/$(TARGET): $(SRC_DIR)/matrix_mult.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) $< -o $@

.PHONY: clean run

clean:
	rm -rf $(BUILD_DIR)

run:
	mpirun -np 4 ./$(BUILD_DIR)/$(TARGET) 