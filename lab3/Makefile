CC = gcc
CFLAGS = -Wall -O3 -pthread
BUILD_DIR = build
SRC_DIR = src

all: $(BUILD_DIR)/matrix_multiply $(BUILD_DIR)/array_sum

$(BUILD_DIR)/matrix_multiply: $(SRC_DIR)/matrix_multiply.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD_DIR)/array_sum: $(SRC_DIR)/array_sum.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean 