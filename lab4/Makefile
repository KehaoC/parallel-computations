CC = gcc
CFLAGS = -Wall -O2 -pthread -lm
BUILD_DIR = build
SRC_DIR = src

all: $(BUILD_DIR)/quadratic_solver $(BUILD_DIR)/monte_carlo_pi

$(BUILD_DIR)/quadratic_solver: $(SRC_DIR)/quadratic_solver.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<

$(BUILD_DIR)/monte_carlo_pi: $(SRC_DIR)/monte_carlo_pi.c
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS) -o $@ $<

clean:
	rm -rf $(BUILD_DIR)/*

.PHONY: all clean 