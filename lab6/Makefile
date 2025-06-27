CC = gcc
# CFLAGS_OMP = -fopenmp -O3 -Wall # Commented out OpenMP flags
CFLAGS_PTH = -pthread -O3 -Wall -lm

SRC_DIR = src
BUILD_DIR = build

# TARGET_OMP = $(BUILD_DIR)/heated_plate_openmp # Commented out OpenMP target
TARGET_PTH = $(BUILD_DIR)/heated_plate_pthreads

all: $(TARGET_PTH) # Changed all to only build Pthreads target

# $(TARGET_OMP): $(SRC_DIR)/heated_plate_openmp.c # Commented out OpenMP rule
# 	mkdir -p $(BUILD_DIR)
# 	$(CC) $(CFLAGS_OMP) $^ -o $@

$(TARGET_PTH): $(SRC_DIR)/heated_plate_pthreads.c $(SRC_DIR)/parallel_for.c $(SRC_DIR)/parallel_for.h
	mkdir -p $(BUILD_DIR)
	$(CC) $(CFLAGS_PTH) $(SRC_DIR)/heated_plate_pthreads.c $(SRC_DIR)/parallel_for.c -o $@

.PHONY: all clean

clean:
	rm -rf $(BUILD_DIR)/* 