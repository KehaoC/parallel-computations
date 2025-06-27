CXX = mpicxx
CC = gcc
CFLAGS = -Wall -O2
OMP_COMPILE_FLAGS = -Xpreprocessor -fopenmp
OMP_LINK_FLAGS = -lomp
OMP_INCLUDE_PATH = -I/opt/homebrew/opt/libomp/include
OMP_LIB_PATH = -L/opt/homebrew/opt/libomp/lib
LDFLAGS =

# --- Part 1: MPI FFT ---
FFT_SERIAL_SRC = src/fft_serial.cpp
FFT_SERIAL_OBJ = $(FFT_SERIAL_SRC:.cpp=.o)
FFT_SERIAL_EXEC = results/fft_serial

FFT_MPI_SRC = src/fft_mpi.cpp
FFT_MPI_OBJ = $(FFT_MPI_SRC:.cpp=.o)
FFT_MPI_EXEC = results/fft_mpi

# --- Part 2: parallel_for heated_plate analysis ---
# heated_plate_openmp uses parallel_for.c
PF_SRC = src/parallel_for.c
PF_OBJ = $(PF_SRC:.c=.o)

HP_OMP_SRC = src/heated_plate_openmp.c
HP_OMP_OBJ = $(HP_OMP_SRC:.c=.o)
HP_OMP_EXEC = results/heated_plate_openmp

.PHONY: all clean fft heated_plate

all: fft heated_plate

fft: $(FFT_SERIAL_EXEC) $(FFT_MPI_EXEC)

heated_plate: $(HP_OMP_EXEC)

# Rule for fft_serial
$(FFT_SERIAL_EXEC): $(FFT_SERIAL_OBJ)
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(FFT_SERIAL_OBJ): $(FFT_SERIAL_SRC)
	$(CXX) $(CFLAGS) -c $< -o $@

# Rule for fft_mpi
$(FFT_MPI_EXEC): $(FFT_MPI_OBJ)
	$(CXX) $(CFLAGS) $^ -o $@ $(LDFLAGS)

$(FFT_MPI_OBJ): $(FFT_MPI_SRC)
	$(CXX) $(CFLAGS) -c $< -o $@

# Rule for heated_plate_openmp (links with parallel_for.o)
$(HP_OMP_EXEC): $(HP_OMP_OBJ) $(PF_OBJ)
	$(CC) $(CFLAGS) $(OMP_COMPILE_FLAGS) $(OMP_INCLUDE_PATH) $^ -o $@ $(LDFLAGS) $(OMP_LIB_PATH) -lm $(OMP_LINK_FLAGS)

$(HP_OMP_OBJ): $(HP_OMP_SRC) src/parallel_for.h
	$(CC) $(CFLAGS) $(OMP_COMPILE_FLAGS) $(OMP_INCLUDE_PATH) -c $< -o $@

$(PF_OBJ): $(PF_SRC) src/parallel_for.h
	$(CC) $(CFLAGS) -c $< -o $@


clean:
	rm -f src/*.o results/*
	rm -f massif.out.* # Clean Valgrind outputs 