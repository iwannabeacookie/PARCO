# Compiler
CC = gcc
CFLAGS = -Iinclude -g -fopenmp -O2

# If gcc-9.1.0 is available, use it
ifneq ($(shell which gcc-9.1.0 > /dev/null 2>&1),)
CC = gcc-9.1.0
endif

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
INCLUDE_DIR = include

# Files
SOURCES = $(SRC_DIR)/main.c $(SRC_DIR)/init_matrix.c $(SRC_DIR)/sequential.c $(SRC_DIR)/omp_parallel.c $(SRC_DIR)/utils.c $(SRC_DIR)/implicit_parallel.c
OBJECTS = $(BUILD_DIR)/main.o $(BUILD_DIR)/init_matrix.o $(BUILD_DIR)/sequential.o $(BUILD_DIR)/omp_parallel.o $(BUILD_DIR)/utils.o $(BUILD_DIR)/implicit_parallel.o
TARGET = $(BIN_DIR)/out

# Create directories if they don't exist
$(BUILD_DIR) $(BIN_DIR):
	@mkdir -p $@

# Rule to build the target executable
$(TARGET): $(BUILD_DIR) $(BIN_DIR) $(OBJECTS)
	@echo "Linking $@"
	@$(CC) $(OBJECTS) $(CFLAGS) -o $(TARGET)
	@echo "Done! ^~^"

# Rule to compile source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	@echo "Compiling $<"
	@$(CC) $(CFLAGS) -c $< -o $@

# Clean rule to remove generated files
.PHONY: clean all verbose
clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)

all: $(TARGET)

verbose: CFLAGS += -Wall -Wextra -fopt-info-vec-optimized
verbose:
	-which gcc-9.1.0
	$(MAKE) -s $(TARGET)
	
# Dependencies
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.c $(INCLUDE_DIR)/init_matrix.h $(INCLUDE_DIR)/sequential.h $(INCLUDE_DIR)/omp_parallel.h $(INCLUDE_DIR)/utils.h $(INCLUDE_DIR)/config.h $(INCLUDE_DIR)/implicit_parallel.h
$(BUILD_DIR)/init_matrix.o: $(SRC_DIR)/init_matrix.c $(INCLUDE_DIR)/init_matrix.h $(INCLUDE_DIR)/config.h
$(BUILD_DIR)/sequential.o: $(SRC_DIR)/sequential.c $(INCLUDE_DIR)/sequential.h $(INCLUDE_DIR)/config.h
$(BUILD_DIR)/omp_parallel.o: $(SRC_DIR)/omp_parallel.c $(INCLUDE_DIR)/omp_parallel.h $(INCLUDE_DIR)/config.h
$(BUILD_DIR)/implicit_parallel.o: $(SRC_DIR)/implicit_parallel.c $(INCLUDE_DIR)/implicit_parallel.h $(INCLUDE_DIR)/config.h
$(BUILD_DIR)/utils.o: $(SRC_DIR)/utils.c $(INCLUDE_DIR)/utils.h $(INCLUDE_DIR)/config.h
