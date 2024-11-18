# Variables
CC = gcc
CFLAGS = -Iinclude -Wall -Wextra -g -fopenmp -O3 -fopt-info-vec-optimized
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
INCLUDE_DIR = include
SOURCES = $(SRC_DIR)/main.c $(SRC_DIR)/init_matrix.c $(SRC_DIR)/sequential.c
OBJECTS = $(BUILD_DIR)/main.o $(BUILD_DIR)/init_matrix.o $(BUILD_DIR)/sequential.o
TARGET = $(BIN_DIR)/out

# Create directories if they don't exist
$(BUILD_DIR) $(BIN_DIR):
	@mkdir -p $@

# Rule to build the target executable
$(TARGET): $(BUILD_DIR) $(BIN_DIR) $(OBJECTS)
	$(CC) $(OBJECTS) $(CFLAGS) -o $(TARGET)

# Rule to compile source files into object files
$(BUILD_DIR)/%.o: $(SRC_DIR)/%.c
	echo "Compiling $<"
	$(CC) $(CFLAGS) -c $< -o $@

# Clean rule to remove generated files
.PHONY: clean
clean:
	rm -rf $(BUILD_DIR)/*.o $(TARGET)

.PHONY: all
all: $(TARGET)

# Dependencies
$(BUILD_DIR)/main.o: $(SRC_DIR)/main.c $(INCLUDE_DIR)/init_matrix.h $(INCLUDE_DIR)/sequential.h
$(BUILD_DIR)/init_matrix.o: $(SRC_DIR)/init_matrix.c $(INCLUDE_DIR)/init_matrix.h
$(BUILD_DIR)/sequential.o: $(SRC_DIR)/sequential.c $(INCLUDE_DIR)/sequential.h

