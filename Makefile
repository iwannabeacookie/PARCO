# Compiler
# CC = gcc
CC = mpicc
CFLAGS = -Iinclude -g -fopenmp -O2 -MMD -MP -ftree-vectorize -march=native -flto -lmpi

# If gcc-9.1.0 is available, use it
ifneq ($(shell which gcc-9.1.0),)
CC = gcc-9.1.0
endif

# Directories
SRC_DIR = src
BUILD_DIR = build
BIN_DIR = bin
INCLUDE_DIR = include

# Files
SOURCES = $(wildcard $(SRC_DIR)/*.c)
OBJECTS = $(patsubst $(SRC_DIR)/%.c,$(BUILD_DIR)/%.o,$(SOURCES))
TARGET = $(BIN_DIR)/out

-include $(OBJECTS:.o=.d)

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
.PHONY: clean all verbose debug
clean:
	rm -rf $(BUILD_DIR)/ $(TARGET)

all: $(TARGET)

verbose: CFLAGS += -Wall -Wextra -fopt-info-vec-optimized
verbose:
	-which gcc-9.1.0
	$(MAKE) -s $(TARGET)

debug: CFLAGS += -g
debug:
	$(MAKE) -s $(TARGET)
