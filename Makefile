IBM_FLAGS = -qsmp=omp
GNU_STANDARD = c++98

bgp: COMP = mpixlcxx_r
bgp: CFLAGS = $(IBM_FLAGS)
gnu: COMP = mpicxx
gnu: CFLAGS = -fopenmp -std=$(GNU_STANDARD)
polus: COMP = mpixlC
polus: CFLAGS = $(IBM_FLAGS)

.PHONY: bgp polus gnu clean
bgp: bgp_bin
gnu: gnu_bin
polus: polus_bin

BINARY_NAME = solver
BUILD_DIR = build
DUMP_DIR = dump
SRC_DIR = src
OBJECT_FILES := $(BUILD_DIR)/main.o

$(BUILD_DIR)/%.o: $(SRC_DIR)/%.cpp $(SRC_DIR)/structures.hpp
	$(COMP) $(CFLAGS) -c -o $@ $<

bgp_bin: $(OBJECT_FILES)
	$(COMP) $(CFLAGS) -o $(BINARY_NAME) $^

gnu_bin: $(OBJECT_FILES)
	$(COMP) $(CFLAGS) -o $(BINARY_NAME) $^

polus_bin: $(OBJECT_FILES)
	$(COMP) $(CFLAGS) -o $(BINARY_NAME) $^

clean:
	rm -f $(BUILD_DIR)/*.o $(DUMP_DIR)/*.csv solver
