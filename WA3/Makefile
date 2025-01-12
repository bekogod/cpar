################################################################################
# Makefile for fluid simulation
#
# by André Pereira
################################################################################

SHELL = /bin/sh
BIN_NAME = fluid_sim

CXX = nvcc
LD  = nvcc

CXXFLAGS   = -O3 -g -std=c++11 -arch=sm_35 -Wno-deprecated-gpu-targets 

SRC_DIR = src
BIN_DIR = bin
BUILD_DIR = build
SRC = $(wildcard $(SRC_DIR)/*.cu) $(wildcard $(SRC_DIR)/*.cpp)
OBJ = $(patsubst src/%.cu,build/%.o,$(patsubst src/%.cpp,build/%.o,$(SRC)))
BIN = $(BIN_NAME)

vpath %.cu $(SRC_DIR)
vpath %.cpp $(SRC_DIR)

################################################################################
# Rules
################################################################################

.DEFAULT_GOAL = all

$(BUILD_DIR)/%.o: %.cu
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

$(BUILD_DIR)/%.o: %.cpp
	$(CXX) -c $(CXXFLAGS) $(INCLUDES) $< -o $@ $(LIBS)

$(BIN_DIR)/$(BIN_NAME): $(OBJ)
	$(CXX) $(CXXFLAGS) $(INCLUDES) -o $@ $(OBJ) $(LIBS)

checkdirs:
	@mkdir -p $(BUILD_DIR)
	@mkdir -p $(BIN_DIR)

all: checkdirs $(BIN_DIR)/$(BIN_NAME)

run:
	sbatch runcuda.sh

clean:
	rm -f $(BUILD_DIR)/* $(BIN_DIR)/* 

