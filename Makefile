CXX = g++
CXXFLAGS = -std=c++17 -Wall -Iinclude -I. -O3 -march=native
LDFLAGS = -lprotobuf

# Gather all source files
ENGINE_SRCS = $(wildcard src/*.cpp)
TEST_SRCS = $(wildcard tests/*.cpp)
PROTO_SRC = include/onnx.proto3.pb.cc

# Generate object file paths
ENGINE_OBJS = $(patsubst src/%.cpp, bin/%.o, $(ENGINE_SRCS))
TEST_OBJS = $(patsubst tests/%.cpp, bin/%.o, $(TEST_SRCS))
PROTO_OBJ = bin/onnx.proto3.pb.o

OBJS = $(ENGINE_OBJS) $(TEST_OBJS) $(PROTO_OBJ)

TARGET = bin/onnx_engine

all: $(TARGET)

bin/%.o: src/%.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/%.o: tests/%.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/onnx.proto3.pb.o: include/onnx.proto3.pb.cc
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

clean:
	rm -rf bin
