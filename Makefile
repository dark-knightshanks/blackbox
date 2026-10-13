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

TARGET_MNIST = bin/test_mnist
TARGET_RESNET = bin/test_resnet18
TARGET_VIT = bin/test_vit

all: $(TARGET_MNIST) $(TARGET_RESNET)

bin/%.o: src/%.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/%.o: tests/%.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

bin/onnx.proto3.pb.o: include/onnx.proto3.pb.cc
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET_MNIST): $(ENGINE_OBJS) bin/test_mnist.o $(PROTO_OBJ)
	$(CXX) $^ $(LDFLAGS) -o $@

$(TARGET_RESNET): $(ENGINE_OBJS) bin/test_resnet18.o $(PROTO_OBJ)
	$(CXX) $^ $(LDFLAGS) -o $@

$(TARGET_VIT): $(ENGINE_OBJS) bin/test_vit.o $(PROTO_OBJ)
	$(CXX) $^ $(LDFLAGS) -o $@

clean:
	rm -rf bin
