CXX = g++
CXXFLAGS = -std=c++17 -Wall -Iinclude -I.
LDFLAGS = -lprotobuf

# Gather all .cpp files and include the generated protobuf .cc file
SRCS = $(wildcard src/*.cpp) onnx.proto3.pb.cc
OBJS = $(patsubst %.cpp, bin/%.o, $(patsubst %.cc, bin/%.o, $(notdir $(SRCS))))

TARGET = bin/onnx_engine

all: $(TARGET)

bin/%.o: src/%.cpp
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

# Rule to compile the generated protobuf .cc file
bin/onnx.proto3.pb.o: src/onnx.proto3.pb.cc
	@mkdir -p bin
	$(CXX) $(CXXFLAGS) -c $< -o $@

$(TARGET): $(OBJS)
	$(CXX) $(OBJS) $(LDFLAGS) -o $@

clean:
	rm -rf bin