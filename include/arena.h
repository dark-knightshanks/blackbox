#pragma once
#include <cstdlib>
#include <cstdint>
#include <cstring>
#include <stdexcept>

class MemoryArena {
private:
    uint8_t* buffer;  // base pointer to arena
    size_t capacity;  // total memory arena to be allocated
    size_t offset;    // offset to the base pointer
public:
    explicit MemoryArena(size_t total_bytes) : capacity(total_bytes), offset(0) {
        buffer = static_cast<uint8_t*>(aligned_alloc(64, total_bytes));
        if (!buffer) {
            throw std::runtime_error("Memory Arens: alloc failed");
        }
        memset(buffer, 0, total_bytes);  // zeros out entire arena
    }
    ~MemoryArena() { free(buffer); }
    MemoryArena(const MemoryArena&) = delete;             // does not allow copy constuctor
    MemoryArena& operator=(const MemoryArena&) = delete;  // does noot allow assignment copying

    uint8_t* alloc(size_t num_bytes) {
        size_t aligned_offset = (offset + 63) & ~63;  // alligns the offset as 64 bytes

        if (aligned_offset + num_bytes > capacity) {
            throw std::runtime_error("Memory Arena: out of memory");
        }
        uint8_t* ptr = buffer + aligned_offset;
        offset = aligned_offset + num_bytes;  // updates offset
        return ptr;
    }

    void reset() { offset = 0; }

    uint8_t* base() const { return buffer; }
    size_t used() const { return offset; }
    size_t remaining() const { return capacity - offset; }
};
