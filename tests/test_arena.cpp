#include <iostream>
#include <cassert>
#include <cstdint>
#include "arena.h"

int main() {
    // 1 MB arena
    MemoryArena arena(1024 * 1024);

    // Test 1: First alloc starts at base and is 64-byte aligned
    uint8_t* p1 = arena.alloc(1000);
    assert(p1 == arena.base());
    assert(reinterpret_cast<uintptr_t>(p1) % 64 == 0);
    std::cout << "[PASS] Test 1: First alloc at base, 64-byte aligned\n";

    // Test 2: Second alloc is non-overlapping and aligned
    uint8_t* p2 = arena.alloc(500);
    assert(p2 > p1);
    assert(reinterpret_cast<uintptr_t>(p2) % 64 == 0);
    assert(p2 >= p1 + 1000);  // no overlap with first allocation
    std::cout << "[PASS] Test 2: Second alloc non-overlapping, 64-byte aligned\n";

    // Test 3: Write and read back float data through arena memory
    float* fp = reinterpret_cast<float*>(p1);
    fp[0] = 3.14f;
    fp[1] = 2.71f;
    assert(fp[0] == 3.14f);
    assert(fp[1] == 2.71f);
    std::cout << "[PASS] Test 3: Write/read float data\n";

    // Test 4: used() + remaining() == total capacity
    size_t u = arena.used();
    size_t r = arena.remaining();
    assert(u + r == 1024 * 1024);
    std::cout << "[PASS] Test 4: used() + remaining() == capacity\n";

    // Test 5: Reset rewinds offset to zero
    arena.reset();
    assert(arena.used() == 0);
    assert(arena.remaining() == 1024 * 1024);
    std::cout << "[PASS] Test 5: Reset rewinds offset to 0\n";

    // Test 6: After reset, allocation starts from base again
    uint8_t* p3 = arena.alloc(64);
    assert(p3 == arena.base());
    std::cout << "[PASS] Test 6: Post-reset alloc starts at base\n";

    // Test 7: Overflow throws runtime_error
    arena.reset();
    bool caught = false;
    try {
        arena.alloc(2 * 1024 * 1024);  // 2 MB > 1 MB arena
    } catch (const std::runtime_error&) {
        caught = true;
    }
    assert(caught);
    std::cout << "[PASS] Test 7: Overflow throws runtime_error\n";

    // Test 8: Many small allocations all stay aligned
    arena.reset();
    for (int i = 0; i < 100; ++i) {
        uint8_t* p = arena.alloc(37);  // odd size, forces alignment padding
        assert(reinterpret_cast<uintptr_t>(p) % 64 == 0);
    }
    std::cout << "[PASS] Test 8: 100 odd-sized allocs all 64-byte aligned\n";

    std::cout << "\n=== All arena tests passed! ===\n";
    return 0;
}
