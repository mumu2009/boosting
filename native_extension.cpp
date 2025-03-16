#include <stdint.h>

extern "C" __declspec(dllexport) void prefetch_data(void* address) {
  __asm__ volatile("prefetcht0 (%0)" :: "r"(address));
}

extern "C" __declspec(dllexport) void write_data_with_non_temporal_store(void* address, float data) {
  __asm__ volatile("movntps %1, (%0)" :: "r"(address), "x"(data));
}