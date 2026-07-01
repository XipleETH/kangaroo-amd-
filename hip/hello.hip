// hello.hip — HIP toolchain validation for AMD RX 7800 XT (gfx1101)
// Gate 1: does hipGetDeviceProperties report a gfx1101 GPU (not just CPU)?
// Gate 2: does a trivial kernel actually run on the GPU and return correct data?
//
// Build + run (from a terminal where the HIP SDK bin is on PATH):
//   hipcc --offload-arch=gfx1101 hello.hip -o hello.exe
//   hello.exe
//
// PASS = prints "Device: AMD Radeon RX 7800 XT", "gcnArchName: gfx1101",
//        and "GPU COMPUTE OK".

#include <hip/hip_runtime.h>
#include <cstdio>

#define CHECK(call)                                                            \
    do {                                                                       \
        hipError_t _e = (call);                                                \
        if (_e != hipSuccess) {                                                \
            printf("HIP error %d at %s:%d -> %s\n", (int)_e, __FILE__,         \
                   __LINE__, hipGetErrorString(_e));                           \
            return 1;                                                          \
        }                                                                      \
    } while (0)

__global__ void square_indices(int* out, int n) {
    int i = blockIdx.x * blockDim.x + threadIdx.x;
    if (i < n) {
        out[i] = i * i;
    }
}

int main() {
    int deviceCount = 0;
    CHECK(hipGetDeviceCount(&deviceCount));
    printf("HIP devices found: %d\n", deviceCount);
    if (deviceCount == 0) {
        printf("NO GPU DEVICE. Toolchain/driver mismatch or gfx1101 not "
               "enumerated. Fix the install before continuing.\n");
        return 1;
    }

    hipDeviceProp_t prop;
    CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device:         %s\n", prop.name);
    printf("gcnArchName:    %s\n", prop.gcnArchName);
    printf("Compute units:  %d\n", prop.multiProcessorCount);
    printf("Warp/wave size: %d\n", prop.warpSize);
    printf("Global mem:     %.1f GB\n", prop.totalGlobalMem / 1073741824.0);

    const int N = 4096;
    int* d_out = nullptr;
    CHECK(hipMalloc(&d_out, N * sizeof(int)));

    const int block = 64;
    const int grid = (N + block - 1) / block;
    hipLaunchKernelGGL(square_indices, dim3(grid), dim3(block), 0, 0, d_out, N);
    CHECK(hipGetLastError());
    CHECK(hipDeviceSynchronize());

    int* h_out = new int[N];
    CHECK(hipMemcpy(h_out, d_out, N * sizeof(int), hipMemcpyDeviceToHost));

    bool ok = true;
    for (int i = 0; i < N; ++i) {
        if (h_out[i] != i * i) { ok = false; break; }
    }
    printf("out[2]=%d out[64]=%d out[4095]=%d (expect 4, 4096, %d)\n",
           h_out[2], h_out[64], h_out[4095], 4095 * 4095);
    printf("%s\n", ok ? "GPU COMPUTE OK  -> toolchain validated, go to Stage 2"
                      : "MISMATCH -> kernel ran but produced wrong data");

    delete[] h_out;
    CHECK(hipFree(d_out));
    return ok ? 0 : 1;
}
