// field_test.hip — secp256k1 field arithmetic in native HIP (u64x4 + __int128),
// with algebraic self-tests and a fe_mul throughput micro-benchmark.
//
// Field prime p = 2^256 - 2^32 - 977, so 2^256 ≡ C (mod p), C = 2^32 + 977 = 0x1000003D1.
//
// Build:  hipcc --offload-arch=gfx1101 field_test.hip -o field_test.exe
// Run:    field_test.exe

#include <hip/hip_runtime.h>
#include <cstdio>
#include <cstdint>

typedef unsigned long long u64;
typedef unsigned __int128 u128;

#define CHECK(call) do { hipError_t _e=(call); if(_e!=hipSuccess){ \
    printf("HIP error %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(_e)); return 1; } } while(0)

struct fe { u64 n[4]; };

// p limbs (little-endian)
__device__ __constant__ u64 P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
// (p-2) for Fermat inverse
__device__ __constant__ u64 Pm2[4] = {
    0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };

__device__ inline bool fe_geq_p(const fe& a) {
    for (int i = 3; i >= 0; --i) {
        if (a.n[i] != P[i]) return a.n[i] > P[i];
    }
    return true; // equal to p
}

// Conditionally subtract p once (input < 2^256, so result < p after at most... may need twice near 2^256).
__device__ inline void fe_cond_sub_p(fe& a) {
    while (fe_geq_p(a)) {
        u128 b = 0;
        for (int i = 0; i < 4; ++i) {
            u128 d = (u128)a.n[i] - P[i] - b;
            a.n[i] = (u64)d;
            b = (d >> 64) & 1;
        }
    }
}

// r = a + b (mod p)
__device__ inline fe fe_add(const fe& a, const fe& b) {
    fe r; u128 c = 0;
    for (int i = 0; i < 4; ++i) { u128 s = (u128)a.n[i] + b.n[i] + c; r.n[i] = (u64)s; c = s >> 64; }
    // carry out (c) means value >= 2^256 ≡ C; fold it in
    if ((u64)c) {
        u128 f = (u128)r.n[0] + 0x1000003D1ULL * (u64)c; r.n[0] = (u64)f; u64 cy = (u64)(f >> 64);
        u128 s1 = (u128)r.n[1] + cy; r.n[1] = (u64)s1; cy = (u64)(s1 >> 64);
        u128 s2 = (u128)r.n[2] + cy; r.n[2] = (u64)s2; cy = (u64)(s2 >> 64);
        r.n[3] += cy;
    }
    fe_cond_sub_p(r);
    return r;
}

// r = a - b (mod p)
__device__ inline fe fe_sub(const fe& a, const fe& b) {
    fe r; u128 bor = 0;
    for (int i = 0; i < 4; ++i) { u128 d = (u128)a.n[i] - b.n[i] - bor; r.n[i] = (u64)d; bor = (d >> 64) & 1; }
    if ((u64)bor) { // a<b: add p back
        u128 c = 0;
        for (int i = 0; i < 4; ++i) { u128 s = (u128)r.n[i] + P[i] + c; r.n[i] = (u64)s; c = s >> 64; }
    }
    return r;
}

// r = a * b (mod p)
__device__ inline fe fe_mul(const fe& a, const fe& b) {
    // 256x256 -> 512 schoolbook
    u64 t[8] = {0,0,0,0,0,0,0,0};
    for (int i = 0; i < 4; ++i) {
        u64 carry = 0;
        for (int j = 0; j < 4; ++j) {
            u128 prod = (u128)a.n[i] * b.n[j] + t[i + j] + carry;
            t[i + j] = (u64)prod;
            carry = (u64)(prod >> 64);
        }
        t[i + 4] = carry;
    }
    // reduce: acc = lo(t0..3) + C*hi(t4..7)
    const u128 C = 0x1000003D1ULL;
    u128 c0 = (u128)t[0] + C * t[4];
    u128 c1 = (u128)t[1] + C * t[5] + (u64)(c0 >> 64);
    u128 c2 = (u128)t[2] + C * t[6] + (u64)(c1 >> 64);
    u128 c3 = (u128)t[3] + C * t[7] + (u64)(c2 >> 64);
    fe r; r.n[0] = (u64)c0; r.n[1] = (u64)c1; r.n[2] = (u64)c2; r.n[3] = (u64)c3;
    u64 ov = (u64)(c3 >> 64);                       // < 2^34
    // fold ov (C*ov < 2^67)
    u128 d0 = (u128)r.n[0] + C * ov; r.n[0] = (u64)d0; u64 cy = (u64)(d0 >> 64);
    u128 d1 = (u128)r.n[1] + cy; r.n[1] = (u64)d1; cy = (u64)(d1 >> 64);
    u128 d2 = (u128)r.n[2] + cy; r.n[2] = (u64)d2; cy = (u64)(d2 >> 64);
    u128 d3 = (u128)r.n[3] + cy; r.n[3] = (u64)d3; u64 ov2 = (u64)(d3 >> 64); // 0/1
    if (ov2) {
        u128 e0 = (u128)r.n[0] + C; r.n[0] = (u64)e0; cy = (u64)(e0 >> 64);
        u128 e1 = (u128)r.n[1] + cy; r.n[1] = (u64)e1; cy = (u64)(e1 >> 64);
        u128 e2 = (u128)r.n[2] + cy; r.n[2] = (u64)e2; cy = (u64)(e2 >> 64);
        r.n[3] += cy;
    }
    fe_cond_sub_p(r);
    return r;
}

__device__ inline fe fe_sqr(const fe& a) { return fe_mul(a, a); }

__device__ inline bool fe_is_zero(const fe& a) { return (a.n[0]|a.n[1]|a.n[2]|a.n[3]) == 0; }
__device__ inline bool fe_eq(const fe& a, const fe& b) {
    return a.n[0]==b.n[0] && a.n[1]==b.n[1] && a.n[2]==b.n[2] && a.n[3]==b.n[3];
}
__device__ inline fe fe_one() { fe r; r.n[0]=1; r.n[1]=r.n[2]=r.n[3]=0; return r; }

// a^(p-2) mod p via square-and-multiply (correctness-first; optimize later).
__device__ inline fe fe_inv(const fe& a) {
    fe result = fe_one();
    fe base = a;
    for (int limb = 0; limb < 4; ++limb) {
        u64 e = Pm2[limb];
        for (int b = 0; b < 64; ++b) {
            if (e & 1ULL) result = fe_mul(result, base);
            base = fe_sqr(base);
            e >>= 1;
        }
    }
    return result;
}

// cheap per-thread PRNG to make test operands
__device__ inline u64 splitmix(u64& s) { s += 0x9E3779B97F4A7C15ULL; u64 z = s;
    z = (z ^ (z >> 30)) * 0xBF58476D1CE4E5B9ULL; z = (z ^ (z >> 27)) * 0x94D049BB133111EBULL; return z ^ (z >> 31); }
__device__ inline fe fe_rand(u64& s) { fe r; for (int i=0;i<4;++i) r.n[i]=splitmix(s); fe_cond_sub_p(r); return r; }

// ---- Correctness kernel: each thread checks algebraic identities, counts failures ----
__global__ void self_test(u64 seed, int iters, unsigned* fail) {
    u64 s = seed ^ (0x1234567ULL * (blockIdx.x * blockDim.x + threadIdx.x + 1));
    unsigned local = 0;
    for (int it = 0; it < iters; ++it) {
        fe a = fe_rand(s), b = fe_rand(s);
        // (a+b)*(a-b) == a*a - b*b
        fe lhs = fe_mul(fe_add(a,b), fe_sub(a,b));
        fe rhs = fe_sub(fe_sqr(a), fe_sqr(b));
        if (!fe_eq(lhs, rhs)) local++;
        // a * inv(a) == 1  (a nonzero w.h.p.)
        if (!fe_is_zero(a)) { if (!fe_eq(fe_mul(a, fe_inv(a)), fe_one())) local++; }
    }
    if (local) atomicAdd(fail, local);
}

// ---- Throughput kernel: dependent fe_mul chain (measures native mul rate) ----
__global__ void bench_mul(fe* out, u64 seed, int muls) {
    int idx = blockIdx.x * blockDim.x + threadIdx.x;
    u64 s = seed ^ (idx + 1);
    fe x = fe_rand(s), y = fe_rand(s);
    for (int i = 0; i < muls; ++i) { x = fe_mul(x, y); y = fe_mul(y, x); }
    out[idx] = x;
}

int main() {
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop, 0));
    printf("Device: %s (%s), %d CUs\n", prop.name, prop.gcnArchName, prop.multiProcessorCount);

    // --- correctness ---
    unsigned* d_fail; CHECK(hipMalloc(&d_fail, sizeof(unsigned)));
    CHECK(hipMemset(d_fail, 0, sizeof(unsigned)));
    const int threads = 4096, block = 256, iters = 200;
    hipLaunchKernelGGL(self_test, dim3(threads/block), dim3(block), 0, 0, 0xDEADBEEFCAFEULL, iters, d_fail);
    CHECK(hipDeviceSynchronize());
    unsigned fails = 0; CHECK(hipMemcpy(&fails, d_fail, sizeof(unsigned), hipMemcpyDeviceToHost));
    long checks = (long)threads * iters * 2;
    printf("Self-test: %ld algebraic checks, %u failures -> %s\n",
           checks, fails, fails == 0 ? "FIELD ARITHMETIC OK" : "BUG");

    // --- throughput ---
    const int bt = 65536, bb = 256, muls = 20000;
    fe* d_out; CHECK(hipMalloc(&d_out, (size_t)bt * sizeof(fe)));
    hipEvent_t e0, e1; CHECK(hipEventCreate(&e0)); CHECK(hipEventCreate(&e1));
    hipLaunchKernelGGL(bench_mul, dim3(bt/bb), dim3(bb), 0, 0, d_out, 0xABCDEFULL, 200); // warmup
    CHECK(hipDeviceSynchronize());
    CHECK(hipEventRecord(e0));
    hipLaunchKernelGGL(bench_mul, dim3(bt/bb), dim3(bb), 0, 0, d_out, 0xABCDEFULL, muls);
    CHECK(hipEventRecord(e1)); CHECK(hipEventSynchronize(e1));
    float ms = 0; CHECK(hipEventElapsedTime(&ms, e0, e1));
    double total = (double)bt * muls * 2.0;               // 2 fe_mul per loop iter
    printf("Throughput: %.3e fe_mul in %.1f ms = %.1f M fe_mul/s\n", total, ms, total / (ms * 1e3));
    printf("(rough kangaroo estimate ~ mul_rate/5 ops/s -> ~%.0f M ops/s)\n", total / (ms * 1e3) / 5.0);

    hipFree(d_fail); hipFree(d_out);
    return fails == 0 ? 0 : 1;
}
