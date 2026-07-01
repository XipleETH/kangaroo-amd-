// kangaroo.hip — minimal native-HIP Pollard's Kangaroo (2-set, u64 distance).
// First working solver: correctness-first (per-step inversion; batch inversion later).
// Solves Q = k*G for k in [0, 2^RANGE_BITS). Test target hard-coded to the 40-bit key.
//
// Build: hipcc -O3 --offload-arch=gfx1101 kangaroo.hip -o kangaroo.exe
#include "secp256k1.h"
#include "ec.h"
#include <cstdio>
#include <cstdint>
#include <string>
#include <unordered_map>
#include <chrono>

#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ printf("HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 1; } }while(0)

// ---- problem / tuning ----
#define RANGE_BITS 40
#define N_JUMPS    32
#define N_TAME     2048
#define N_WILD     2048
#define N_KANG     (N_TAME + N_WILD)
#define DP_BITS    12
#define STEPS      512
#define MAX_DP     (1u << 20)
#define MAX_ITERS  4000

// affine add of DISTINCT points, inline (kangaroo hot path)
__device__ inline void ec_add_inplace(fe& x, fe& y, const fe& jx, const fe& jy) {
    fe lam = fe_mul(fe_sub(jy, y), fe_inv(fe_sub(jx, x)));
    fe nx  = fe_sub(fe_sub(fe_sqr(lam), x), jx);
    fe ny  = fe_sub(fe_mul(lam, fe_sub(x, nx)), y);
    x = nx; y = ny;
}

__global__ void k_mkQ(fe* Qx, fe* Qy, u64 k0, u64 k1, u64 k2, u64 k3) {
    jpt Q = scalar_mul_G(fe_set(k0,k1,k2,k3));
    *Qx = Q.x; *Qy = Q.y;
}

__global__ void k_mkjumps(fe* jx, fe* jy, const u64* js, int nj) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= nj) return;
    jpt J = scalar_mul_G(fe_set(js[i],0,0,0));
    jx[i] = J.x; jy[i] = J.y;
}

// tame start = middle of range; wild start = Q + j*G
__global__ void k_mkkang(fe* kx, fe* ky, u64* kd, unsigned char* kt,
                         const fe* Qx, const fe* Qy, u64 mid) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_KANG) return;
    if (i < N_TAME) {
        u64 s = mid + (u64)i;
        jpt P = scalar_mul_G(fe_set(s,0,0,0));
        kx[i]=P.x; ky[i]=P.y; kd[i]=s; kt[i]=0;
    } else {
        u64 w = (u64)(i - N_TAME);
        jpt off = scalar_mul_G(fe_set(w,0,0,0));
        jpt Q; Q.inf=false; Q.x=*Qx; Q.y=*Qy;
        jpt P = j_add(Q, off);
        kx[i]=P.x; ky[i]=P.y; kd[i]=w; kt[i]=1;
    }
}

__global__ void k_walk(fe* kx, fe* ky, u64* kd, const unsigned char* kt,
                       const fe* jx, const fe* jy, const u64* js, int nj,
                       fe* dpx, u64* dpd, unsigned char* dpt, unsigned* dpcnt,
                       int steps, u64 dpmask) {
    int i = blockIdx.x*blockDim.x + threadIdx.x;
    if (i >= N_KANG) return;
    fe x = kx[i], y = ky[i]; u64 d = kd[i]; unsigned char t = kt[i];
    for (int s = 0; s < steps; ++s) {
        int idx = (int)(x.n[0] % (u64)nj);
        ec_add_inplace(x, y, jx[idx], jy[idx]);
        d += js[idx];
        if ((x.n[0] & dpmask) == 0ULL) {
            unsigned pos = atomicAdd(dpcnt, 1u);
            if (pos < MAX_DP) { dpx[pos]=x; dpd[pos]=d; dpt[pos]=t; }
        }
    }
    kx[i]=x; ky[i]=y; kd[i]=d;
}

__global__ void k_verify(const fe* Qx, u64 k0, u64 k1, u64 k2, u64 k3, unsigned* ok) {
    jpt P = scalar_mul_G(fe_set(k0,k1,k2,k3));
    *ok = fe_eq(P.x, *Qx) ? 1u : 0u;
}

int main() {
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop,0));
    printf("Device: %s (%s)\n", prop.name, prop.gcnArchName);
    printf("Problem: k*G, k in [0, 2^%d).  kangaroos=%d  dp_bits=%d\n", RANGE_BITS, N_KANG, DP_BITS);

    // test key (0xe9ae4933d6) -> builds Q = k*G on device (solver recovers k independently)
    const u64 KTEST[4] = { 0xE9AE4933D6ULL, 0, 0, 0 };

    // jump scalars: mean ~ 2^(RANGE_BITS/2)
    u64 h_js[N_JUMPS];
    u64 rng = 0x1234567ULL, span = 1ULL << (RANGE_BITS/2 + 1);
    for (int i=0;i<N_JUMPS;++i){ rng = rng*6364136223846793005ULL + 1442695040888963407ULL; h_js[i] = 1ULL + (rng % span); }

    // device buffers
    fe *Qx,*Qy,*jx,*jy,*kx,*ky,*dpx; u64 *js,*kd,*dpd; unsigned char *kt,*dpt; unsigned *dpcnt,*d_ok;
    CHECK(hipMalloc(&Qx,sizeof(fe))); CHECK(hipMalloc(&Qy,sizeof(fe)));
    CHECK(hipMalloc(&jx,N_JUMPS*sizeof(fe))); CHECK(hipMalloc(&jy,N_JUMPS*sizeof(fe)));
    CHECK(hipMalloc(&js,N_JUMPS*sizeof(u64)));
    CHECK(hipMalloc(&kx,N_KANG*sizeof(fe))); CHECK(hipMalloc(&ky,N_KANG*sizeof(fe)));
    CHECK(hipMalloc(&kd,N_KANG*sizeof(u64))); CHECK(hipMalloc(&kt,N_KANG*sizeof(unsigned char)));
    CHECK(hipMalloc(&dpx,MAX_DP*sizeof(fe))); CHECK(hipMalloc(&dpd,MAX_DP*sizeof(u64)));
    CHECK(hipMalloc(&dpt,MAX_DP*sizeof(unsigned char))); CHECK(hipMalloc(&dpcnt,sizeof(unsigned)));
    CHECK(hipMalloc(&d_ok,sizeof(unsigned)));
    CHECK(hipMemcpy(js,h_js,N_JUMPS*sizeof(u64),hipMemcpyHostToDevice));

    // setup
    hipLaunchKernelGGL(k_mkQ, dim3(1),dim3(1),0,0, Qx,Qy, KTEST[0],KTEST[1],KTEST[2],KTEST[3]);
    hipLaunchKernelGGL(k_mkjumps, dim3(1),dim3(N_JUMPS),0,0, jx,jy,js,N_JUMPS);
    u64 mid = 1ULL << (RANGE_BITS-1);
    hipLaunchKernelGGL(k_mkkang, dim3((N_KANG+255)/256),dim3(256),0,0, kx,ky,kd,kt,Qx,Qy,mid);
    CHECK(hipDeviceSynchronize());
    printf("setup done. walking...\n");

    // host DP table
    std::unordered_map<std::string, std::pair<u64,unsigned char>> table;
    table.reserve(1<<20);
    fe*  h_dpx = new fe[MAX_DP];
    u64* h_dpd = new u64[MAX_DP];
    unsigned char* h_dpt = new unsigned char[MAX_DP];
    const u64 dpmask = (1ULL<<DP_BITS)-1ULL;

    u64 total_ops = 0; bool solved=false; u64 found_k=0;
    auto t0 = std::chrono::high_resolution_clock::now();

    for (int it=0; it<MAX_ITERS && !solved; ++it) {
        CHECK(hipMemset(dpcnt,0,sizeof(unsigned)));
        hipLaunchKernelGGL(k_walk, dim3((N_KANG+255)/256),dim3(256),0,0,
                           kx,ky,kd,kt, jx,jy,js,N_JUMPS, dpx,dpd,dpt,dpcnt, STEPS, dpmask);
        CHECK(hipDeviceSynchronize());
        total_ops += (u64)N_KANG*STEPS;

        unsigned cnt=0; CHECK(hipMemcpy(&cnt,dpcnt,sizeof(unsigned),hipMemcpyDeviceToHost));
        if (cnt>MAX_DP) cnt=MAX_DP;
        if (cnt) {
            CHECK(hipMemcpy(h_dpx,dpx,cnt*sizeof(fe),hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(h_dpd,dpd,cnt*sizeof(u64),hipMemcpyDeviceToHost));
            CHECK(hipMemcpy(h_dpt,dpt,cnt*sizeof(unsigned char),hipMemcpyDeviceToHost));
        }
        for (unsigned j=0;j<cnt && !solved;++j) {
            std::string key((const char*)h_dpx[j].n, 32);
            auto f = table.find(key);
            if (f==table.end()) { table.emplace(key, std::make_pair(h_dpd[j], h_dpt[j])); }
            else if (f->second.second != h_dpt[j]) {
                u64 dt = (h_dpt[j]==0)? h_dpd[j] : f->second.first;   // tame dist
                u64 dw = (h_dpt[j]==1)? h_dpd[j] : f->second.first;   // wild dist
                u64 k = dt - dw;                                      // k = tame - wild
                // verify k*G == Q on device
                unsigned ok=0;
                hipLaunchKernelGGL(k_verify, dim3(1),dim3(1),0,0, Qx, k,0ULL,0ULL,0ULL, d_ok);
                hipDeviceSynchronize();
                hipMemcpy(&ok,d_ok,sizeof(unsigned),hipMemcpyDeviceToHost);
                if (ok) { solved=true; found_k=k; }
            }
        }
        if ((it % 200)==0) printf("  it=%d ops=%lluM dps=%zu\n", it, (unsigned long long)(total_ops/1000000), table.size());
    }

    auto t1 = std::chrono::high_resolution_clock::now();
    double sec = std::chrono::duration<double>(t1-t0).count();

    if (solved) {
        printf("\nSOLVED! k = 0x%llx  (expected 0xe9ae4933d6)\n", (unsigned long long)found_k);
        printf("%s\n", found_k==0xE9AE4933D6ULL ? "MATCH -> NATIVE HIP KANGAROO WORKS" : "verified but != expected (?)");
    } else printf("\nNOT solved within %d iters\n", MAX_ITERS);
    printf("ops=%.3e in %.2fs = %.1f M ops/s\n", (double)total_ops, sec, total_ops/(sec*1e6));
    return solved?0:1;
}
