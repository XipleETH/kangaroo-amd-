// sqr_test.hip — validate the dedicated fe_sqr against fe_mul(a,a) + compare speed.
#include "secp256k1.h"
#include <cstdio>
#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ printf("HIP %s\n",hipGetErrorString(e)); return 1; } }while(0)

__global__ void check(u64 seed, int iters, unsigned* fail) {
    u64 s = seed ^ (0x9E37ULL*(blockIdx.x*blockDim.x+threadIdx.x+1));
    unsigned f=0;
    for (int i=0;i<iters;++i){ fe a=fe_rand(s); if(!fe_eq(fe_sqr(a), fe_mul(a,a))) f++; }
    if(f) atomicAdd(fail,f);
}
__global__ void bench_sqr(fe* out, u64 seed, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; u64 s=seed^(i+1);
    fe x=fe_rand(s); for(int k=0;k<n;++k) x=fe_sqr(x); out[i]=x; }
__global__ void bench_mul(fe* out, u64 seed, int n){ int i=blockIdx.x*blockDim.x+threadIdx.x; u64 s=seed^(i+1);
    fe x=fe_rand(s), y=fe_rand(s); for(int k=0;k<n;++k) x=fe_mul(x,y); out[i]=x; }

int main(){
    unsigned* d_f; CHECK(hipMalloc(&d_f,4)); CHECK(hipMemset(d_f,0,4));
    hipLaunchKernelGGL(check,dim3(16),dim3(256),0,0, 0xABCDEF12ULL, 2000, d_f);
    CHECK(hipDeviceSynchronize());
    unsigned f; CHECK(hipMemcpy(&f,d_f,4,hipMemcpyDeviceToHost));
    long checks=(long)16*256*2000;
    printf("fe_sqr == fe_mul(a,a): %ld checks, %u mismatches -> %s\n", checks, f, f==0?"OK":"BUG");

    const int T=65536, B=256, N=20000;
    fe* o; CHECK(hipMalloc(&o,(size_t)T*sizeof(fe)));
    hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
    hipLaunchKernelGGL(bench_sqr,dim3(T/B),dim3(B),0,0,o,1ULL,200); CHECK(hipDeviceSynchronize());
    hipEventRecord(e0); hipLaunchKernelGGL(bench_sqr,dim3(T/B),dim3(B),0,0,o,1ULL,N); hipEventRecord(e1); hipEventSynchronize(e1);
    float ms=0; hipEventElapsedTime(&ms,e0,e1); printf("fe_sqr: %.0f M/s\n",(double)T*N/(ms*1e3));
    hipEventRecord(e0); hipLaunchKernelGGL(bench_mul,dim3(T/B),dim3(B),0,0,o,1ULL,N); hipEventRecord(e1); hipEventSynchronize(e1);
    hipEventElapsedTime(&ms,e0,e1); printf("fe_mul: %.0f M/s\n",(double)T*N/(ms*1e3));
    return f==0?0:1;
}
