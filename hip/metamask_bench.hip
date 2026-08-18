// metamask_bench.hip — measure full BIP39->ETH derivations/second on this GPU.
#include "bip32.h"
#include <cstdio>
#include <cstring>
#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ printf("HIP %s\n",hipGetErrorString(e)); return 1; } }while(0)

__global__ void bench(const u8* base,int mlen,int iters,u32* sink){
    int t=threadIdx.x+blockIdx.x*blockDim.x;
    u8 mn[110]; for(int i=0;i<mlen;i++) mn[i]=base[i];
    u32 acc=0;
    for(int it=0;it<iters;it++){
        mn[3]=base[3]+(u8)(t*7+it);
        u8 seed[64]; bip39_seed(mn,mlen,(const u8*)"mnemonic",8,seed);
        u8 addr[20]; metamask_address(seed,addr);
        acc^=addr[0]^addr[19];
    }
    sink[t]=acc;
}
__global__ void bench_seed(const u8* base,int mlen,int iters,u32* sink){
    int t=threadIdx.x+blockIdx.x*blockDim.x;
    u8 mn[110]; for(int i=0;i<mlen;i++) mn[i]=base[i];
    u32 acc=0;
    for(int it=0;it<iters;it++){
        mn[3]=base[3]+(u8)(t*7+it);
        u8 seed[64]; bip39_seed(mn,mlen,(const u8*)"mnemonic",8,seed);
        acc^=seed[0]^seed[63];
    }
    sink[t]=acc;
}
int main(){
    setvbuf(stdout,NULL,_IONBF,0);
    const char* mn="abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
    int mlen=strlen(mn);
    u8* dmn; CHECK(hipMalloc(&dmn,mlen)); CHECK(hipMemcpy(dmn,mn,mlen,hipMemcpyHostToDevice));
    int T=131072, B=64, iters=4; u32* dsink; CHECK(hipMalloc(&dsink,(size_t)T*4));
    for(int mode=0;mode<2;mode++){
        auto fn = mode? bench_seed : bench;
        hipLaunchKernelGGL(fn,dim3(T/B),dim3(B),0,0,dmn,mlen,1,dsink); CHECK(hipDeviceSynchronize());
        hipEvent_t e0,e1; hipEventCreate(&e0); hipEventCreate(&e1);
        hipEventRecord(e0);
        hipLaunchKernelGGL(fn,dim3(T/B),dim3(B),0,0,dmn,mlen,iters,dsink);
        hipEventRecord(e1); CHECK(hipEventSynchronize(e1));
        float ms=0; hipEventElapsedTime(&ms,e0,e1);
        double n=(double)T*iters;
        printf("%-16s %.0f ops in %.1f ms -> %.0f/s  (%.3fM/s)\n",
               mode?"PBKDF2-only":"full-pipeline", n,ms,n/(ms/1000.0),n/(ms*1000.0));
    }
    return 0;
}
