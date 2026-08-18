// bip32_test.hip — validate keccak256 + the full mnemonic->seed->BIP32->ETH address pipeline.
#include "bip32.h"
#include <cstdio>
#include <cstring>
#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ printf("HIP %s\n",hipGetErrorString(e)); return 1; } }while(0)
static void hex(const u8* p,int n,char* s){ const char* h="0123456789abcdef"; for(int i=0;i<n;i++){ s[2*i]=h[p[i]>>4]; s[2*i+1]=h[p[i]&15]; } s[2*n]=0; }

__global__ void k_keccak(const u8* in,int len,u8* out){ keccak256(in,len,out); }
// full pipeline for a mnemonic (as bytes) with salt "mnemonic"
__global__ void k_full(const u8* mn,int mlen,u8* addr){
    u8 seed[64]; bip39_seed(mn,mlen,(const u8*)"mnemonic",8,seed);
    metamask_address(seed,addr);
}

int main(){
    int fail=0;
    // 1) keccak256 vectors
    { auto t=[&](const char*msg,int len,const char*exp,const char*lbl){
        u8 *di,*doo; hipMalloc(&di,len>0?len:1); hipMalloc(&doo,32);
        if(len) hipMemcpy(di,msg,len,hipMemcpyHostToDevice);
        hipLaunchKernelGGL(k_keccak,dim3(1),dim3(1),0,0,di,len,doo); hipDeviceSynchronize();
        u8 o[32]; hipMemcpy(o,doo,32,hipMemcpyDeviceToHost); char s[70]; hex(o,32,s);
        int ok=strcmp(s,exp)==0; printf("%-22s %s %s\n",lbl,ok?"OK":"FAIL",ok?"":s); fail|=!ok;
        hipFree(di);hipFree(doo); };
      t("",0,"c5d2460186f7233c927e7db2dcc703c0e500b653ca82273b7bfad8045d85a470","keccak256('')");
      t("abc",3,"4e03657aea45a94fc7d47ba826c8d667c0d1e6e33a64a036ec44f58fa12d6c45","keccak256('abc')");
    }
    // 2) FULL pipeline: abandon x11 + about -> 0x9858effd232b4033e47d90003d41ec34ecaeda94
    { const char* mn="abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
      int mlen=strlen(mn);
      u8 *dm,*da; CHECK(hipMalloc(&dm,mlen)); CHECK(hipMalloc(&da,20));
      CHECK(hipMemcpy(dm,mn,mlen,hipMemcpyHostToDevice));
      hipLaunchKernelGGL(k_full,dim3(1),dim3(1),0,0,dm,mlen,da); CHECK(hipDeviceSynchronize());
      u8 a[20]; CHECK(hipMemcpy(a,da,20,hipMemcpyDeviceToHost)); char s[45]; hex(a,20,s);
      int ok=strcmp(s,"9858effd232b4033e47d90003d41ec34ecaeda94")==0;
      printf("FULL MetaMask pipeline %s  0x%s\n",ok?"OK":"FAIL",s); fail|=!ok;
      hipFree(dm);hipFree(da); }
    printf(fail?"\nPIPELINE FAILED\n":"\n*** FULL BIP39->ETH PIPELINE VALIDATED ***\n");
    return fail;
}
