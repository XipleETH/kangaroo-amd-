// sha_test.hip — validate SHA-256 / SHA-512 device code against FIPS vectors.
#include "sha.h"
#include <cstdio>
#include <cstring>
#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ printf("HIP %s\n",hipGetErrorString(e)); return 1; } }while(0)

__global__ void k_sha256(const u8* msg,int len,u8* out){ sha256(msg,len,out); }
__global__ void k_sha512(const u8* msg,int len,u8* out){ sha512(msg,len,out); }

static void hex(const u8* p,int n,char* s){ const char* h="0123456789abcdef"; for(int i=0;i<n;i++){ s[2*i]=h[p[i]>>4]; s[2*i+1]=h[p[i]&15]; } s[2*n]=0; }

static int run(const char* label,const char* msg,int len,int is512,const char* expect){
    u8 *dmsg,*dout; CHECK(hipMalloc(&dmsg, len>0?len:1)); CHECK(hipMalloc(&dout,64));
    if(len) CHECK(hipMemcpy(dmsg,msg,len,hipMemcpyHostToDevice));
    if(is512) hipLaunchKernelGGL(k_sha512,dim3(1),dim3(1),0,0,dmsg,len,dout);
    else      hipLaunchKernelGGL(k_sha256,dim3(1),dim3(1),0,0,dmsg,len,dout);
    CHECK(hipDeviceSynchronize());
    u8 out[64]; int n=is512?64:32; CHECK(hipMemcpy(out,dout,n,hipMemcpyDeviceToHost));
    char s[130]; hex(out,n,s);
    int ok=strcmp(s,expect)==0;
    printf("%-28s %s  %s\n", label, ok?"OK":"FAIL", ok?"":s);
    hipFree(dmsg); hipFree(dout);
    return ok?0:1;
}
int main(){
    int f=0;
    f|=run("sha256('abc')","abc",3,0,"ba7816bf8f01cfea414140de5dae2223b00361a396177a9cb410ff61f20015ad");
    f|=run("sha256('')","",0,0,"e3b0c44298fc1c149afbf4c8996fb92427ae41e4649b934ca495991b7852b855");
    // 56-byte edge (exactly forces a second block)
    f|=run("sha256(56*'a')","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",56,0,"b35439a4ac6f0948b6d6f9e3c6af0f5f590ce20f1bde7090ef7970686ec6738a");
    f|=run("sha512('abc')","abc",3,1,"ddaf35a193617abacc417349ae20413112e6fa4e89a97ea20a9eeee64b55d39a2192992a274fc1a836ba3c23a3feebbd454d4423643ce80e2a9ac94fa54ca49f");
    f|=run("sha512('')","",0,1,"cf83e1357eefb8bdf1542850d66d8007d620e4050b5715dc83f4a921d36ce9ce47d0d13c5d85f2b0ff8318d2877eec2f63b931bd47417a81a538327af927da3e");
    // 111 and 112-byte edges (padding boundary for sha512)
    f|=run("sha512(111*'a')","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",111,1,"fa9121c7b32b9e01733d034cfc78cbf67f926c7ed83e82200ef86818196921760b4beff48404df811b953828274461673c68d04e297b0eb7b2b4d60fc6b566a2");
    f|=run("sha512(112*'a')","aaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaaa",112,1,"c01d080efd492776a1c43bd23dd99d0a2e626d481e16782e75d54c2503b5dc32bd05f0f1ba33e568b88fd2d970929b719ecbb152f58f130a407c8830604b70ca");
    printf(f?"\nSOME FAILED\n":"\nALL SHA VECTORS OK\n");
    return f;
}
