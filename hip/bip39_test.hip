// bip39_test.hip — validate HMAC-SHA512 + BIP39 seed (PBKDF2) against known vectors.
#include "bip39.h"
#include <cstdio>
#include <cstring>
#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ printf("HIP %s\n",hipGetErrorString(e)); return 1; } }while(0)

__global__ void k_hmac(const u8* key,int kl,const u8* msg,int ml,u8* out){ hmac_sha512(key,kl,msg,ml,out); }
__global__ void k_seed(const u8* pwd,int pl,const u8* salt,int sl,u8* out){ bip39_seed(pwd,pl,salt,sl,out); }
static void hex(const u8* p,int n,char* s){ const char* h="0123456789abcdef"; for(int i=0;i<n;i++){ s[2*i]=h[p[i]>>4]; s[2*i+1]=h[p[i]&15]; } s[2*n]=0; }

int main(){
    int fail=0;
    // 1) HMAC-SHA512 test vector (RFC 4231 test case 1): key=0x0b*20, data="Hi There"
    { u8 key[20]; for(int i=0;i<20;i++)key[i]=0x0b; const char* d="Hi There";
      u8 *dk,*dm,*dout; CHECK(hipMalloc(&dk,20)); CHECK(hipMalloc(&dm,8)); CHECK(hipMalloc(&dout,64));
      CHECK(hipMemcpy(dk,key,20,hipMemcpyHostToDevice)); CHECK(hipMemcpy(dm,d,8,hipMemcpyHostToDevice));
      hipLaunchKernelGGL(k_hmac,dim3(1),dim3(1),0,0,dk,20,dm,8,dout); CHECK(hipDeviceSynchronize());
      u8 o[64]; CHECK(hipMemcpy(o,dout,64,hipMemcpyDeviceToHost)); char s[130]; hex(o,64,s);
      const char* exp="87aa7cdea5ef619d4ff0b4241a1d6cb02379f4e2ce4ec2787ad0b30545e17cdedaa833b7d6b8a702038b274eaea3f4e4be9d914eeb61f1702e696c203a126854";
      int ok=strcmp(s,exp)==0; printf("HMAC-SHA512 RFC4231#1   %s %s\n",ok?"OK":"FAIL",ok?"":s); fail|=!ok;
      hipFree(dk);hipFree(dm);hipFree(dout); }
    // 2) BIP39 seed: "abandon"x11+"about", passphrase "" -> salt "mnemonic"
    { const char* mn="abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon abandon about";
      const char* salt="mnemonic"; int pl=strlen(mn), sl=strlen(salt);
      u8 *dp,*ds,*dout; CHECK(hipMalloc(&dp,pl)); CHECK(hipMalloc(&ds,sl)); CHECK(hipMalloc(&dout,64));
      CHECK(hipMemcpy(dp,mn,pl,hipMemcpyHostToDevice)); CHECK(hipMemcpy(ds,salt,sl,hipMemcpyHostToDevice));
      hipLaunchKernelGGL(k_seed,dim3(1),dim3(1),0,0,dp,pl,ds,sl,dout); CHECK(hipDeviceSynchronize());
      u8 o[64]; CHECK(hipMemcpy(o,dout,64,hipMemcpyDeviceToHost)); char s[130]; hex(o,64,s);
      const char* exp="5eb00bbddcf069084889a8ab9155568165f5c453ccb85e70811aaed6f6da5fc19a5ac40b389cd370d086206dec8aa6c43daea6690f20ad3d8d48b2d2ce9e38e4";
      int ok=strcmp(s,exp)==0; printf("BIP39 seed (abandon...)  %s %s\n",ok?"OK":"FAIL",ok?"":s); fail|=!ok;
      hipFree(dp);hipFree(ds);hipFree(dout); }
    printf(fail?"\nFAILED\n":"\nHMAC + BIP39 SEED OK\n");
    return fail;
}
