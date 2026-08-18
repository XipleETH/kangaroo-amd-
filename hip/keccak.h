// keccak.h — Keccak-256 (Ethereum variant, 0x01 padding). Validated in bip32_test.hip.
#pragma once
#include "sha.h"

__device__ __constant__ u64 KECCAK_RC[24]={
0x0000000000000001ULL,0x0000000000008082ULL,0x800000000000808aULL,0x8000000080008000ULL,
0x000000000000808bULL,0x0000000080000001ULL,0x8000000080008081ULL,0x8000000000008009ULL,
0x000000000000008aULL,0x0000000000000088ULL,0x0000000080008009ULL,0x000000008000000aULL,
0x000000008000808bULL,0x800000000000008bULL,0x8000000000008089ULL,0x8000000000008003ULL,
0x8000000000008002ULL,0x8000000000000080ULL,0x000000000000800aULL,0x800000008000000aULL,
0x8000000080008081ULL,0x8000000000008080ULL,0x0000000080000001ULL,0x8000000080008008ULL};
__device__ __constant__ int KECCAK_ROT[24]={1,3,6,10,15,21,28,36,45,55,2,14,27,41,56,8,25,43,62,18,39,61,20,44};
__device__ __constant__ int KECCAK_PI[24]={10,7,11,17,18,3,5,16,8,21,24,4,15,23,19,13,12,2,20,14,22,9,6,1};
__device__ inline u64 rotl64(u64 x,int n){ return (x<<n)|(x>>(64-n)); }

__device__ inline void keccakf(u64 st[25]){
    for(int r=0;r<24;r++){
        u64 bc[5];
        for(int i=0;i<5;i++) bc[i]=st[i]^st[i+5]^st[i+10]^st[i+15]^st[i+20];
        for(int i=0;i<5;i++){ u64 t=bc[(i+4)%5]^rotl64(bc[(i+1)%5],1);
            for(int j=0;j<25;j+=5) st[j+i]^=t; }
        u64 t=st[1];
        for(int i=0;i<24;i++){ int j=KECCAK_PI[i]; u64 tmp=st[j]; st[j]=rotl64(t,KECCAK_ROT[i]); t=tmp; }
        for(int j=0;j<25;j+=5){ u64 s0=st[j],s1=st[j+1],s2=st[j+2],s3=st[j+3],s4=st[j+4];
            st[j]  =s0^((~s1)&s2); st[j+1]=s1^((~s2)&s3); st[j+2]=s2^((~s3)&s4);
            st[j+3]=s3^((~s4)&s0); st[j+4]=s4^((~s0)&s1); }
        st[0]^=KECCAK_RC[r];
    }
}
// keccak-256 of in[len] (len < 136 for our use, general single-rate loop otherwise)
__device__ inline void keccak256(const u8* in,int len,u8 out[32]){
    u64 st[25]; for(int i=0;i<25;i++)st[i]=0;
    const int rate=136; int i=0;
    while(len-i>=rate){
        for(int j=0;j<rate/8;j++){ u64 v=0; for(int k=0;k<8;k++) v|=(u64)in[i+j*8+k]<<(8*k); st[j]^=v; }
        keccakf(st); i+=rate;
    }
    u8 blk[136]; int rem=len-i; for(int k=0;k<rem;k++) blk[k]=in[i+k]; for(int k=rem;k<rate;k++) blk[k]=0;
    blk[rem]^=0x01; blk[rate-1]^=0x80;
    for(int j=0;j<rate/8;j++){ u64 v=0; for(int k=0;k<8;k++) v|=(u64)blk[j*8+k]<<(8*k); st[j]^=v; }
    keccakf(st);
    for(int j=0;j<4;j++) for(int k=0;k<8;k++) out[j*8+k]=(u8)(st[j]>>(8*k));
}
