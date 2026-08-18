// bip32.h — BIP32 derivation (m/44'/60'/0'/0/0) + ETH address. Reuses secp256k1 (ec.h).
// Validated end-to-end in bip32_test.hip against the canonical MetaMask vector.
#pragma once
#include "secp256k1.h"
#include "ec.h"
#include "keccak.h"
#include "bip39.h"

// secp256k1 group order n (little-endian limbs)
__device__ __constant__ u64 SECP_N[4]={
    0xBFD25E8CD0364141ULL,0xBAAEDCE6AF48A03BULL,0xFFFFFFFFFFFFFFFEULL,0xFFFFFFFFFFFFFFFFULL};
__device__ inline bool geq_n(const u64 a[4]){
    for(int i=3;i>=0;--i) if(a[i]!=SECP_N[i]) return a[i]>SECP_N[i];
    return true;
}
// big-endian 32 bytes -> fe (little-endian u64x4)
__device__ inline fe be32_to_fe(const u8* b){
    fe r;
    for(int i=0;i<4;i++){ u64 v=0; const u8* p=b+(3-i)*8; for(int k=0;k<8;k++) v=(v<<8)|p[k]; r.n[i]=v; }
    return r;
}
__device__ inline void fe_to_be32(const fe& a, u8 out[32]){
    for(int i=0;i<4;i++){ u64 v=a.n[3-i]; u8* p=out+i*8; for(int k=0;k<8;k++) p[k]=(u8)(v>>(56-8*k)); }
}
// r = (a + b) mod n,  a,b < n
__device__ inline fe add_mod_n(const fe& a,const fe& b){
    fe r; u128 c=0;
    for(int i=0;i<4;i++){ u128 s=(u128)a.n[i]+b.n[i]+c; r.n[i]=(u64)s; c=s>>64; }
    u64 carry=(u64)c;
    if(carry || geq_n(r.n)){ u128 br=0; for(int i=0;i<4;i++){ u128 d=(u128)r.n[i]-SECP_N[i]-br; r.n[i]=(u64)d; br=(d>>64)&1; } }
    return r;
}

struct bip32 { u8 k[32]; u8 c[32]; };

__device__ inline void priv_to_compressed(const u8 k[32], u8 out[33]){
    jpt P=scalar_mul_G_jac(be32_to_fe(k));
    out[0]=(P.y.n[0]&1ULL)?0x03:0x02;
    fe_to_be32(P.x,out+1);
}
__device__ inline bip32 bip32_ckd(const bip32& par,u32 index){
    u8 data[37]; u8 I[64];
    if(index & 0x80000000u){ data[0]=0; for(int i=0;i<32;i++) data[1+i]=par.k[i]; }
    else { u8 pub[33]; priv_to_compressed(par.k,pub); for(int i=0;i<33;i++) data[i]=pub[i]; }
    data[33]=(u8)(index>>24); data[34]=(u8)(index>>16); data[35]=(u8)(index>>8); data[36]=(u8)index;
    hmac_sha512(par.c,32,data,37,I);
    fe kc=add_mod_n(be32_to_fe(I), be32_to_fe(par.k));
    bip32 out; fe_to_be32(kc,out.k); for(int i=0;i<32;i++) out.c[i]=I[32+i];
    return out;
}
// seed(64) -> MetaMask default ETH address (20 bytes) at m/44'/60'/0'/0/0
__device__ inline void metamask_address(const u8 seed[64], u8 addr[20]){
    bip32 node; { u8 I[64]; hmac_sha512((const u8*)"Bitcoin seed",12,seed,64,I);
                  for(int i=0;i<32;i++){ node.k[i]=I[i]; node.c[i]=I[32+i]; } }
    const u32 path[5]={44u|0x80000000u,60u|0x80000000u,0u|0x80000000u,0u,0u};
    for(int s=0;s<5;s++) node=bip32_ckd(node,path[s]);
    jpt P=scalar_mul_G_jac(be32_to_fe(node.k));
    u8 pub[64]; fe_to_be32(P.x,pub); fe_to_be32(P.y,pub+32);
    u8 h[32]; keccak256(pub,64,h);
    for(int i=0;i<20;i++) addr[i]=h[12+i];
}
