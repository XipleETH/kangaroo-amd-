// secp256k1.h — device-side secp256k1 field arithmetic for the HIP kangaroo port.
// u64 x4 limbs (little-endian), __int128 products. p = 2^256 - 2^32 - 977.
// Validated by field_test.hip (1.6M algebraic checks, 0 failures, ~13.6 G fe_mul/s).
#pragma once
#include <hip/hip_runtime.h>

typedef unsigned long long u64;
typedef unsigned __int128 u128;

struct fe { u64 n[4]; };

__device__ __constant__ u64 SECP_P[4] = {
    0xFFFFFFFEFFFFFC2FULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };
__device__ __constant__ u64 SECP_Pm2[4] = {
    0xFFFFFFFEFFFFFC2DULL, 0xFFFFFFFFFFFFFFFFULL,
    0xFFFFFFFFFFFFFFFFULL, 0xFFFFFFFFFFFFFFFFULL };

__device__ inline fe fe_set(u64 a0, u64 a1, u64 a2, u64 a3) { fe r; r.n[0]=a0; r.n[1]=a1; r.n[2]=a2; r.n[3]=a3; return r; }
__device__ inline fe fe_zero() { return fe_set(0,0,0,0); }
__device__ inline fe fe_one()  { return fe_set(1,0,0,0); }
__device__ inline bool fe_is_zero(const fe& a) { return (a.n[0]|a.n[1]|a.n[2]|a.n[3]) == 0; }
__device__ inline bool fe_eq(const fe& a, const fe& b) {
    return a.n[0]==b.n[0] && a.n[1]==b.n[1] && a.n[2]==b.n[2] && a.n[3]==b.n[3]; }

__device__ inline bool fe_geq_p(const fe& a) {
    for (int i=3;i>=0;--i) if (a.n[i]!=SECP_P[i]) return a.n[i]>SECP_P[i];
    return true;
}
__device__ inline void fe_cond_sub_p(fe& a) {
    while (fe_geq_p(a)) {
        u128 b=0;
        for (int i=0;i<4;++i){ u128 d=(u128)a.n[i]-SECP_P[i]-b; a.n[i]=(u64)d; b=(d>>64)&1; }
    }
}

__device__ inline fe fe_add(const fe& a, const fe& b) {
    fe r; u128 c=0;
    for (int i=0;i<4;++i){ u128 s=(u128)a.n[i]+b.n[i]+c; r.n[i]=(u64)s; c=s>>64; }
    if ((u64)c) {
        u128 f=(u128)r.n[0]+0x1000003D1ULL*(u64)c; r.n[0]=(u64)f; u64 cy=(u64)(f>>64);
        u128 s1=(u128)r.n[1]+cy; r.n[1]=(u64)s1; cy=(u64)(s1>>64);
        u128 s2=(u128)r.n[2]+cy; r.n[2]=(u64)s2; cy=(u64)(s2>>64);
        r.n[3]+=cy;
    }
    fe_cond_sub_p(r); return r;
}
__device__ inline fe fe_sub(const fe& a, const fe& b) {
    fe r; u128 bor=0;
    for (int i=0;i<4;++i){ u128 d=(u128)a.n[i]-b.n[i]-bor; r.n[i]=(u64)d; bor=(d>>64)&1; }
    if ((u64)bor){ u128 c=0; for(int i=0;i<4;++i){ u128 s=(u128)r.n[i]+SECP_P[i]+c; r.n[i]=(u64)s; c=s>>64; } }
    return r;
}
__device__ inline fe fe_mul(const fe& a, const fe& b) {
    u64 t[8]={0,0,0,0,0,0,0,0};
    for (int i=0;i<4;++i){
        u64 carry=0;
        for (int j=0;j<4;++j){ u128 prod=(u128)a.n[i]*b.n[j]+t[i+j]+carry; t[i+j]=(u64)prod; carry=(u64)(prod>>64); }
        t[i+4]=carry;
    }
    const u128 C=0x1000003D1ULL;
    u128 c0=(u128)t[0]+C*t[4];
    u128 c1=(u128)t[1]+C*t[5]+(u64)(c0>>64);
    u128 c2=(u128)t[2]+C*t[6]+(u64)(c1>>64);
    u128 c3=(u128)t[3]+C*t[7]+(u64)(c2>>64);
    fe r; r.n[0]=(u64)c0; r.n[1]=(u64)c1; r.n[2]=(u64)c2; r.n[3]=(u64)c3;
    u64 ov=(u64)(c3>>64);
    u128 d0=(u128)r.n[0]+C*ov; r.n[0]=(u64)d0; u64 cy=(u64)(d0>>64);
    u128 d1=(u128)r.n[1]+cy; r.n[1]=(u64)d1; cy=(u64)(d1>>64);
    u128 d2=(u128)r.n[2]+cy; r.n[2]=(u64)d2; cy=(u64)(d2>>64);
    u128 d3=(u128)r.n[3]+cy; r.n[3]=(u64)d3; u64 ov2=(u64)(d3>>64);
    if (ov2){
        u128 e0=(u128)r.n[0]+C; r.n[0]=(u64)e0; cy=(u64)(e0>>64);
        u128 e1=(u128)r.n[1]+cy; r.n[1]=(u64)e1; cy=(u64)(e1>>64);
        u128 e2=(u128)r.n[2]+cy; r.n[2]=(u64)e2; cy=(u64)(e2>>64);
        r.n[3]+=cy;
    }
    fe_cond_sub_p(r); return r;
}
__device__ inline fe fe_sqr(const fe& a){ return fe_mul(a,a); }

__device__ inline fe fe_inv(const fe& a){
    fe result=fe_one(), base=a;
    for (int limb=0;limb<4;++limb){
        u64 e=SECP_Pm2[limb];
        for (int b=0;b<64;++b){ if (e&1ULL) result=fe_mul(result,base); base=fe_sqr(base); e>>=1; }
    }
    return result;
}

// small PRNG for tests
__device__ inline u64 splitmix(u64& s){ s+=0x9E3779B97F4A7C15ULL; u64 z=s;
    z=(z^(z>>30))*0xBF58476D1CE4E5B9ULL; z=(z^(z>>27))*0x94D049BB133111EBULL; return z^(z>>31); }
__device__ inline fe fe_rand(u64& s){ fe r; for(int i=0;i<4;++i) r.n[i]=splitmix(s); fe_cond_sub_p(r); return r; }
