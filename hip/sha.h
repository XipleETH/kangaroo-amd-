// sha.h — SHA-256 and SHA-512 for the HIP MetaMask sweeper (BIP39 checksum + PBKDF2/BIP32).
// Single-block and streaming forms. Validated in sha_test.hip against FIPS vectors.
#pragma once
#include <hip/hip_runtime.h>
typedef unsigned long long u64;
typedef unsigned int u32;
typedef unsigned char u8;

// ---------------- SHA-256 ----------------
__device__ __constant__ u32 K256[64] = {
0x428a2f98,0x71374491,0xb5c0fbcf,0xe9b5dba5,0x3956c25b,0x59f111f1,0x923f82a4,0xab1c5ed5,
0xd807aa98,0x12835b01,0x243185be,0x550c7dc3,0x72be5d74,0x80deb1fe,0x9bdc06a7,0xc19bf174,
0xe49b69c1,0xefbe4786,0x0fc19dc6,0x240ca1cc,0x2de92c6f,0x4a7484aa,0x5cb0a9dc,0x76f988da,
0x983e5152,0xa831c66d,0xb00327c8,0xbf597fc7,0xc6e00bf3,0xd5a79147,0x06ca6351,0x14292967,
0x27b70a85,0x2e1b2138,0x4d2c6dfc,0x53380d13,0x650a7354,0x766a0abb,0x81c2c92e,0x92722c85,
0xa2bfe8a1,0xa81a664b,0xc24b8b70,0xc76c51a3,0xd192e819,0xd6990624,0xf40e3585,0x106aa070,
0x19a4c116,0x1e376c08,0x2748774c,0x34b0bcb5,0x391c0cb3,0x4ed8aa4a,0x5b9cca4f,0x682e6ff3,
0x748f82ee,0x78a5636f,0x84c87814,0x8cc70208,0x90befffa,0xa4506ceb,0xbef9a3f7,0xc67178f2};
__device__ inline u32 ror32(u32 x,int n){ return (x>>n)|(x<<(32-n)); }
// process one 64-byte block into state h[8]
__device__ inline void sha256_block(u32 h[8], const u8* p){
    u32 w[64];
    #pragma unroll
    for(int i=0;i<16;i++) w[i]=((u32)p[i*4]<<24)|((u32)p[i*4+1]<<16)|((u32)p[i*4+2]<<8)|((u32)p[i*4+3]);
    for(int i=16;i<64;i++){
        u32 s0=ror32(w[i-15],7)^ror32(w[i-15],18)^(w[i-15]>>3);
        u32 s1=ror32(w[i-2],17)^ror32(w[i-2],19)^(w[i-2]>>10);
        w[i]=w[i-16]+s0+w[i-7]+s1;
    }
    u32 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    for(int i=0;i<64;i++){
        u32 S1=ror32(e,6)^ror32(e,11)^ror32(e,25);
        u32 ch=(e&f)^((~e)&g);
        u32 t1=hh+S1+ch+K256[i]+w[i];
        u32 S0=ror32(a,2)^ror32(a,13)^ror32(a,22);
        u32 maj=(a&b)^(a&c)^(b&c);
        u32 t2=S0+maj;
        hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
}
// full SHA-256 of msg[len] -> out[32]
__device__ inline void sha256(const u8* msg,int len,u8 out[32]){
    u32 h[8]={0x6a09e667,0xbb67ae85,0x3c6ef372,0xa54ff53a,0x510e527f,0x9b05688c,0x1f83d9ab,0x5be0cd19};
    u8 blk[64]; int i=0;
    while(len-i>=64){ sha256_block(h,msg+i); i+=64; }
    int rem=len-i; for(int k=0;k<rem;k++) blk[k]=msg[i+k];
    blk[rem]=0x80; int pad=rem+1;
    if(pad>56){ for(int k=pad;k<64;k++) blk[k]=0; sha256_block(h,blk); pad=0; for(int k=0;k<56;k++) blk[k]=0; }
    else for(int k=pad;k<56;k++) blk[k]=0;
    u64 bits=(u64)len*8;
    for(int k=0;k<8;k++) blk[56+k]=(u8)(bits>>(56-8*k));
    sha256_block(h,blk);
    for(int k=0;k<8;k++){ out[k*4]=h[k]>>24; out[k*4+1]=h[k]>>16; out[k*4+2]=h[k]>>8; out[k*4+3]=h[k]; }
}

// ---------------- SHA-512 ----------------
__device__ __constant__ u64 K512[80]={
0x428a2f98d728ae22ULL,0x7137449123ef65cdULL,0xb5c0fbcfec4d3b2fULL,0xe9b5dba58189dbbcULL,
0x3956c25bf348b538ULL,0x59f111f1b605d019ULL,0x923f82a4af194f9bULL,0xab1c5ed5da6d8118ULL,
0xd807aa98a3030242ULL,0x12835b0145706fbeULL,0x243185be4ee4b28cULL,0x550c7dc3d5ffb4e2ULL,
0x72be5d74f27b896fULL,0x80deb1fe3b1696b1ULL,0x9bdc06a725c71235ULL,0xc19bf174cf692694ULL,
0xe49b69c19ef14ad2ULL,0xefbe4786384f25e3ULL,0x0fc19dc68b8cd5b5ULL,0x240ca1cc77ac9c65ULL,
0x2de92c6f592b0275ULL,0x4a7484aa6ea6e483ULL,0x5cb0a9dcbd41fbd4ULL,0x76f988da831153b5ULL,
0x983e5152ee66dfabULL,0xa831c66d2db43210ULL,0xb00327c898fb213fULL,0xbf597fc7beef0ee4ULL,
0xc6e00bf33da88fc2ULL,0xd5a79147930aa725ULL,0x06ca6351e003826fULL,0x142929670a0e6e70ULL,
0x27b70a8546d22ffcULL,0x2e1b21385c26c926ULL,0x4d2c6dfc5ac42aedULL,0x53380d139d95b3dfULL,
0x650a73548baf63deULL,0x766a0abb3c77b2a8ULL,0x81c2c92e47edaee6ULL,0x92722c851482353bULL,
0xa2bfe8a14cf10364ULL,0xa81a664bbc423001ULL,0xc24b8b70d0f89791ULL,0xc76c51a30654be30ULL,
0xd192e819d6ef5218ULL,0xd69906245565a910ULL,0xf40e35855771202aULL,0x106aa07032bbd1b8ULL,
0x19a4c116b8d2d0c8ULL,0x1e376c085141ab53ULL,0x2748774cdf8eeb99ULL,0x34b0bcb5e19b48a8ULL,
0x391c0cb3c5c95a63ULL,0x4ed8aa4ae3418acbULL,0x5b9cca4f7763e373ULL,0x682e6ff3d6b2b8a3ULL,
0x748f82ee5defb2fcULL,0x78a5636f43172f60ULL,0x84c87814a1f0ab72ULL,0x8cc702081a6439ecULL,
0x90befffa23631e28ULL,0xa4506cebde82bde9ULL,0xbef9a3f7b2c67915ULL,0xc67178f2e372532bULL,
0xca273eceea26619cULL,0xd186b8c721c0c207ULL,0xeada7dd6cde0eb1eULL,0xf57d4f7fee6ed178ULL,
0x06f067aa72176fbaULL,0x0a637dc5a2c898a6ULL,0x113f9804bef90daeULL,0x1b710b35131c471bULL,
0x28db77f523047d84ULL,0x32caab7b40c72493ULL,0x3c9ebe0a15c9bebcULL,0x431d67c49c100d4cULL,
0x4cc5d4becb3e42b6ULL,0x597f299cfc657e2aULL,0x5fcb6fab3ad6faecULL,0x6c44198c4a475817ULL};
__device__ inline u64 ror64(u64 x,int n){ return (x>>n)|(x<<(64-n)); }
__device__ __constant__ u64 SHA512_IV[8]={
    0x6a09e667f3bcc908ULL,0xbb67ae8584caa73bULL,0x3c6ef372fe94f82bULL,0xa54ff53a5f1d36f1ULL,
    0x510e527fade682d1ULL,0x9b05688c2b3e6c1fULL,0x1f83d9abfb41bd6bULL,0x5be0cd19137e2179ULL};
// absorb one block given the 16 big-endian message words (w is overwritten by the schedule).
__device__ inline void sha512_absorb_words(u64 h[8], u64 w[16]){
    u64 a=h[0],b=h[1],c=h[2],d=h[3],e=h[4],f=h[5],g=h[6],hh=h[7];
    #pragma unroll
    for(int i=0;i<80;i++){
        u64 wi;
        if(i<16) wi=w[i&15];
        else{
            u64 w1=w[(i+1)&15], w14=w[(i+14)&15];
            u64 s0=ror64(w1,1)^ror64(w1,8)^(w1>>7);
            u64 s1=ror64(w14,19)^ror64(w14,61)^(w14>>6);
            wi = w[i&15] + s0 + w[(i+9)&15] + s1;
            w[i&15]=wi;
        }
        u64 S1=ror64(e,14)^ror64(e,18)^ror64(e,41);
        u64 ch=(e&f)^((~e)&g);
        u64 t1=hh+S1+ch+K512[i]+wi;
        u64 S0=ror64(a,28)^ror64(a,34)^ror64(a,39);
        u64 maj=(a&b)^(a&c)^(b&c);
        u64 t2=S0+maj;
        hh=g;g=f;f=e;e=d+t1;d=c;c=b;b=a;a=t1+t2;
    }
    h[0]+=a;h[1]+=b;h[2]+=c;h[3]+=d;h[4]+=e;h[5]+=f;h[6]+=g;h[7]+=hh;
}
// byte-oriented wrapper: load 16 big-endian words then absorb.
__device__ inline void sha512_absorb_block(u64 h[8], const u8* p){
    u64 w[16];
    #pragma unroll
    for(int i=0;i<16;i++){ u64 v=0; for(int k=0;k<8;k++) v=(v<<8)|p[i*8+k]; w[i]=v; }
    sha512_absorb_words(h,w);
}
// finalize: h has absorbed full blocks totalling (total_len - tlen) bytes; process `tail`
// (tlen < 128) with padding for a message of total_len bytes. Emits 64-byte digest.
__device__ inline void sha512_finalize(u64 h[8], u64 total_len, const u8* tail, int tlen, u8 out[64]){
    u8 blk[256]; int n=0;
    for(int i=0;i<tlen;i++) blk[n++]=tail[i];
    blk[n++]=0x80;
    while((n&127)!=112) blk[n++]=0;
    u64 bits=total_len*8;   // total_len < 2^61
    for(int k=0;k<8;k++) blk[n++]=0;
    for(int k=0;k<8;k++) blk[n++]=(u8)(bits>>(56-8*k));
    for(int off=0;off<n;off+=128) sha512_absorb_block(h,blk+off);
    for(int k=0;k<8;k++) for(int b2=0;b2<8;b2++) out[k*8+b2]=(u8)(h[k]>>(56-8*b2));
}
__device__ inline void sha512(const u8* msg,int len,u8 out[64]){
    u64 h[8]; for(int i=0;i<8;i++) h[i]=SHA512_IV[i];
    int i=0; while(len-i>=128){ sha512_absorb_block(h,msg+i); i+=128; }
    sha512_finalize(h,(u64)len,msg+i,len-i,out);
}
