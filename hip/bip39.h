// bip39.h — HMAC-SHA512 + PBKDF2-HMAC-SHA512 (BIP39 seed). Validated in bip39_test.hip.
#pragma once
#include "sha.h"

// Precompute the SHA-512 midstates for an HMAC key: state after the ipad/opad block.
__device__ inline void hmac_sha512_states(const u8* key,int klen,u64 ipad_st[8],u64 opad_st[8]){
    u8 k[128];
    if(klen>128){ u8 hk[64]; sha512(key,klen,hk); for(int i=0;i<64;i++)k[i]=hk[i]; for(int i=64;i<128;i++)k[i]=0; }
    else { for(int i=0;i<klen;i++)k[i]=key[i]; for(int i=klen;i<128;i++)k[i]=0; }
    u8 blk[128];
    for(int i=0;i<128;i++) blk[i]=k[i]^0x36;
    for(int i=0;i<8;i++) ipad_st[i]=SHA512_IV[i]; sha512_absorb_block(ipad_st,blk);
    for(int i=0;i<128;i++) blk[i]=k[i]^0x5c;
    for(int i=0;i<8;i++) opad_st[i]=SHA512_IV[i]; sha512_absorb_block(opad_st,blk);
}
// HMAC given precomputed midstates: out = SHA512(opad || SHA512(ipad || msg)), msg < 128 bytes.
__device__ inline void hmac_sha512_from(const u64 ipad_st[8],const u64 opad_st[8],const u8* msg,int mlen,u8 out[64]){
    u64 st[8]; for(int i=0;i<8;i++) st[i]=ipad_st[i];
    u8 inner[64]; sha512_finalize(st,(u64)(128+mlen),msg,mlen,inner);
    for(int i=0;i<8;i++) st[i]=opad_st[i];
    sha512_finalize(st,(u64)(128+64),inner,64,out);
}
// legacy single-shot HMAC (used by non-hot paths / tests)
__device__ inline void hmac_sha512(const u8* key,int klen,const u8* msg,int mlen,u8 out[64]){
    u64 ip[8],op[8]; hmac_sha512_states(key,klen,ip,op);
    hmac_sha512_from(ip,op,msg,mlen,out);
}
// HMAC of a 64-byte (8-word) message, all in u64 — the PBKDF2 hot loop, no byte shuffling.
// Padding for a 192-byte message: 0x80 word then zeros then length 1536.
__device__ inline void hmac_sha512_64w(const u64 ip[8],const u64 op[8],u64 msg[8]){
    u64 w[16];
    #pragma unroll
    for(int i=0;i<8;i++) w[i]=msg[i];
    w[8]=0x8000000000000000ULL; w[9]=0;w[10]=0;w[11]=0;w[12]=0;w[13]=0;w[14]=0; w[15]=1536ULL;
    u64 st[8]; for(int i=0;i<8;i++) st[i]=ip[i]; sha512_absorb_words(st,w);   // inner
    #pragma unroll
    for(int i=0;i<8;i++) w[i]=st[i];
    w[8]=0x8000000000000000ULL; w[9]=0;w[10]=0;w[11]=0;w[12]=0;w[13]=0;w[14]=0; w[15]=1536ULL;
    for(int i=0;i<8;i++) st[i]=op[i]; sha512_absorb_words(st,w);              // outer
    #pragma unroll
    for(int i=0;i<8;i++) msg[i]=st[i];
}
// PBKDF2-HMAC-SHA512, dklen=64, password=mnemonic (fixed key -> precomputed states).
__device__ inline void bip39_seed(const u8* pwd,int plen,const u8* salt,int slen,u8 out[64]){
    u64 ip[8],op[8]; hmac_sha512_states(pwd,plen,ip,op);
    u8 s[72];
    for(int i=0;i<slen;i++) s[i]=salt[i];
    s[slen]=0; s[slen+1]=0; s[slen+2]=0; s[slen+3]=1;
    u8 U0[64];
    hmac_sha512_from(ip,op,s,slen+4,U0);       // U_1 (salt||INT)
    u64 U[8], T[8];
    #pragma unroll
    for(int i=0;i<8;i++){ u64 v=0; for(int k=0;k<8;k++) v=(v<<8)|U0[i*8+k]; U[i]=v; T[i]=v; }
    for(int it=1;it<2048;it++){
        hmac_sha512_64w(ip,op,U);
        #pragma unroll
        for(int i=0;i<8;i++) T[i]^=U[i];
    }
    #pragma unroll
    for(int i=0;i<8;i++) for(int b=0;b<8;b++) out[i*8+b]=(u8)(T[i]>>(56-8*b));
}
