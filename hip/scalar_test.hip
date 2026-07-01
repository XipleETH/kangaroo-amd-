// scalar_test.hip — validate k*G (double-and-add) against known anchors.
// Build: hipcc -O3 --offload-arch=gfx1101 scalar_test.hip -o scalar_test.exe
#include "secp256k1.h"
#include <cstdio>
#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ printf("HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 1; } }while(0)

// affine point with an explicit infinity flag (for general add / scalar mult)
struct jpt { fe x, y; bool inf; };

__device__ inline jpt J_G() {
    jpt g; g.inf=false;
    g.x = fe_set(0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL);
    g.y = fe_set(0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL);
    return g;
}

__device__ inline jpt j_double(const jpt& p) {
    if (p.inf) return p;
    fe num = fe_add(fe_add(fe_sqr(p.x), fe_sqr(p.x)), fe_sqr(p.x)); // 3x^2
    fe den = fe_add(p.y, p.y);                                      // 2y
    fe lam = fe_mul(num, fe_inv(den));
    jpt r; r.inf=false;
    r.x = fe_sub(fe_sqr(lam), fe_add(p.x, p.x));
    r.y = fe_sub(fe_mul(lam, fe_sub(p.x, r.x)), p.y);
    return r;
}

__device__ inline jpt j_add(const jpt& p, const jpt& q) {
    if (p.inf) return q;
    if (q.inf) return p;
    if (fe_eq(p.x, q.x)) {
        if (fe_eq(p.y, q.y)) return j_double(p);
        jpt r; r.inf=true; return r;   // P + (-P) = O
    }
    fe lam = fe_mul(fe_sub(q.y, p.y), fe_inv(fe_sub(q.x, p.x)));
    jpt r; r.inf=false;
    r.x = fe_sub(fe_sub(fe_sqr(lam), p.x), q.x);
    r.y = fe_sub(fe_mul(lam, fe_sub(p.x, r.x)), p.y);
    return r;
}

// k * G via double-and-add (MSB first)
__device__ inline jpt scalar_mul(const fe& k) {
    jpt R; R.inf = true;
    jpt G = J_G();
    for (int limb = 3; limb >= 0; --limb) {
        u64 e = k.n[limb];
        for (int b = 63; b >= 0; --b) {
            R = j_double(R);
            if ((e >> b) & 1ULL) R = j_add(R, G);
        }
    }
    return R;
}

__global__ void kat(unsigned* fail, fe* pubx) {
    unsigned f = 0;
    // 2*G matches known 2G.x
    jpt g2 = scalar_mul(fe_set(2,0,0,0));
    if (!(g2.x.n[3]==0xC6047F9441ED7D6DULL && g2.x.n[0]==0xABAC09B95C709EE5ULL)) f |= 1;
    // key 0xe9ae4933d6 * G == the test pubkey (x = a2efa402...de0d4, y odd)
    jpt Q = scalar_mul(fe_set(0xE9AE4933D6ULL, 0, 0, 0));
    fe want = fe_set(0xB9F0EDBEE53DE0D4ULL, 0x09EDEDEE7C4020E4ULL, 0x0C77C20E574BA864ULL, 0xA2EFA402FD526840ULL);
    if (!fe_eq(Q.x, want)) f |= 2;
    if ((Q.y.n[0] & 1ULL) != 1ULL) f |= 4;   // 03 prefix => odd y
    *fail = f;
    *pubx = Q.x;
}

int main() {
    unsigned* d_fail; CHECK(hipMalloc(&d_fail,4));
    fe* d_px; CHECK(hipMalloc(&d_px,sizeof(fe)));
    hipLaunchKernelGGL(kat, dim3(1), dim3(1), 0, 0, d_fail, d_px);
    CHECK(hipDeviceSynchronize());
    unsigned f; CHECK(hipMemcpy(&f,d_fail,4,hipMemcpyDeviceToHost));
    fe px; CHECK(hipMemcpy(&px,d_px,sizeof(fe),hipMemcpyDeviceToHost));
    printf("scalar_mul KAT failbits=0x%02X -> %s\n", f, f==0?"PASS":"FAIL");
    printf("key*G .x = %016llx%016llx%016llx%016llx\n", px.n[3],px.n[2],px.n[1],px.n[0]);
    printf("pubkey.x = a2efa402fd5268400c77c20e574ba86409ededee7c4020e4b9f0edbee53de0d4\n");
    printf("%s\n", f==0 ? "SCALAR MUL OK -> ready for kangaroo setup" : "BUG");
    return f==0?0:1;
}
