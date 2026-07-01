// ec_test.hip — secp256k1 affine point ops (add/double) + correctness test.
// Build: hipcc -O3 --offload-arch=gfx1101 ec_test.hip -o ec_test.exe
#include "secp256k1.h"
#include <cstdio>

#define CHECK(c) do{ hipError_t e=(c); if(e!=hipSuccess){ printf("HIP %s:%d %s\n",__FILE__,__LINE__,hipGetErrorString(e)); return 1; } }while(0)

struct ecpt { fe x, y; };

// P + Q for DISTINCT points (x1 != x2) — the kangaroo primitive.
__device__ inline ecpt ec_add(const ecpt& p, const ecpt& q) {
    fe lam = fe_mul(fe_sub(q.y, p.y), fe_inv(fe_sub(q.x, p.x)));
    fe x3  = fe_sub(fe_sub(fe_sqr(lam), p.x), q.x);
    fe y3  = fe_sub(fe_mul(lam, fe_sub(p.x, x3)), p.y);
    ecpt r; r.x = x3; r.y = y3; return r;
}
// 2P
__device__ inline ecpt ec_double(const ecpt& p) {
    fe x2  = fe_sqr(p.x);
    fe num = fe_add(fe_add(x2, x2), x2);      // 3x^2  (a=0 for secp256k1)
    fe den = fe_add(p.y, p.y);                // 2y
    fe lam = fe_mul(num, fe_inv(den));
    fe x3  = fe_sub(fe_sqr(lam), fe_add(p.x, p.x));
    fe y3  = fe_sub(fe_mul(lam, fe_sub(p.x, x3)), p.y);
    ecpt r; r.x = x3; r.y = y3; return r;
}
// y^2 == x^3 + 7  (mod p)
__device__ inline bool on_curve(const ecpt& p) {
    fe y2  = fe_sqr(p.y);
    fe rhs = fe_add(fe_mul(fe_sqr(p.x), p.x), fe_set(7,0,0,0));
    return fe_eq(y2, rhs);
}
__device__ inline bool pt_eq(const ecpt& a, const ecpt& b) { return fe_eq(a.x,b.x) && fe_eq(a.y,b.y); }

__global__ void ec_kat(unsigned* fail, fe* g2out) {
    unsigned f = 0;
    ecpt G;
    G.x = fe_set(0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL);
    G.y = fe_set(0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL);

    if (!on_curve(G))  f |= 1;                    // generator on curve
    ecpt G2 = ec_double(G);   if (!on_curve(G2)) f |= 2;
    ecpt G3 = ec_add(G, G2);  if (!on_curve(G3)) f |= 4;
    ecpt G3b= ec_add(G2, G);  if (!pt_eq(G3,G3b)) f |= 8;    // commutativity
    ecpt G4a= ec_double(G2);  if (!on_curve(G4a))f |= 16;
    ecpt G4b= ec_add(G3, G);  if (!pt_eq(G4a,G4b))f |= 32;   // 2(2G) == 3G+G
    ecpt G5 = ec_add(G4a, G); if (!on_curve(G5)) f |= 64;
    ecpt G5b= ec_add(G4b, G); if (!pt_eq(G5,G5b)) f |= 128;

    *fail = f;
    *g2out = G2.x;
}

int main() {
    hipDeviceProp_t prop; CHECK(hipGetDeviceProperties(&prop,0));
    printf("Device: %s (%s)\n", prop.name, prop.gcnArchName);
    unsigned* d_fail; CHECK(hipMalloc(&d_fail, 4));
    fe* d_g2; CHECK(hipMalloc(&d_g2, sizeof(fe)));
    hipLaunchKernelGGL(ec_kat, dim3(1), dim3(1), 0, 0, d_fail, d_g2);
    CHECK(hipDeviceSynchronize());
    unsigned f; CHECK(hipMemcpy(&f, d_fail, 4, hipMemcpyDeviceToHost));
    fe g2; CHECK(hipMemcpy(&g2, d_g2, sizeof(fe), hipMemcpyDeviceToHost));

    printf("EC self-consistency: failbits=0x%02X -> %s\n", f, f==0 ? "PASS" : "FAIL");
    printf("computed 2G.x = %016llx%016llx%016llx%016llx\n", g2.n[3], g2.n[2], g2.n[1], g2.n[0]);
    printf("known    2G.x = c6047f9441ed7d6d3045406e95c07cd85c778e4b8cef3ca7abac09b95c709ee5\n");
    bool anchor = (g2.n[3]==0xC6047F9441ED7D6DULL && g2.n[2]==0x3045406E95C07CD8ULL &&
                   g2.n[1]==0x5C778E4B8CEF3CA7ULL && g2.n[0]==0xABAC09B95C709EE5ULL);
    printf("external anchor (2G matches secp256k1): %s\n", anchor ? "YES" : "NO");
    printf("%s\n", (f==0 && anchor) ? "EC POINT OPS OK -> Stage 2b done" : "BUG");
    return (f==0 && anchor) ? 0 : 1;
}
