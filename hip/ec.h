// ec.h — secp256k1 point ops with infinity flag + scalar multiplication.
// Validated by ec_test.hip and scalar_test.hip on the RX 7800 XT.
#pragma once
#include "secp256k1.h"

struct jpt { fe x, y; bool inf; };  // affine point + point-at-infinity flag

__device__ inline jpt J_G() {
    jpt g; g.inf = false;
    g.x = fe_set(0x59F2815B16F81798ULL, 0x029BFCDB2DCE28D9ULL, 0x55A06295CE870B07ULL, 0x79BE667EF9DCBBACULL);
    g.y = fe_set(0x9C47D08FFB10D4B8ULL, 0xFD17B448A6855419ULL, 0x5DA4FBFC0E1108A8ULL, 0x483ADA7726A3C465ULL);
    return g;
}

__device__ inline jpt j_double(const jpt& p) {
    if (p.inf) return p;
    fe x2 = fe_sqr(p.x);
    fe num = fe_add(fe_add(x2, x2), x2);   // 3x^2
    fe den = fe_add(p.y, p.y);             // 2y
    fe lam = fe_mul(num, fe_inv(den));
    jpt r; r.inf = false;
    r.x = fe_sub(fe_sqr(lam), fe_add(p.x, p.x));
    r.y = fe_sub(fe_mul(lam, fe_sub(p.x, r.x)), p.y);
    return r;
}

__device__ inline jpt j_add(const jpt& p, const jpt& q) {
    if (p.inf) return q;
    if (q.inf) return p;
    if (fe_eq(p.x, q.x)) {
        if (fe_eq(p.y, q.y)) return j_double(p);
        jpt r; r.inf = true; return r;   // P + (-P) = O
    }
    fe lam = fe_mul(fe_sub(q.y, p.y), fe_inv(fe_sub(q.x, p.x)));
    jpt r; r.inf = false;
    r.x = fe_sub(fe_sub(fe_sqr(lam), p.x), q.x);
    r.y = fe_sub(fe_mul(lam, fe_sub(p.x, r.x)), p.y);
    return r;
}

// k * G via double-and-add (MSB first). k is a 256-bit scalar in `fe` limbs.
__device__ inline jpt scalar_mul_G(const fe& k) {
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
