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
// AFFINE version — correct but slow (a fe_inv per op). Use scalar_mul_G_jac for setup.
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

// ---- Jacobian coordinates (no inversion during the walk; one inv at the end) ----
struct jac { fe X, Y, Z; };   // affine (X/Z^2, Y/Z^3); Z==0 is infinity

__device__ inline jac jac_double(const jac& p) {   // a = 0 (secp256k1)
    fe A = fe_sqr(p.X);
    fe B = fe_sqr(p.Y);
    fe C = fe_sqr(B);
    fe t = fe_sqr(fe_add(p.X, B)); t = fe_sub(fe_sub(t, A), C); fe D = fe_add(t, t);   // 2((X+B)^2-A-C)
    fe E = fe_add(fe_add(A, A), A);        // 3A
    fe F = fe_sqr(E);
    fe c2 = fe_add(C, C), c4 = fe_add(c2, c2), c8 = fe_add(c4, c4);
    jac r;
    r.X = fe_sub(F, fe_add(D, D));
    r.Y = fe_sub(fe_mul(E, fe_sub(D, r.X)), c8);
    r.Z = fe_add(fe_mul(p.Y, p.Z), fe_mul(p.Y, p.Z));   // 2YZ
    return r;
}

// P (Jacobian) + Q (affine qx,qy)
__device__ inline jac jac_add_affine(const jac& p, const fe& qx, const fe& qy) {
    if (fe_is_zero(p.Z)) { jac r; r.X=qx; r.Y=qy; r.Z=fe_one(); return r; }
    fe Z1Z1 = fe_sqr(p.Z);
    fe U2 = fe_mul(qx, Z1Z1);
    fe S2 = fe_mul(fe_mul(qy, p.Z), Z1Z1);   // qy*Z1^3
    fe H = fe_sub(U2, p.X);
    if (fe_is_zero(H)) {
        if (fe_eq(S2, p.Y)) return jac_double(p);
        jac r; r.X=fe_one(); r.Y=fe_one(); r.Z=fe_zero(); return r;   // P == -Q
    }
    fe HH = fe_sqr(H);
    fe I = fe_add(fe_add(HH, HH), fe_add(HH, HH));   // 4HH
    fe J = fe_mul(H, I);
    fe r2 = fe_sub(S2, p.Y); r2 = fe_add(r2, r2);    // r = 2(S2-Y1)
    fe V = fe_mul(p.X, I);
    jac out;
    out.X = fe_sub(fe_sub(fe_sqr(r2), J), fe_add(V, V));
    fe y1j = fe_mul(p.Y, J);
    out.Y = fe_sub(fe_mul(r2, fe_sub(V, out.X)), fe_add(y1j, y1j));
    out.Z = fe_sub(fe_sub(fe_sqr(fe_add(p.Z, H)), Z1Z1), HH);   // 2*Z1*H
    return out;
}

// k * G in Jacobian, converted to affine with a SINGLE inversion. Fast setup path.
__device__ inline jpt scalar_mul_G_jac(const fe& k) {
    jac R; R.X = fe_one(); R.Y = fe_one(); R.Z = fe_zero();   // infinity
    jpt G = J_G();
    for (int limb = 3; limb >= 0; --limb) {
        u64 e = k.n[limb];
        for (int b = 63; b >= 0; --b) {
            R = jac_double(R);
            if ((e >> b) & 1ULL) R = jac_add_affine(R, G.x, G.y);
        }
    }
    jpt out;
    if (fe_is_zero(R.Z)) { out.inf = true; out.x = fe_zero(); out.y = fe_zero(); return out; }
    fe zi = fe_inv(R.Z);
    fe zi2 = fe_sqr(zi);
    out.inf = false;
    out.x = fe_mul(R.X, zi2);
    out.y = fe_mul(R.Y, fe_mul(zi2, zi));
    return out;
}
