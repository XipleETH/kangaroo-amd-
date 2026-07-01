// =============================================================================
// Pollard's Kangaroo Algorithm - GPU Kernel (Affine, PER-THREAD batch inversion)
// =============================================================================
// Each thread owns GROUP_N kangaroos and performs a purely SEQUENTIAL Montgomery
// batch inversion (forward prefix products -> ONE fe_inv -> backward pass).
// This eliminates ALL workgroup barriers and idle lanes of the old tree design:
// every lane does a full fe_inv amortized over GROUP_N points, in parallel.
//
// Correctness is byte-for-byte equivalent to the previous tree kernel:
//   - same jump selection (jump_index_from_x + anti-repeat + cycle escape)
//   - same negation-map class representative (even-y) and distance sign flip
//   - same per-point dx==0 guard (substitute 1 before folding into the product)
//   - same DP timing (one DP per kangaroo per dispatch, pre-walk + post-jump)

// -----------------------------------------------------------------------------
// Configuration
// -----------------------------------------------------------------------------

struct Config {
    dp_meta: vec4<u32>,
    num_kangaroos: u32,
    steps_per_call: u32,
    jump_table_size: u32,
    cycle_cap: u32
}

// Must match Rust GpuKangaroo struct layout (128 bytes)!
struct Kangaroo {
    x: array<u32, 8>,
    y: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    is_active: u32,
    cycle_counter: u32,
    repeat_count: u32,
    last_jump: u32,
    _padding: array<u32, 3>
}

const REPEAT_THRESHOLD: u32 = 3u;

// Kangaroos processed per thread. Compile-time constant, fully unrolled below
// (do NOT turn this into an override constant or a dynamically-indexed loop:
//  the AMD SPIR-V compiler needs literal indices to keep group state in VGPRs).
// MUST match `GROUP_N` in src/solver.rs.
const GROUP_N: u32 = 4u;

struct DistinguishedPoint {
    x: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    kangaroo_id: u32,
    _padding: array<u32, 6>
}

// -----------------------------------------------------------------------------
// Buffers
// -----------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> jump_points: array<AffinePoint, 256>;
@group(0) @binding(2) var<storage, read> jump_distances: array<array<u32, 8>, 256>;
@group(0) @binding(3) var<storage, read_write> kangaroos: array<Kangaroo>;
@group(0) @binding(4) var<storage, read_write> dp_buffer: array<DistinguishedPoint>;
@group(0) @binding(5) var<storage, read_write> dp_count: atomic<u32>;

// WORKGROUP_SIZE now only controls launch granularity (no shared memory, no barriers).
override WORKGROUP_SIZE: u32 = 128u;

// -----------------------------------------------------------------------------
// Store distinguished point (from loose fields, so we don't keep a full
// Kangaroo struct alive just to store the rare DP).
// -----------------------------------------------------------------------------

fn store_dp_fields(x: array<u32, 8>, dist: array<u32, 8>, ktype: u32, kangaroo_id: u32) {
    let idx = atomicAdd(&dp_count, 1u);
    if (idx < 65536u) {
        var dp: DistinguishedPoint;
        dp.x = x;
        dp.dist = dist;
        dp.ktype = ktype;
        dp.kangaroo_id = kangaroo_id;
        dp._padding = array<u32, 6>(0u, 0u, 0u, 0u, 0u, 0u);
        dp_buffer[idx] = dp;
    }
}

// -----------------------------------------------------------------------------
// Affine point addition: R = P + Q with precomputed inv = 1/(x2-x1)
//   λ = (y2 - y1) * inv ; x3 = λ² - x1 - x2 ; y3 = λ*(x1 - x3) - y1
// Cost: 2M + 1S.
// -----------------------------------------------------------------------------

fn affine_add_with_inv(
    x1: array<u32, 8>,
    y1: array<u32, 8>,
    x2: array<u32, 8>,
    y2: array<u32, 8>,
    dx_inv: array<u32, 8>
) -> AffinePoint {
    let dy = fe_sub(y2, y1);
    let lambda = fe_mul(dy, dx_inv);

    let lambda_sq = fe_square(lambda);
    let x3 = fe_sub(fe_sub(lambda_sq, x1), x2);

    let x1_minus_x3 = fe_sub(x1, x3);
    let y3 = fe_sub(fe_mul(lambda, x1_minus_x3), y1);

    var result: AffinePoint;
    result.x = x3;
    result.y = y3;
    return result;
}

fn is_distinguished(px: array<u32, 8>) -> bool {
    let full_limbs = min(config.dp_meta.x, 8u);
    var limb = 0u;
    loop {
        if (limb >= full_limbs) {
            break;
        }
        if (px[limb] != 0u) {
            return false;
        }
        limb = limb + 1u;
    }

    let partial_mask = config.dp_meta.y;
    if (partial_mask == 0u || full_limbs >= 8u) {
        return true;
    }

    return (px[full_limbs] & partial_mask) == 0u;
}

fn jump_index_from_x(px: array<u32, 8>) -> u32 {
    let mixed = (px[0] ^ (px[3] >> 11u) ^ (px[5] << 7u)) * 0x9e3779b9u;
    return (mixed >> 24u) & 0xFFu;
}

fn escape_index_from_state(px: array<u32, 8>, kid: u32, cycle_counter: u32, step: u32) -> u32 {
    let seed = px[0]
        ^ px[2]
        ^ (kid * 0x85ebca6bu)
        ^ (cycle_counter * 0xc2b2ae35u)
        ^ step;
    let mixed = seed * 0x27d4eb2du;
    return (mixed >> 24u) & 0xFFu;
}

// -----------------------------------------------------------------------------
// PHASE A (per point): select the effective jump index (anti-repeat + cycle
// escape, identical to the old kernel) and compute dx = jump.x - px.
// Returns the FINAL jump index (after all adjustments) plus the mutated
// cycle/repeat/last_jump state. dx==0 is replaced by 1 so it can never zero the
// running batch product (guard flag returned separately).
// -----------------------------------------------------------------------------

struct PhaseA {
    jidx: u32,
    dx: array<u32, 8>,
    dxwz: bool,
    cyc: u32,
    rep: u32,
    lastj: u32,
}

fn phase_a(
    px: array<u32, 8>,
    valid: bool,
    cyc_in: u32,
    rep_in: u32,
    lastj_in: u32,
    step: u32,
    kid: u32
) -> PhaseA {
    var eidx = jump_index_from_x(px);
    var cyc = cyc_in;
    var rep = rep_in;
    var lastj = lastj_in;

    if (valid) {
        let in_cycle = (cyc > config.cycle_cap)
            || ((rep & 0xFFFFu) > REPEAT_THRESHOLD);
        if (in_cycle) {
            eidx = escape_index_from_state(px, kid, cyc, step);
            cyc = 0u;
            rep = 0u;
        } else {
            if (eidx == lastj) {
                eidx = (eidx + 1u) & 0xFFu;
            }
            lastj = eidx;
        }
    }

    // Stash the FINAL index (after escape / anti-repeat) — dx and the add MUST
    // both use this exact index.
    var dx = fe_sub(jump_points[eidx].x, px);
    var wz = fe_is_zero(dx);
    if (wz) {
        dx = fe_one();
    }

    var out: PhaseA;
    out.jidx = eidx;
    out.dx = dx;
    out.dxwz = wz;
    out.cyc = cyc;
    out.rep = rep;
    out.lastj = lastj;
    return out;
}

// -----------------------------------------------------------------------------
// PHASE B (per point): negation-map add using the recovered dx_inv, plus the
// distance sign flip and cycle/repeat bookkeeping. Identical math to the old
// kernel lines 401-429. Only runs when valid && !dxwz.
// -----------------------------------------------------------------------------

struct PhaseB {
    px: array<u32, 8>,
    py: array<u32, 8>,
    dist: array<u32, 8>,
    cyc: u32,
    rep: u32,
    moved: bool,
}

fn phase_b_add(
    px: array<u32, 8>,
    py: array<u32, 8>,
    dist_in: array<u32, 8>,
    cyc_in: u32,
    rep_in: u32,
    jidx: u32,
    dx_inv: array<u32, 8>,
    valid: bool,
    dxwz: bool
) -> PhaseB {
    var out: PhaseB;
    out.px = px;
    out.py = py;
    out.dist = dist_in;
    out.cyc = cyc_in;
    out.rep = rep_in;
    out.moved = false;

    if (valid && !dxwz) {
        let jp = jump_points[jidx];
        let jd = jump_distances[jidx];

        let y_odd = (py[0] & 1u) != 0u;

        // Class representative {P,-P} -> even-y. x unchanged, so dx_inv is valid.
        var repr_y = py;
        if (y_odd) {
            repr_y = fe_sub(fe_zero(), py);
        }

        let r = affine_add_with_inv(px, repr_y, jp.x, jp.y, dx_inv);
        out.px = r.x;
        out.py = r.y;

        if (y_odd) {
            // (-dist) + jump == jump - dist  (mod 2^256)
            out.dist = scalar_sub_256(jd, dist_in);
        } else {
            out.dist = scalar_add_256(dist_in, jd);
        }

        out.cyc = cyc_in + 1u;
        let new_jump = r.x[0] & 0xFFFFu;
        if (new_jump == (rep_in >> 16u)) {
            let cnt = (rep_in & 0xFFFFu) + 1u;
            out.rep = (new_jump << 16u) | cnt;
        } else {
            out.rep = (new_jump << 16u) | 1u;
        }
        out.moved = true;
    }

    return out;
}

// -----------------------------------------------------------------------------
// Main compute shader — one thread walks GROUP_N kangaroos (strided mapping).
// -----------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let t = global_id.x;
    let num_threads = (config.num_kangaroos + GROUP_N - 1u) / GROUP_N;

    // ---- Per-point state (constant-indexed arrays -> SROA to registers) ----
    var kid: array<u32, 4>;
    var valid: array<bool, 4>;
    var dp_stored: array<bool, 4>;
    var px: array<array<u32, 8>, 4>;
    var py: array<array<u32, 8>, 4>;
    var dist: array<array<u32, 8>, 4>;
    var ktype: array<u32, 4>;
    var cyc: array<u32, 4>;
    var rep: array<u32, 4>;
    var lastj: array<u32, 4>;

    // ---- LOAD (strided: thread t owns kangaroos t, t+T, t+2T, t+3T) ----
    // i = 0
    kid[0] = t;
    valid[0] = false; dp_stored[0] = false;
    px[0] = fe_one(); py[0] = fe_one(); dist[0] = fe_zero();
    ktype[0] = 0u; cyc[0] = 0u; rep[0] = 0u; lastj[0] = 0xFFFFFFFFu;
    if (kid[0] < config.num_kangaroos) {
        let kk = kangaroos[kid[0]];
        ktype[0] = kk.ktype; dist[0] = kk.dist;
        cyc[0] = kk.cycle_counter; rep[0] = kk.repeat_count; lastj[0] = kk.last_jump;
        if (kk.is_active != 0u) { valid[0] = true; px[0] = kk.x; py[0] = kk.y; }
    }
    if (valid[0] && is_distinguished(px[0])) {
        store_dp_fields(px[0], dist[0], ktype[0], kid[0]); dp_stored[0] = true;
    }
    // i = 1
    kid[1] = t + num_threads;
    valid[1] = false; dp_stored[1] = false;
    px[1] = fe_one(); py[1] = fe_one(); dist[1] = fe_zero();
    ktype[1] = 0u; cyc[1] = 0u; rep[1] = 0u; lastj[1] = 0xFFFFFFFFu;
    if (kid[1] < config.num_kangaroos) {
        let kk = kangaroos[kid[1]];
        ktype[1] = kk.ktype; dist[1] = kk.dist;
        cyc[1] = kk.cycle_counter; rep[1] = kk.repeat_count; lastj[1] = kk.last_jump;
        if (kk.is_active != 0u) { valid[1] = true; px[1] = kk.x; py[1] = kk.y; }
    }
    if (valid[1] && is_distinguished(px[1])) {
        store_dp_fields(px[1], dist[1], ktype[1], kid[1]); dp_stored[1] = true;
    }
    // i = 2
    kid[2] = t + 2u * num_threads;
    valid[2] = false; dp_stored[2] = false;
    px[2] = fe_one(); py[2] = fe_one(); dist[2] = fe_zero();
    ktype[2] = 0u; cyc[2] = 0u; rep[2] = 0u; lastj[2] = 0xFFFFFFFFu;
    if (kid[2] < config.num_kangaroos) {
        let kk = kangaroos[kid[2]];
        ktype[2] = kk.ktype; dist[2] = kk.dist;
        cyc[2] = kk.cycle_counter; rep[2] = kk.repeat_count; lastj[2] = kk.last_jump;
        if (kk.is_active != 0u) { valid[2] = true; px[2] = kk.x; py[2] = kk.y; }
    }
    if (valid[2] && is_distinguished(px[2])) {
        store_dp_fields(px[2], dist[2], ktype[2], kid[2]); dp_stored[2] = true;
    }
    // i = 3
    kid[3] = t + 3u * num_threads;
    valid[3] = false; dp_stored[3] = false;
    px[3] = fe_one(); py[3] = fe_one(); dist[3] = fe_zero();
    ktype[3] = 0u; cyc[3] = 0u; rep[3] = 0u; lastj[3] = 0xFFFFFFFFu;
    if (kid[3] < config.num_kangaroos) {
        let kk = kangaroos[kid[3]];
        ktype[3] = kk.ktype; dist[3] = kk.dist;
        cyc[3] = kk.cycle_counter; rep[3] = kk.repeat_count; lastj[3] = kk.last_jump;
        if (kk.is_active != 0u) { valid[3] = true; px[3] = kk.x; py[3] = kk.y; }
    }
    if (valid[3] && is_distinguished(px[3])) {
        store_dp_fields(px[3], dist[3], ktype[3], kid[3]); dp_stored[3] = true;
    }

    // ---- Main walk ----
    for (var step = 0u; step < config.steps_per_call; step++) {
        // ===== PHASE A: select jumps + build dx for all GROUP_N points =====
        var jidx: array<u32, 4>;
        var dxwz: array<bool, 4>;
        var dx: array<array<u32, 8>, 4>;

        let a0 = phase_a(px[0], valid[0], cyc[0], rep[0], lastj[0], step, kid[0]);
        jidx[0] = a0.jidx; dx[0] = a0.dx; dxwz[0] = a0.dxwz;
        cyc[0] = a0.cyc; rep[0] = a0.rep; lastj[0] = a0.lastj;

        let a1 = phase_a(px[1], valid[1], cyc[1], rep[1], lastj[1], step, kid[1]);
        jidx[1] = a1.jidx; dx[1] = a1.dx; dxwz[1] = a1.dxwz;
        cyc[1] = a1.cyc; rep[1] = a1.rep; lastj[1] = a1.lastj;

        let a2 = phase_a(px[2], valid[2], cyc[2], rep[2], lastj[2], step, kid[2]);
        jidx[2] = a2.jidx; dx[2] = a2.dx; dxwz[2] = a2.dxwz;
        cyc[2] = a2.cyc; rep[2] = a2.rep; lastj[2] = a2.lastj;

        let a3 = phase_a(px[3], valid[3], cyc[3], rep[3], lastj[3], step, kid[3]);
        jidx[3] = a3.jidx; dx[3] = a3.dx; dxwz[3] = a3.dxwz;
        cyc[3] = a3.cyc; rep[3] = a3.rep; lastj[3] = a3.lastj;

        // ===== Forward prefix products =====
        var subp: array<array<u32, 8>, 4>;
        subp[0] = dx[0];
        subp[1] = fe_mul(subp[0], dx[1]);
        subp[2] = fe_mul(subp[1], dx[2]);
        subp[3] = fe_mul(subp[2], dx[3]);

        // ===== Single field inversion of the whole-group product =====
        var inv = fe_inv(subp[3]);

        // ===== PHASE B backward: recover 1/dx_i then add, i = 3..0 =====
        // i = 3
        let dxinv3 = fe_mul(subp[2], inv);
        inv = fe_mul(inv, dx[3]);
        let b3 = phase_b_add(px[3], py[3], dist[3], cyc[3], rep[3], jidx[3], dxinv3, valid[3], dxwz[3]);
        px[3] = b3.px; py[3] = b3.py; dist[3] = b3.dist; cyc[3] = b3.cyc; rep[3] = b3.rep;
        if (b3.moved && !dp_stored[3] && is_distinguished(px[3])) {
            store_dp_fields(px[3], dist[3], ktype[3], kid[3]); dp_stored[3] = true;
        }

        // i = 2
        let dxinv2 = fe_mul(subp[1], inv);
        inv = fe_mul(inv, dx[2]);
        let b2 = phase_b_add(px[2], py[2], dist[2], cyc[2], rep[2], jidx[2], dxinv2, valid[2], dxwz[2]);
        px[2] = b2.px; py[2] = b2.py; dist[2] = b2.dist; cyc[2] = b2.cyc; rep[2] = b2.rep;
        if (b2.moved && !dp_stored[2] && is_distinguished(px[2])) {
            store_dp_fields(px[2], dist[2], ktype[2], kid[2]); dp_stored[2] = true;
        }

        // i = 1
        let dxinv1 = fe_mul(subp[0], inv);
        inv = fe_mul(inv, dx[1]);
        let b1 = phase_b_add(px[1], py[1], dist[1], cyc[1], rep[1], jidx[1], dxinv1, valid[1], dxwz[1]);
        px[1] = b1.px; py[1] = b1.py; dist[1] = b1.dist; cyc[1] = b1.cyc; rep[1] = b1.rep;
        if (b1.moved && !dp_stored[1] && is_distinguished(px[1])) {
            store_dp_fields(px[1], dist[1], ktype[1], kid[1]); dp_stored[1] = true;
        }

        // i = 0
        let dxinv0 = inv; // = 1/dx[0]
        let b0 = phase_b_add(px[0], py[0], dist[0], cyc[0], rep[0], jidx[0], dxinv0, valid[0], dxwz[0]);
        px[0] = b0.px; py[0] = b0.py; dist[0] = b0.dist; cyc[0] = b0.cyc; rep[0] = b0.rep;
        if (b0.moved && !dp_stored[0] && is_distinguished(px[0])) {
            store_dp_fields(px[0], dist[0], ktype[0], kid[0]); dp_stored[0] = true;
        }
    }

    // ---- WRITE BACK (unrolled) ----
    if (valid[0]) {
        var out: Kangaroo;
        out.x = px[0]; out.y = py[0]; out.dist = dist[0];
        out.ktype = ktype[0]; out.is_active = 1u;
        out.cycle_counter = cyc[0]; out.repeat_count = rep[0]; out.last_jump = lastj[0];
        out._padding = array<u32, 3>(0u, 0u, 0u);
        kangaroos[kid[0]] = out;
    }
    if (valid[1]) {
        var out: Kangaroo;
        out.x = px[1]; out.y = py[1]; out.dist = dist[1];
        out.ktype = ktype[1]; out.is_active = 1u;
        out.cycle_counter = cyc[1]; out.repeat_count = rep[1]; out.last_jump = lastj[1];
        out._padding = array<u32, 3>(0u, 0u, 0u);
        kangaroos[kid[1]] = out;
    }
    if (valid[2]) {
        var out: Kangaroo;
        out.x = px[2]; out.y = py[2]; out.dist = dist[2];
        out.ktype = ktype[2]; out.is_active = 1u;
        out.cycle_counter = cyc[2]; out.repeat_count = rep[2]; out.last_jump = lastj[2];
        out._padding = array<u32, 3>(0u, 0u, 0u);
        kangaroos[kid[2]] = out;
    }
    if (valid[3]) {
        var out: Kangaroo;
        out.x = px[3]; out.y = py[3]; out.dist = dist[3];
        out.ktype = ktype[3]; out.is_active = 1u;
        out.cycle_counter = cyc[3]; out.repeat_count = rep[3]; out.last_jump = lastj[3];
        out._padding = array<u32, 3>(0u, 0u, 0u);
        kangaroos[kid[3]] = out;
    }
}
