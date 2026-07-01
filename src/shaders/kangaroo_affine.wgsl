// =============================================================================
// Pollard's Kangaroo - GPU Kernel (global-backed group, per-thread inversion)
// =============================================================================
// Each thread owns GROUP_N kangaroos and performs a SEQUENTIAL Montgomery batch
// inversion. The group point state lives in the global `kangaroos` buffer
// (streamed per step); only the rolling product / inverse stays in registers.
// This keeps the register footprint low (high occupancy) while amortizing the
// single fe_inv over a large GROUP_N (few inversions), the RCKangaroo layout.
//
// Per step, per thread:
//   PASS 1 (forward): for each of GROUP_N kangaroos, re-derive dx = jump.x - px
//     (jump selection is deterministic and side-effect-free here), fold into the
//     running product `acc`, store the prefix product to the scratch buffer.
//   fe_inv(acc)  -- ONE inversion for the whole group.
//   PASS 2 (backward): recover 1/dx_i (scratch prefix * rolling inv), re-run the
//     SAME deterministic jump selection to get the mutated walk state, do the
//     negation-map add, and write the kangaroo back.
//
// Correctness is identical to the register kernel: same jump selection, negation
// map, distance sign, dx==0 guard, and one DP per kangaroo per dispatch.

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

// Kangaroos processed per thread. MUST match `GROUP_N` in src/gpu/mod.rs.
// Kept <= 32 (DP-seen bitmask is a single u32). N=16 measured best on RDNA3.
const GROUP_N: u32 = 16u;

struct DistinguishedPoint {
    x: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    kangaroo_id: u32,
    _padding: array<u32, 6>
}

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> jump_points: array<AffinePoint, 256>;
@group(0) @binding(2) var<storage, read> jump_distances: array<array<u32, 8>, 256>;
@group(0) @binding(3) var<storage, read_write> kangaroos: array<Kangaroo>;
@group(0) @binding(4) var<storage, read_write> dp_buffer: array<DistinguishedPoint>;
@group(0) @binding(5) var<storage, read_write> dp_count: atomic<u32>;
// Prefix-product scratch, SoA-indexed [i * num_threads + t] for coalescing.
@group(0) @binding(6) var<storage, read_write> subp_scratch: array<array<u32, 8>>;

override WORKGROUP_SIZE: u32 = 128u;

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

// R = P + Q with precomputed inv = 1/(x2-x1). 2M + 1S.
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

// PHASE A: deterministic, side-effect-free jump selection + dx. Returns the
// FINAL jump index (after escape / anti-repeat) and the mutated walk state.
// Safe to call twice (pass 1 and pass 2) with the same inputs -> same output.
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

// PHASE B: negation-map add + distance sign + cycle/repeat bookkeeping.
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
        var repr_y = py;
        if (y_odd) {
            repr_y = fe_sub(fe_zero(), py);
        }
        let r = affine_add_with_inv(px, repr_y, jp.x, jp.y, dx_inv);
        out.px = r.x;
        out.py = r.y;
        if (y_odd) {
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
// Main compute shader
// -----------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let t = global_id.x;
    let num_threads = (config.num_kangaroos + GROUP_N - 1u) / GROUP_N;
    // Guard padding threads: each kangaroo is owned by exactly one t < num_threads
    // (strided kid = t + i*num_threads), so this prevents double-processing and
    // keeps scratch indices in bounds.
    if (t >= num_threads) {
        return;
    }

    // DP-already-stored bitmask for this thread's GROUP_N kangaroos (N <= 32).
    var dp_mask: u32 = 0u;

    // ---- Pre-walk DP check on the START positions ----
    for (var i = 0u; i < GROUP_N; i = i + 1u) {
        let kid = t + i * num_threads;
        if (kid < config.num_kangaroos) {
            let k = kangaroos[kid];
            if (k.is_active != 0u && is_distinguished(k.x)) {
                store_dp_fields(k.x, k.dist, k.ktype, kid);
                dp_mask = dp_mask | (1u << i);
            }
        }
    }

    for (var step = 0u; step < config.steps_per_call; step = step + 1u) {
        // ===== PASS 1 (forward): build prefix products into scratch =====
        var acc = fe_one();
        for (var i = 0u; i < GROUP_N; i = i + 1u) {
            let kid = t + i * num_threads;
            var dxv = fe_one();
            if (kid < config.num_kangaroos) {
                let k = kangaroos[kid];
                if (k.is_active != 0u) {
                    let a = phase_a(k.x, true, k.cycle_counter, k.repeat_count, k.last_jump, step, kid);
                    dxv = a.dx;
                }
            }
            acc = fe_mul(acc, dxv);
            subp_scratch[i * num_threads + t] = acc;
        }

        // ===== Single inversion of the whole-group product =====
        var inv = fe_inv(acc);

        // ===== PASS 2 (backward): recover 1/dx_i, add, write back =====
        for (var j = 0u; j < GROUP_N; j = j + 1u) {
            let i = GROUP_N - 1u - j;
            let kid = t + i * num_threads;
            if (kid >= config.num_kangaroos) {
                continue;
            }
            let k = kangaroos[kid];
            if (k.is_active == 0u) {
                continue;  // dx==1 for inactive: inv is unchanged, so nothing to do
            }

            // Re-run the deterministic selection (same result as pass 1).
            let a = phase_a(k.x, true, k.cycle_counter, k.repeat_count, k.last_jump, step, kid);

            var dx_inv: array<u32, 8>;
            if (i == 0u) {
                dx_inv = inv;
            } else {
                let sp = subp_scratch[(i - 1u) * num_threads + t];
                dx_inv = fe_mul(sp, inv);
                inv = fe_mul(inv, a.dx);
            }

            let b = phase_b_add(k.x, k.y, k.dist, a.cyc, a.rep, a.jidx, dx_inv, true, a.dxwz);

            var out: Kangaroo;
            out.x = b.px;
            out.y = b.py;
            out.dist = b.dist;
            out.ktype = k.ktype;
            out.is_active = k.is_active;
            out.cycle_counter = b.cyc;
            out.repeat_count = b.rep;
            out.last_jump = a.lastj;
            out._padding = array<u32, 3>(0u, 0u, 0u);
            kangaroos[kid] = out;

            if (b.moved && ((dp_mask >> i) & 1u) == 0u && is_distinguished(b.px)) {
                store_dp_fields(b.px, b.dist, k.ktype, kid);
                dp_mask = dp_mask | (1u << i);
            }
        }
    }
}
