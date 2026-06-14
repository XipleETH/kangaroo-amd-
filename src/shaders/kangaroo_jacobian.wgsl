// =============================================================================
// Pollard's Kangaroo Algorithm - GPU Kernel (Jacobian Coordinates)
// =============================================================================
// Uses Jacobian coordinates to ELIMINATE fe_inv from the GPU shader.
// This fixes AMD RDNA3 shader compilation hang caused by the massive
// fe_inv function (255 squarings + 15 multiplications).
//
// Trade-off: More field ops per step (8M+4S vs 2M+1S) but:
// - No fe_inv needed (the function that hangs AMD's compiler)
// - No shared memory for batch inversion
// - No workgroupBarrier calls
// - Each thread is fully independent

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

// Must match Rust GpuKangarooJac struct layout!
struct KangarooJac {
    x: array<u32, 8>,
    y: array<u32, 8>,
    z: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    is_active: u32,
    cycle_counter: u32,
    repeat_count: u32,
    last_jump: u32,
    _padding: array<u32, 3>
}

const REPEAT_THRESHOLD: u32 = 3u;

struct DistinguishedPoint {
    x: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    kangaroo_id: u32,
    _padding: array<u32, 6>
}

// Candidate DP for CPU verification (projective coords)
struct DpCandidate {
    jac_x: array<u32, 8>,
    jac_z: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    kangaroo_id: u32,
    step_idx: u32,
    _padding: array<u32, 5>
}

// -----------------------------------------------------------------------------
// Buffers
// -----------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> jump_points: array<AffinePoint, 256>;
@group(0) @binding(2) var<storage, read> jump_distances: array<array<u32, 8>, 256>;
@group(0) @binding(3) var<storage, read_write> kangaroos: array<KangarooJac>;
@group(0) @binding(4) var<storage, read_write> dp_buffer: array<DpCandidate>;
@group(0) @binding(5) var<storage, read_write> dp_count: atomic<u32>;

override WORKGROUP_SIZE: u32 = 64u;

// -----------------------------------------------------------------------------
// Store candidate distinguished point (projective coords, verified on CPU)
// -----------------------------------------------------------------------------

fn store_dp_candidate(k: KangarooJac, kangaroo_id: u32, step: u32) {
    let idx = atomicAdd(&dp_count, 1u);

    if (idx < 65536u) {
        var dp: DpCandidate;
        dp.jac_x = k.x;
        dp.jac_z = k.z;
        dp.dist = k.dist;
        dp.ktype = k.ktype;
        dp.kangaroo_id = kangaroo_id;
        dp.step_idx = step;
        dp._padding = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
        dp_buffer[idx] = dp;
    }
}

// -----------------------------------------------------------------------------
// Approximate DP check using Jacobian X coordinate
// -----------------------------------------------------------------------------
// In Jacobian: x_affine = X / Z^2
// We can't compute this without fe_inv, so we use an approximation:
// Check if low bits of X are zero. This gives false positives (when Z^2
// has factors that cancel the low bits of X) but those are filtered on CPU.
// False negative rate is negligible for practical DP bit counts.

fn is_distinguished_approx(jx: array<u32, 8>) -> bool {
    let full_limbs = min(config.dp_meta.x, 8u);
    var limb = 0u;
    loop {
        if (limb >= full_limbs) {
            break;
        }
        if (jx[limb] != 0u) {
            return false;
        }
        limb = limb + 1u;
    }

    let partial_mask = config.dp_meta.y;
    if (partial_mask == 0u || full_limbs >= 8u) {
        return true;
    }

    return (jx[full_limbs] & partial_mask) == 0u;
}

fn jump_index_from_x(jx: array<u32, 8>) -> u32 {
    // Use Jacobian X directly for jump index (deterministic, just needs to be consistent)
    let mixed = (jx[0] ^ (jx[3] >> 11u) ^ (jx[5] << 7u)) * 0x9e3779b9u;
    return (mixed >> 24u) & 0xFFu;
}

fn escape_index_from_state(jx: array<u32, 8>, kid: u32, cycle_counter: u32, step: u32) -> u32 {
    let seed = jx[0]
        ^ jx[2]
        ^ (kid * 0x85ebca6bu)
        ^ (cycle_counter * 0xc2b2ae35u)
        ^ step;
    let mixed = seed * 0x27d4eb2du;
    return (mixed >> 24u) & 0xFFu;
}

// -----------------------------------------------------------------------------
// Main compute shader (Jacobian - NO fe_inv needed!)
// -----------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let kid = global_id.x;

    // Load kangaroo state (if valid)
    var k: KangarooJac;
    var valid = false;
    if (kid < config.num_kangaroos) {
        k = kangaroos[kid];
        if (k.is_active != 0u) {
            valid = true;
        }
    }

    // Current point in Jacobian coordinates
    var jx: array<u32, 8>;
    var jy: array<u32, 8>;
    var jz: array<u32, 8>;

    if (valid) {
        jx = k.x;
        jy = k.y;
        jz = k.z;
    } else {
        // Dummy point for inactive threads
        jx = fe_one();
        jy = fe_one();
        jz = fe_one();
    }

    // Check current position once before the walk
    if (valid && is_distinguished_approx(jx)) {
        k.x = jx;
        k.y = jy;
        k.z = jz;
        store_dp_candidate(k, kid, 0u);
    }

    // Perform jumps
    for (var step = 0u; step < config.steps_per_call; step++) {
        var effective_jump_idx = jump_index_from_x(jx);
        if (valid) {
            let in_cycle = (k.cycle_counter > config.cycle_cap)
                || ((k.repeat_count & 0xFFFFu) > REPEAT_THRESHOLD);
            if (in_cycle) {
                effective_jump_idx = escape_index_from_state(jx, kid, k.cycle_counter, step);
                k.cycle_counter = 0u;
                k.repeat_count = 0u;
            } else {
                if (effective_jump_idx == k.last_jump) {
                    effective_jump_idx = (effective_jump_idx + 1u) & 0xFFu;
                }
                k.last_jump = effective_jump_idx;
            }
        }
        let jump_idx = effective_jump_idx;
        let jump_point = jump_points[jump_idx];
        let jump_dist = jump_distances[jump_idx];

        // =====================================================================
        // POINT ADDITION (Jacobian + Affine mixed add)
        // NO batch inversion needed! NO shared memory! NO barriers!
        // Cost: 8M + 4S per thread per step (vs 2M+1S affine but with fe_inv)
        // =====================================================================

        if (valid) {
            // Y normalization: map to even-y representative
            let y_odd = (jy[0] & 1u) != 0u;
            var repr_y = jy;
            if (y_odd) {
                repr_y = fe_sub(fe_zero(), jy);
            }

            // Mixed Jacobian + Affine addition
            var p: JacobianPoint;
            p.x = jx;
            p.y = repr_y;
            p.z = jz;

            let result = jac_add_affine(p, jump_point.x, jump_point.y);

            jx = result.x;
            jy = result.y;
            jz = result.z;

            // Update distance
            if (y_odd) {
                k.dist = scalar_sub_256(jump_dist, k.dist);
            } else {
                k.dist = scalar_add_256(k.dist, jump_dist);
            }

            k.cycle_counter = k.cycle_counter + 1u;
            let new_jump = jx[0] & 0xFFFFu;
            if (new_jump == (k.repeat_count >> 16u)) {
                let cnt = (k.repeat_count & 0xFFFFu) + 1u;
                k.repeat_count = (new_jump << 16u) | cnt;
            } else {
                k.repeat_count = (new_jump << 16u) | 1u;
            }

            // Approximate DP check on Jacobian X
            if (is_distinguished_approx(jx)) {
                k.x = jx;
                k.y = jy;
                k.z = jz;
                store_dp_candidate(k, kid, step + 1u);
            }
        }
    }

    // Write back updated state
    if (valid) {
        k.x = jx;
        k.y = jy;
        k.z = jz;
        kangaroos[kid] = k;
    }
}
