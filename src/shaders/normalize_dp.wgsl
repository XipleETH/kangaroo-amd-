// =============================================================================
// Jacobian → Affine Normalization + Exact Distinguished Point Check
// =============================================================================
// Runs AFTER kangaroo_jacobian.wgsl walk shader.
// Converts each kangaroo's Jacobian (X, Y, Z) to Affine (X/Z², Y/Z³)
// and performs an EXACT DP check on the affine X coordinate.
//
// This shader is CONCATENATED with field.wgsl and curve.wgsl at compile time.
// Do NOT redeclare: fe_mul, fe_square, fe_sqr_n, fe_inv, fe_sub, fe_add,
//                   fe_zero, fe_one, fe_is_zero, JacobianPoint, AffinePoint, etc.

// -----------------------------------------------------------------------------
// Struct definitions (must match walk shader and Rust layout exactly)
// Each shader is compiled as a SEPARATE pipeline, so structs are defined here.
// -----------------------------------------------------------------------------

struct Config {
    dp_meta: vec4<u32>,
    num_kangaroos: u32,
    steps_per_call: u32,
    jump_table_size: u32,
    cycle_cap: u32
}

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

struct DpCandidate {
    jac_x: array<u32, 8>,
    jac_z: array<u32, 8>,
    dist: array<u32, 8>,
    ktype: u32,
    kangaroo_id: u32,
    step_idx: u32,
    _padding: array<u32, 5>
}

// Note: AffinePoint is already defined in curve.wgsl (concatenated at compile time)

// -----------------------------------------------------------------------------
// Buffer bindings (MUST match kangaroo_jacobian.wgsl exactly)
// -----------------------------------------------------------------------------

@group(0) @binding(0) var<uniform> config: Config;
@group(0) @binding(1) var<storage, read> jump_points: array<AffinePoint, 256>;
@group(0) @binding(2) var<storage, read> jump_distances: array<array<u32, 8>, 256>;
@group(0) @binding(3) var<storage, read_write> kangaroos: array<KangarooJac>;
@group(0) @binding(4) var<storage, read_write> dp_buffer: array<DpCandidate>;
@group(0) @binding(5) var<storage, read_write> dp_count: atomic<u32>;

override WORKGROUP_SIZE: u32 = 64u;

// -----------------------------------------------------------------------------
// Check if Z is already [1, 0, 0, 0, 0, 0, 0, 0] (already affine)
// -----------------------------------------------------------------------------

fn z_is_one(z: array<u32, 8>) -> bool {
    return z[0] == 1u && z[1] == 0u && z[2] == 0u && z[3] == 0u &&
           z[4] == 0u && z[5] == 0u && z[6] == 0u && z[7] == 0u;
}

// -----------------------------------------------------------------------------
// Exact DP check on affine X coordinate
// dp_meta.x = number of full zero limbs required
// dp_meta.y = partial bitmask for the next limb (bits that must be zero)
// -----------------------------------------------------------------------------

fn is_distinguished_exact(ax: array<u32, 8>) -> bool {
    let full_limbs = min(config.dp_meta.x, 8u);

    // Check that X_affine[0..full_limbs] are all zero
    var limb = 0u;
    loop {
        if (limb >= full_limbs) {
            break;
        }
        if (ax[limb] != 0u) {
            return false;
        }
        limb = limb + 1u;
    }

    // Check partial mask on next limb
    let partial_mask = config.dp_meta.y;
    if (partial_mask == 0u || full_limbs >= 8u) {
        return true;
    }

    return (ax[full_limbs] & partial_mask) == 0u;
}

// -----------------------------------------------------------------------------
// Main entry point: normalize Jacobian → Affine, check for DPs
// -----------------------------------------------------------------------------

@compute @workgroup_size(WORKGROUP_SIZE)
fn normalize_main(@builtin(global_invocation_id) global_id: vec3<u32>) {
    let kid = global_id.x;

    // Bounds check
    if (kid >= config.num_kangaroos) {
        return;
    }

    // Load kangaroo state
    var k = kangaroos[kid];

    // Skip inactive kangaroos
    if (k.is_active == 0u) {
        return;
    }

    // --- Normalize Jacobian (X, Y, Z) to Affine (X/Z², Y/Z³) ---

    var x_affine = k.x;
    var y_affine = k.y;

    if (!z_is_one(k.z)) {
        // Z != 1: perform full inversion and normalization
        // Z_inv = Z^(-1) mod p
        let z_inv = fe_inv(k.z);

        // Z_inv2 = Z_inv² = Z^(-2)
        let z_inv2 = fe_square(z_inv);

        // Z_inv3 = Z_inv² * Z_inv = Z^(-3)
        let z_inv3 = fe_mul(z_inv2, z_inv);

        // X_affine = X * Z^(-2)
        x_affine = fe_mul(k.x, z_inv2);

        // Y_affine = Y * Z^(-3)
        y_affine = fe_mul(k.y, z_inv3);
    }

    // --- Write normalized coordinates back ---
    k.x = x_affine;
    k.y = y_affine;
    k.z = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);

    // --- Exact DP check on affine X ---
    if (is_distinguished_exact(x_affine)) {
        let idx = atomicAdd(&dp_count, 1u);

        if (idx < 65536u) {
            var dp: DpCandidate;
            dp.jac_x = x_affine;
            dp.jac_z = array<u32, 8>(1u, 0u, 0u, 0u, 0u, 0u, 0u, 0u);
            dp.dist = k.dist;
            dp.ktype = k.ktype;
            dp.kangaroo_id = kid;
            dp.step_idx = 0u;
            dp._padding = array<u32, 5>(0u, 0u, 0u, 0u, 0u);
            dp_buffer[idx] = dp;
        }
    }

    // --- Write normalized kangaroo back to buffer ---
    kangaroos[kid] = k;
}
