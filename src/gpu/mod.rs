//! GPU compute module

mod buffers;
mod pipeline;

pub use crate::gpu_crypto::{GpuAffinePoint, GpuContext};
pub use buffers::{GpuBuffers, JumpTableData};
pub use pipeline::{KangarooPipeline, WorkgroupVariant};

use bytemuck::{Pod, Zeroable};

/// Kangaroos processed per GPU thread (per-thread sequential batch inversion).
/// MUST match `GROUP_N` in src/shaders/kangaroo_affine.wgsl. Kept <= 32 (the
/// shader's DP-seen bitmask is a single u32).
pub const GROUP_N: u32 = 16;

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuConfig {
    pub dp_meta: [u32; 4],
    pub num_kangaroos: u32,
    pub steps_per_call: u32,
    pub jump_table_size: u32,
    pub cycle_cap: u32,
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuKangaroo {
    pub x: [u32; 8],
    pub y: [u32; 8],
    pub dist: [u32; 8],
    pub ktype: u32,
    pub is_active: u32,
    pub cycle_counter: u32,
    pub repeat_count: u32,
    pub last_jump: u32,
    pub _padding: [u32; 3],
}

#[repr(C)]
#[derive(Clone, Copy, Debug, Pod, Zeroable)]
pub struct GpuDistinguishedPoint {
    pub x: [u32; 8],
    pub dist: [u32; 8],
    pub ktype: u32,
    pub kangaroo_id: u32,
    pub _padding: [u32; 6],
}

const _: [(); 128] = [(); core::mem::size_of::<GpuKangaroo>()];
