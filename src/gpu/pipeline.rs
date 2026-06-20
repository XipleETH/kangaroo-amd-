//! Compute pipeline setup

use super::{GpuConfig, GpuContext};
use anyhow::Result;
use std::collections::HashMap;
use std::sync::{Arc, Mutex, OnceLock};
use tracing::info;
use wgpu::{BindGroupLayout, ComputePipeline};

#[derive(Clone, Copy, Debug, Eq, Hash, PartialEq)]
#[allow(dead_code)]
pub enum WorkgroupVariant {
    Wg64,
    Wg128,
}

impl WorkgroupVariant {
    pub fn size(self) -> u32 {
        match self {
            Self::Wg64 => 64,
            Self::Wg128 => 128,
        }
    }
}

/// Kangaroo compute pipeline (Clone is cheap - wgpu types are Arc-wrapped)
#[derive(Clone)]
pub struct KangarooPipeline {
    pub pipeline: Arc<ComputePipeline>,
    pub bind_group_layout: Arc<BindGroupLayout>,
    pub variant: WorkgroupVariant,
}

impl KangarooPipeline {
    pub fn new(ctx: &GpuContext, variant: WorkgroupVariant) -> Result<Self> {
        static PIPELINE_CACHE: OnceLock<
            Mutex<HashMap<(usize, WorkgroupVariant), KangarooPipeline>>,
        > = OnceLock::new();

        let device_key = Arc::as_ptr(&ctx.device) as usize;
        let cache = PIPELINE_CACHE.get_or_init(|| Mutex::new(HashMap::new()));
        if let Some(pipeline) = cache
            .lock()
            .expect("pipeline cache poisoned")
            .get(&(device_key, variant))
            .cloned()
        {
            return Ok(pipeline);
        }

        info!("Loading Jacobian shader sources (no fe_inv - AMD compatible)...");

        // Use the Jacobian kernel which eliminates fe_inv from the GPU
        // This prevents AMD's shader compiler from hanging on the massive
        // secp256k1 field inversion function.
        let field = crate::gpu_crypto::shaders::FIELD_WGSL;
        let curve = crate::gpu_crypto::shaders::CURVE_WGSL;
        let kangaroo = include_str!("../shaders/kangaroo_jacobian.wgsl");

        let constants = [("WORKGROUP_SIZE", variant.size() as f64)];

        info!("Creating shader module...");
        let shader =
            ctx.create_shader_module("Kangaroo Jacobian Shader", &[field, curve, kangaroo]);
        info!("Shader module created");

        info!("Creating bind group layout...");
        // Create bind group layout
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Kangaroo Bind Group Layout"),
                    entries: &[
                        // Config (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(
                                    std::mem::size_of::<GpuConfig>() as u64,
                                ),
                            },
                            count: None,
                        },
                        // Jump points (storage, read_only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Jump distances (storage, read_only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Kangaroos (storage, read_write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // DP buffer (storage, read_write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // DP count (storage, read_write atomic)
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        info!("Bind group layout created");

        info!("Creating pipeline layout...");
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Kangaroo Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });
        info!("Pipeline layout created");

        info!("Creating compute pipeline...");
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Kangaroo Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });
        info!("Compute pipeline created");

        let pipeline = Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
            variant,
        };

        cache
            .lock()
            .expect("pipeline cache poisoned")
            .insert((device_key, variant), pipeline.clone());

        Ok(pipeline)
    }
}

/// Normalization compute pipeline for converting Jacobian → Affine coordinates
/// before DP extraction. No pipeline cache — each GPU gets its own instance.
#[derive(Clone)]
pub struct NormalizePipeline {
    pub pipeline: Arc<ComputePipeline>,
    #[allow(dead_code)]
    pub bind_group_layout: Arc<BindGroupLayout>,
}

impl NormalizePipeline {
    pub fn new(ctx: &GpuContext, variant: WorkgroupVariant) -> Result<Self> {
        info!("Loading normalization shader sources...");

        let field = crate::gpu_crypto::shaders::FIELD_WGSL;
        let curve = crate::gpu_crypto::shaders::CURVE_WGSL;
        let normalize = include_str!("../shaders/normalize_dp.wgsl");

        let constants = [("WORKGROUP_SIZE", variant.size() as f64)];

        info!("Creating normalize shader module...");
        let shader = ctx.create_shader_module("Normalize DP Shader", &[field, curve, normalize]);
        info!("Normalize shader module created");

        info!("Creating normalize bind group layout...");
        let bind_group_layout =
            ctx.device
                .create_bind_group_layout(&wgpu::BindGroupLayoutDescriptor {
                    label: Some("Normalize Bind Group Layout"),
                    entries: &[
                        // Config (uniform)
                        wgpu::BindGroupLayoutEntry {
                            binding: 0,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Uniform,
                                has_dynamic_offset: false,
                                min_binding_size: wgpu::BufferSize::new(
                                    std::mem::size_of::<GpuConfig>() as u64,
                                ),
                            },
                            count: None,
                        },
                        // Jump points (storage, read_only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 1,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Jump distances (storage, read_only)
                        wgpu::BindGroupLayoutEntry {
                            binding: 2,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: true },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // Kangaroos (storage, read_write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 3,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // DP buffer (storage, read_write)
                        wgpu::BindGroupLayoutEntry {
                            binding: 4,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                        // DP count (storage, read_write atomic)
                        wgpu::BindGroupLayoutEntry {
                            binding: 5,
                            visibility: wgpu::ShaderStages::COMPUTE,
                            ty: wgpu::BindingType::Buffer {
                                ty: wgpu::BufferBindingType::Storage { read_only: false },
                                has_dynamic_offset: false,
                                min_binding_size: None,
                            },
                            count: None,
                        },
                    ],
                });
        info!("Normalize bind group layout created");

        info!("Creating normalize pipeline layout...");
        let pipeline_layout = ctx
            .device
            .create_pipeline_layout(&wgpu::PipelineLayoutDescriptor {
                label: Some("Normalize Pipeline Layout"),
                bind_group_layouts: &[&bind_group_layout],
                immediate_size: 0,
            });
        info!("Normalize pipeline layout created");

        info!("Creating normalize compute pipeline...");
        let pipeline = ctx
            .device
            .create_compute_pipeline(&wgpu::ComputePipelineDescriptor {
                label: Some("Normalize Compute Pipeline"),
                layout: Some(&pipeline_layout),
                module: &shader,
                entry_point: Some("normalize_main"),
                compilation_options: wgpu::PipelineCompilationOptions {
                    constants: &constants,
                    zero_initialize_workgroup_memory: false,
                },
                cache: None,
            });
        info!("Normalize compute pipeline created");

        Ok(Self {
            pipeline: Arc::new(pipeline),
            bind_group_layout: Arc::new(bind_group_layout),
        })
    }
}
