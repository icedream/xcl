//! Memory census: bytes by DXGI format and duplicate texture page tracking.
//!
//! `XCL_CENSUS_BYTES=1`: extend the census summary to report total bytes by DXGI format,
//! and for BC7 formats (98=BC7_UNORM, 99=BC7_UNORM_SRGB) the bytes whose resource has no
//! alpha-bearing flag as a separate bucket.
//!
//! `XCL_PAGE_DEDUP_CENSUS=1`: hash texture upload content per page as it passes `upload_batch`
//! and count, at exit, how many uploaded bytes had a page hash already seen.

use std::collections::BTreeMap;
use std::sync::{LazyLock, Mutex, Once};

use windows::Win32::Graphics::Direct3D12::{
    D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET, D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS,
    D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS, D3D12_RESOURCE_DIMENSION_BUFFER,
};

/// Bytes by DXGI format (the format value as u32).
/// For BC7 formats, we also track separately the bytes in resources without alpha-bearing flags.
#[derive(Debug, Default, Clone)]
pub struct FormatBytes {
    /// Total bytes for this format across all resources
    pub total: u64,
    /// For BC7: bytes in resources without ALLOW_RENDER_TARGET, ALLOW_UNORDERED_ACCESS, or
    /// ALLOW_SIMULTANEOUS_ACCESS flags (opaque, no alpha usage)
    pub bc7_opaque: u64,
}

/// Alpha-bearing resource flags: if any of these is set, the resource may have alpha.
/// From D3D12_RESOURCE_FLAGS enum:
/// - ALLOW_RENDER_TARGET (0x00000001)
/// - ALLOW_DEPTH_STENCIL (0x00000002)
/// - ALLOW_UNORDERED_ACCESS (0x00000004)
/// - DENY_SHADER_RESOURCE (0x00000008)
/// - ALLOW_CROSS_ADAPTER (0x00000010)
/// - ALLOW_SIMULTANEOUS_ACCESS (0x00000020)
/// - ALLOW_ALL_SHADER_RESOURCE (0x00000040)
///
/// For our purposes, a resource "has alpha" if it can be a render target, UAV, or
/// simultaneously accessed (which often correlates with having an alpha channel that
/// matters). The conservative interpretation: if none of these flags are set, the
/// resource is likely a static texture that can be downsampled.
const ALPHA_BEARING_FLAGS: u32 = D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 as u32
    | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32
    | D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS.0 as u32;

/// BC7 format values (DXGI_FORMAT enumeration values)
const DXGI_FORMAT_BC7_UNORM: u32 = 98;
const DXGI_FORMAT_BC7_UNORM_SRGB: u32 = 99;

/// Is this a BC7 format?
fn is_bc7(format: u32) -> bool {
    format == DXGI_FORMAT_BC7_UNORM || format == DXGI_FORMAT_BC7_UNORM_SRGB
}

/// Does the resource have any alpha-bearing flags?
fn has_alpha_flags(flags: u32) -> bool {
    flags & ALPHA_BEARING_FLAGS != 0
}

/// Compute the size in bytes of a texture resource from its description.
///
/// For block-compressed formats (BC1-BC7), size is computed in blocks, not pixels.
/// Each block is 4x4 texels and occupies a format-specific number of bytes:
/// - BC1: 8 bytes per block (4x4 = 16 pixels, 0.5 bpp)
/// - BC2/BC3: 16 bytes per block (4x4 = 16 pixels, 1 bpp)
/// - BC4/BC5: 8 bytes per block (4x4 = 16 pixels, 0.5 bpp)
/// - BC6H/BC7: 16 bytes per block (4x4 = 16 pixels, 1 bpp)
///
/// Formula: blocks_w * blocks_h * blocks_d * bytes_per_block
/// where blocks_w = (width + 3) / 4, blocks_h = (height + 3) / 4, blocks_d = (depth + 3) / 4
///
/// For uncompressed formats: width * height * depth * bytes_per_pixel
///
/// For buffers: width (size in bytes)
fn compute_resource_size(dim: u32, format: u32, width: u64, height: u32, depth: u32, mip_levels: u32) -> u64 {
    // For buffers, the size is simply the width
    if dim == D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32 {
        return width;
    }

    // Get bytes per block or bytes per pixel for the format
    let (bytes_per_block, is_block_compressed) = match format {
        // BC1 formats: 8 bytes per 4x4 block
        71 | 72 | 73 | 74 => (8, true), // BC1_UNORM, BC1_UNORM_SRGB, BC1_TYPELESS, BC1_UNORM_SRGB (duplicates for safety)
        // BC2 formats: 16 bytes per 4x4 block
        75 | 76 | 77 | 78 => (16, true),
        // BC3 formats: 16 bytes per 4x4 block
        79 | 80 | 81 | 82 => (16, true),
        // BC4 formats: 8 bytes per 4x4 block
        83 | 84 | 85 => (8, true),
        // BC5 formats: 16 bytes per 4x4 block
        86 | 87 | 88 => (16, true),
        // BC6H formats: 16 bytes per 4x4 block
        94 | 95 | 96 => (16, true),
        // BC7 formats: 16 bytes per 4x4 block
        98 | 99 | 100 => (16, true),
        // Uncompressed formats: compute bytes per pixel
        _ => {
            // For non-block formats, we need to compute based on the format
            // This is a simplified approach - we compute the total size including all mips
            // by summing up each mip level
            return compute_uncompressed_size(dim, format, width, height, depth, mip_levels);
        }
    };

    if is_block_compressed {
        compute_block_compressed_size(width, height as u64, depth as u64, bytes_per_block, mip_levels)
    } else {
        // Shouldn't reach here for block formats, but fallback
        compute_uncompressed_size(dim, format, width, height, depth, mip_levels)
    }
}

/// Compute size for block-compressed formats.
/// Each mip level is rounded up to the nearest block.
fn compute_block_compressed_size(width: u64, height: u64, depth: u64, bytes_per_block: u64, mip_levels: u32) -> u64 {
    let mut total = 0u64;
    let mut w = width;
    let mut h = height;
    let mut d = depth;

    for _ in 0..mip_levels {
        if w == 0 || h == 0 || d == 0 {
            break;
        }
        // Number of blocks in each dimension (rounded up)
        let blocks_w = (w + 3) / 4;
        let blocks_h = (h + 3) / 4;
        let blocks_d = (d + 3) / 4;
        total += blocks_w * blocks_h * blocks_d * bytes_per_block;

        // Next mip level
        w = w.max(1) / 2;
        h = h.max(1) / 2;
        d = d.max(1) / 2;
    }
    total
}

/// Compute size for uncompressed formats by summing each mip level.
fn compute_uncompressed_size(dim: u32, format: u32, width: u64, height: u32, depth: u32, mip_levels: u32) -> u64 {
    let bpp = bytes_per_pixel(format);
    if bpp == 0 {
        // Unknown format, use a default of 4 bytes per pixel
        return compute_default_size(dim, width, height, depth, mip_levels);
    }

    let mut total = 0u64;
    let mut w = width;
    let mut h = height as u64;
    let mut d = depth as u64;

    for _ in 0..mip_levels {
        if w == 0 || h == 0 || d == 0 {
            break;
        }
        let level_size = w * h * d * bpp;
        total += level_size;

        w = w.max(1) / 2;
        h = h.max(1) / 2;
        d = d.max(1) / 2;
    }
    total
}

/// Default size computation when format is unknown.
fn compute_default_size(_dim: u32, width: u64, height: u32, depth: u32, mip_levels: u32) -> u64 {
    // Assume 4 bytes per pixel (RGBA8)
    let bpp = 4u64;
    let mut total = 0u64;
    let mut w = width;
    let mut h = height as u64;
    let mut d = depth as u64;

    for _ in 0..mip_levels {
        if w == 0 || h == 0 || d == 0 {
            break;
        }
        let level_size = w * h * d * bpp;
        total += level_size;

        w = w.max(1) / 2;
        h = h.max(1) / 2;
        d = d.max(1) / 2;
    }
    total
}

/// Get bytes per pixel for uncompressed DXGI formats.
fn bytes_per_pixel(format: u32) -> u64 {
    // Based on DXGI_FORMAT values
    // Common formats and their bytes per pixel:
    match format {
        // 8-bit formats
        61 => 1, // DXGI_FORMAT_R8_UNORM
        62 => 1, // DXGI_FORMAT_R8_SNORM
        63 => 1, // DXGI_FORMAT_R8_UINT
        64 => 1, // DXGI_FORMAT_R8_SINT
        
        // 16-bit formats
        55 => 2, // DXGI_FORMAT_R16_UNORM
        56 => 2, // DXGI_FORMAT_R16_SNORM
        57 => 2, // DXGI_FORMAT_R16_UINT
        58 => 2, // DXGI_FORMAT_R16_SINT
        59 => 2, // DXGI_FORMAT_R16_FLOAT
        65 => 4, // DXGI_FORMAT_R16G16_UNORM
        66 => 4, // DXGI_FORMAT_R16G16_SNORM
        67 => 4, // DXGI_FORMAT_R16G16_UINT
        68 => 4, // DXGI_FORMAT_R16G16_SINT
        69 => 4, // DXGI_FORMAT_R16G16_FLOAT
        
        // 24-bit formats (packed in 32 bits)
        70 => 4, // DXGI_FORMAT_R16G16B16A16_UNORM
        71 => 4, // DXGI_FORMAT_R16G16B16A16_SNORM
        72 => 4, // DXGI_FORMAT_R16G16B16A16_UINT
        73 => 4, // DXGI_FORMAT_R16G16B16A16_SINT
        74 => 4, // DXGI_FORMAT_R16G16B16A16_FLOAT
        
        // 32-bit formats
        41 => 4, // DXGI_FORMAT_R32_UINT
        42 => 4, // DXGI_FORMAT_R32_SINT
        43 => 4, // DXGI_FORMAT_R32_FLOAT
        75 => 8, // DXGI_FORMAT_R32G32_UINT
        76 => 8, // DXGI_FORMAT_R32G32_SINT
        77 => 8, // DXGI_FORMAT_R32G32_FLOAT
        78 => 16, // DXGI_FORMAT_R32G32B32_UINT
        79 => 16, // DXGI_FORMAT_R32G32B32_SINT
        80 => 16, // DXGI_FORMAT_R32G32B32_FLOAT
        81 => 16, // DXGI_FORMAT_R32G32B32A32_UINT
        82 => 16, // DXGI_FORMAT_R32G32B32A32_SINT
        83 => 16, // DXGI_FORMAT_R32G32B32A32_FLOAT
        
        // 8-bit per channel (32 bits total)
        28 => 4, // DXGI_FORMAT_R8G8B8A8_UNORM
        29 => 4, // DXGI_FORMAT_R8G8B8A8_UNORM_SRGB
        30 => 4, // DXGI_FORMAT_R8G8B8A8_UINT
        31 => 4, // DXGI_FORMAT_R8G8B8A8_SINT
        32 => 4, // DXGI_FORMAT_R8G8B8A8_SNORM
        
        // 10-bit formats (packed)
        34 => 4, // DXGI_FORMAT_R10G10B10A2_UNORM
        35 => 4, // DXGI_FORMAT_R10G10B10A2_UINT
        
        // 11/11/10 format
        36 => 4, // DXGI_FORMAT_R11G11B10_FLOAT
        
        // 16-bit per channel packed
        37 => 4, // DXGI_FORMAT_B8G8R8A8_UNORM
        38 => 4, // DXGI_FORMAT_B8G8R8A8_UNORM_SRGB
        
        // 16-bit BGR
        89 => 2, // DXGI_FORMAT_B5G6R5_UNORM
        90 => 2, // DXGI_FORMAT_B5G5R5A1_UNORM
        91 => 2, // DXGI_FORMAT_B4G4R4A4_UNORM
        
        // Default to 4 bytes per pixel for unknown formats
        _ => 4,
    }
}

/// Page deduplication census.
/// Tracks page hashes and counts duplicate bytes.
#[derive(Debug, Default)]
struct DedupCensus {
    /// Map from page hash to the number of times we've seen this exact page content
    page_hashes: std::collections::HashMap<u64, u64>,
    /// Total pages uploaded
    total_pages: u64,
    /// Total bytes uploaded
    total_bytes: u64,
    /// Total duplicate bytes (bytes from pages whose hash was already seen)
    duplicate_bytes: u64,
    /// Total duplicate pages
    duplicate_pages: u64,
}

/// Global state for the memory census.
static MEMORY_CENSUS: LazyLock<Mutex<MemoryCensusState>> = LazyLock::new(|| Mutex::new(MemoryCensusState::default()));

#[derive(Debug, Default)]
struct MemoryCensusState {
    /// Bytes by format
    format_bytes: BTreeMap<u32, FormatBytes>,
    /// Dedup census (only populated when XCL_PAGE_DEDUP_CENSUS=1)
    dedup: DedupCensus,
    /// Whether census bytes tracking is enabled
    bytes_enabled: bool,
    /// Whether dedup tracking is enabled
    dedup_enabled: bool,
}

/// Check if XCL_CENSUS_BYTES is enabled (cached).
fn bytes_enabled() -> bool {
    static ONCE: Once = Once::new();
    static ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    ONCE.call_once(|| {
        ENABLED.store(std::env::var_os("XCL_CENSUS_BYTES").is_some(), std::sync::atomic::Ordering::Relaxed);
    });
    ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Check if XCL_PAGE_DEDUP_CENSUS is enabled (cached).
fn dedup_enabled() -> bool {
    static ONCE: Once = Once::new();
    static ENABLED: std::sync::atomic::AtomicBool = std::sync::atomic::AtomicBool::new(false);
    ONCE.call_once(|| {
        ENABLED.store(std::env::var_os("XCL_PAGE_DEDUP_CENSUS").is_some(), std::sync::atomic::Ordering::Relaxed);
    });
    ENABLED.load(std::sync::atomic::Ordering::Relaxed)
}

/// Initialize the census state based on environment variables.
pub fn init() {
    let mut state = MEMORY_CENSUS.lock().unwrap();
    state.bytes_enabled = bytes_enabled();
    state.dedup_enabled = dedup_enabled();
}

/// Record a resource for the byte census.
/// Called from census_resource when XCL_CENSUS_BYTES=1.
pub fn record_resource(_path: &'static str, desc: usize, _display_heap: bool) {
    if !bytes_enabled() {
        return;
    }

    if desc == 0 || !crate::gpu::shader::readable(desc, 0x38) {
        return;
    }

    let x: [u8; 0x38] = unsafe { (desc as *const [u8; 0x38]).read_unaligned() };
    let u = |o: usize| u32::from_le_bytes(x[o..o + 4].try_into().unwrap());
    let u16 = |o: usize| u16::from_le_bytes(x[o..o + 2].try_into().unwrap());
    let dim = u(0);
    let w = u64::from_le_bytes(x[0x10..0x18].try_into().unwrap());
    let h = u(0x18);
    // DepthOrArraySize is at 0x1C (u16 in the PC struct, but the console desc is 0x38 bytes)
    // MipLevels is at 0x1E (u16)
    let depth_or_array = u16(0x1C) as u32;
    let mip_levels = u16(0x1E) as u32;
    let flags = u(0x30);
    let format = u(0x20);

    let size = compute_resource_size(dim, format, w, h, depth_or_array, mip_levels);

    let mut state = MEMORY_CENSUS.lock().unwrap();
    let entry = state.format_bytes.entry(format).or_default();
    entry.total += size;

    // For BC7 formats, track opaque bytes separately
    if is_bc7(format) && !has_alpha_flags(flags) {
        entry.bc7_opaque += size;
    }
}

/// Record texture page data for deduplication census.
/// Called from upload_batch when XCL_PAGE_DEDUP_CENSUS=1.
/// 
/// A "page" here is a 4x4 block for block-compressed formats, or a row-aligned region
/// for uncompressed formats. For simplicity and consistency, we hash at the granularity
/// that makes sense for the format.
pub fn record_page_data(format: u32, data: &[u8], row_bytes: usize, rows: usize, _depth: usize) {
    if !dedup_enabled() {
        return;
    }

    // For block-compressed formats, each 4x4 block is a natural page unit
    // For uncompressed formats, we use rows as the unit
    let mut state = MEMORY_CENSUS.lock().unwrap();

    // Determine page size based on format
    let (page_bytes, is_block) = match format {
        // BC formats: 4x4 blocks
        f if is_bc7(f) => (16, true), // BC7 is 16 bytes per 4x4 block
        71 | 72 | 73 | 74 => (8, true),  // BC1
        75 | 76 | 77 | 78 => (16, true), // BC2
        79 | 80 | 81 | 82 => (16, true), // BC3
        83 | 84 | 85 => (8, true),   // BC4
        86 | 87 | 88 => (16, true), // BC5
        94 | 95 | 96 => (16, true), // BC6H
        _ => (row_bytes, false), // For uncompressed, use full rows
    };

    // Hash each page
    if is_block {
        // For block-compressed: hash each 4x4 block
        let block_size = page_bytes;
        let num_blocks = data.len() / block_size;
        
        for i in 0..num_blocks {
            let start = i * block_size;
            let end = start + block_size;
            let page_data = &data[start..end];
            let hash = crate::gpu::shader::fnv64(page_data);
            
            state.dedup.total_pages += 1;
            state.dedup.total_bytes += block_size as u64;
            
            let is_duplicate = {
                let count = state.dedup.page_hashes.entry(hash).or_insert(0);
                let was_seen = *count > 0;
                *count += 1;
                was_seen
            };
            
            if is_duplicate {
                state.dedup.duplicate_pages += 1;
                state.dedup.duplicate_bytes += block_size as u64;
            }
        }
    } else {
        // For uncompressed: hash each row
        let row_size = row_bytes;
        let num_rows = rows;
        
        for i in 0..num_rows {
            let start = i * row_size;
            let end = start + row_size.min(data.len() - start);
            let page_data = &data[start..end];
            let hash = crate::gpu::shader::fnv64(page_data);
            
            state.dedup.total_pages += 1;
            state.dedup.total_bytes += page_data.len() as u64;
            
            let is_duplicate = {
                let count = state.dedup.page_hashes.entry(hash).or_insert(0);
                let was_seen = *count > 0;
                *count += 1;
                was_seen
            };
            
            if is_duplicate {
                state.dedup.duplicate_pages += 1;
                state.dedup.duplicate_bytes += page_data.len() as u64;
            }
        }
    }
}

/// Get the current census summary.
pub fn get_summary() -> (BTreeMap<u32, FormatBytes>, Option<DedupSummary>) {
    let state = MEMORY_CENSUS.lock().unwrap();
    let dedup_summary = if state.dedup_enabled {
        Some(DedupSummary {
            total_pages: state.dedup.total_pages,
            total_bytes: state.dedup.total_bytes,
            duplicate_pages: state.dedup.duplicate_pages,
            duplicate_bytes: state.dedup.duplicate_bytes,
        })
    } else {
        None
    };
    (state.format_bytes.clone(), dedup_summary)
}

/// Summary of page deduplication census.
#[derive(Debug, Default, Clone)]
pub struct DedupSummary {
    pub total_pages: u64,
    pub total_bytes: u64,
    pub duplicate_pages: u64,
    pub duplicate_bytes: u64,
}

/// Log the census summary.
pub fn log_summary() {
    let (format_bytes, dedup_summary) = get_summary();
    
    // Log format bytes
    if !format_bytes.is_empty() {
        let mut total_all = 0u64;
        let mut bc7_all = 0u64;
        let mut bc7_opaque = 0u64;
        
        for (format, bytes) in &format_bytes {
            total_all += bytes.total;
            if is_bc7(*format) {
                bc7_all += bytes.total;
                bc7_opaque += bytes.bc7_opaque;
            }
        }
        
        // Build format breakdown
        let mut formats: Vec<_> = format_bytes.iter().collect();
        formats.sort_by(|a, b| b.1.total.cmp(&a.1.total));
        
        let mut parts = Vec::new();
        for (format, bytes) in formats {
            if is_bc7(*format) {
                parts.push(format!("format_{}=({}, bc7_opaque={})", format, bytes.total, bytes.bc7_opaque));
            } else {
                parts.push(format!("format_{}={}", format, bytes.total));
            }
        }
        
        crate::xlog_info!("D3D12.X: memory census bytes: total={}, bc7_total={}, bc7_opaque={}, {}", 
            total_all, bc7_all, bc7_opaque, parts.join(", "));
    }
    
    // Log dedup summary
    if let Some(dedup) = dedup_summary {
        crate::xlog_info!(
            "D3D12.X: dedup census: {} pages, {} bytes uploaded, {} bytes duplicate",
            dedup.total_pages, dedup.total_bytes, dedup.duplicate_bytes
        );
    }
}

/// Get the metrics for report.json.
pub fn get_metrics() -> MemoryCensusMetrics {
    let (format_bytes, dedup_summary) = get_summary();
    
    let mut total_all = 0u64;
    let mut bc7_all = 0u64;
    let mut bc7_opaque = 0u64;
    
    let mut format_totals: BTreeMap<String, u64> = BTreeMap::new();
    let mut format_bc7_opaque: BTreeMap<String, u64> = BTreeMap::new();
    
    for (format, bytes) in &format_bytes {
        total_all += bytes.total;
        let format_str = format.to_string();
        format_totals.insert(format_str.clone(), bytes.total);
        
        if is_bc7(*format) {
            bc7_all += bytes.total;
            bc7_opaque += bytes.bc7_opaque;
            format_bc7_opaque.insert(format_str, bytes.bc7_opaque);
        }
    }
    
    MemoryCensusMetrics {
        total_bytes: total_all,
        bc7_total_bytes: bc7_all,
        bc7_opaque_bytes: bc7_opaque,
        format_bytes: format_totals,
        format_bc7_opaque_bytes: format_bc7_opaque,
        dedup_pages: dedup_summary.as_ref().map(|d| d.total_pages).unwrap_or(0),
        dedup_bytes_uploaded: dedup_summary.as_ref().map(|d| d.total_bytes).unwrap_or(0),
        dedup_bytes_duplicate: dedup_summary.as_ref().map(|d| d.duplicate_bytes).unwrap_or(0),
    }
}

/// Metrics for report.json under d3d12.metrics.memory_census.
#[derive(Debug, Default, Clone, serde::Serialize, serde::Deserialize)]
pub struct MemoryCensusMetrics {
    /// Total bytes across all resources
    pub total_bytes: u64,
    /// Total bytes in BC7 resources
    pub bc7_total_bytes: u64,
    /// Bytes in BC7 resources without alpha-bearing flags
    pub bc7_opaque_bytes: u64,
    /// Bytes by format (format value -> bytes)
    pub format_bytes: BTreeMap<String, u64>,
    /// BC7 opaque bytes by format
    pub format_bc7_opaque_bytes: BTreeMap<String, u64>,
    /// Dedup: total pages uploaded
    pub dedup_pages: u64,
    /// Dedup: total bytes uploaded
    pub dedup_bytes_uploaded: u64,
    /// Dedup: duplicate bytes
    pub dedup_bytes_duplicate: u64,
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn block_compressed_size_bc7_1024x1024() {
        // BC7: 16 bytes per 4x4 block
        // 1024x1024 = 256x256 blocks = 65536 blocks
        // 65536 * 16 = 1048576 bytes per mip
        // With full mip chain: 1024 -> 512 -> 256 -> 128 -> 64 -> 32 -> 16 -> 8 -> 4 -> 2 -> 1
        // But we typically don't have all mips in practice
        
        // Single mip (mip_levels=1)
        let size = compute_block_compressed_size(1024, 1024, 1, 16, 1);
        assert_eq!(size, 1048576, "BC7 1024x1024 single mip should be 1048576 bytes");
        
        // Two mips
        let size = compute_block_compressed_size(1024, 1024, 1, 16, 2);
        // 1024x1024: 65536 blocks * 16 = 1048576
        // 512x512: 16384 blocks * 16 = 262144
        // Total: 1310720
        assert_eq!(size, 1310720, "BC7 1024x1024 with 2 mips should be 1310720 bytes");
    }

    #[test]
    fn block_compressed_size_bc1_256x256() {
        // BC1: 8 bytes per 4x4 block
        // 256x256 = 64x64 blocks = 4096 blocks
        // 4096 * 8 = 32768 bytes per mip
        let size = compute_block_compressed_size(256, 256, 1, 8, 1);
        assert_eq!(size, 32768, "BC1 256x256 single mip should be 32768 bytes");
    }

    #[test]
    fn block_compressed_size_bc7_odd_dimensions() {
        // 100x50 texture with BC7 (16 bytes per 4x4 block)
        // blocks_w = ceil(100/4) = 25
        // blocks_h = ceil(50/4) = 13
        // total blocks = 25 * 13 = 325
        // size = 325 * 16 = 5200
        let size = compute_block_compressed_size(100, 50, 1, 16, 1);
        assert_eq!(size, 5200, "BC7 100x50 should be 5200 bytes");
    }

    #[test]
    fn block_compressed_size_3d() {
        // 3D texture 64x64x16 with BC7
        // blocks_w = 16, blocks_h = 16, blocks_d = 4
        // total blocks = 16 * 16 * 4 = 1024
        // size = 1024 * 16 = 16384
        let size = compute_block_compressed_size(64, 64, 16, 16, 1);
        assert_eq!(size, 16384, "BC7 64x64x16 3D should be 16384 bytes");
    }

    #[test]
    fn uncompressed_size_rgba8_1024x1024() {
        // RGBA8: 4 bytes per pixel
        // 1024x1024 = 1048576 pixels
        // 1048576 * 4 = 4194304 bytes
        // Note: our format mapping uses internal numbers, not DXGI_FORMAT enum values
        // For this test, we need to use a format that maps to 4 bpp
        // DXGI_FORMAT_R8G8B8A8_UNORM = 28
        let size = compute_uncompressed_size(
            DXGI_FORMAT::D3D12_RESOURCE_DIMENSION_TEXTURE2D.0 as u32,
            28, // R8G8B8A8_UNORM
            1024, 1024, 1, 1
        );
        assert_eq!(size, 4194304, "RGBA8 1024x1024 should be 4194304 bytes");
    }

    #[test]
    fn buffer_size() {
        // Buffer: size is the width field
        let size = compute_resource_size(
            DXGI_FORMAT::D3D12_RESOURCE_DIMENSION_BUFFER.0 as u32,
            0, // format doesn't matter for buffers
            1024 * 1024, // 1MB buffer
            0, 0, 1
        );
        assert_eq!(size, 1024 * 1024, "Buffer of 1MB should report 1MB");
    }

    #[test]
    fn bc7_format_detection() {
        assert!(is_bc7(98), "98 should be BC7_UNORM");
        assert!(is_bc7(99), "99 should be BC7_UNORM_SRGB");
        assert!(!is_bc7(28), "28 should not be BC7 (it's R8G8B8A8_UNORM)");
    }

    #[test]
    fn alpha_bearing_flags_detection() {
        // No flags = no alpha bearing
        assert!(!has_alpha_flags(0));
        
        // ALLOW_RENDER_TARGET flag
        assert!(has_alpha_flags(D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 as u32));
        
        // ALLOW_UNORDERED_ACCESS flag
        assert!(has_alpha_flags(D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32));
        
        // ALLOW_SIMULTANEOUS_ACCESS flag
        assert!(has_alpha_flags(D3D12_RESOURCE_FLAG_ALLOW_SIMULTANEOUS_ACCESS.0 as u32));
        
        // Multiple flags
        assert!(has_alpha_flags(
            D3D12_RESOURCE_FLAG_ALLOW_RENDER_TARGET.0 as u32 
            | D3D12_RESOURCE_FLAG_ALLOW_UNORDERED_ACCESS.0 as u32
        ));
    }

    #[test]
    fn dedup_accounting_single_page() {
        // Simulate hashing the same page twice
        let mut dedup = DedupCensus::default();
        
        let data = vec![1u8, 2, 3, 4, 5, 6, 7, 8];
        let hash = crate::gpu::shader::fnv64(&data);
        
        // First occurrence: not a duplicate
        let count = dedup.page_hashes.entry(hash).or_insert(0);
        *count += 1;
        dedup.total_pages += 1;
        dedup.total_bytes += data.len() as u64;
        
        // Second occurrence: is a duplicate
        let count = dedup.page_hashes.entry(hash).or_insert(0);
        if *count > 0 {
            dedup.duplicate_pages += 1;
            dedup.duplicate_bytes += data.len() as u64;
        }
        *count += 1;
        dedup.total_pages += 1;
        dedup.total_bytes += data.len() as u64;
        
        assert_eq!(dedup.total_pages, 2);
        assert_eq!(dedup.total_bytes, 16);
        assert_eq!(dedup.duplicate_pages, 1);
        assert_eq!(dedup.duplicate_bytes, 8);
    }

    #[test]
    fn dedup_accounting_different_pages() {
        let mut dedup = DedupCensus::default();
        
        let data1 = vec![1u8, 2, 3, 4];
        let data2 = vec![5u8, 6, 7, 8];
        
        let hash1 = crate::gpu::shader::fnv64(&data1);
        let hash2 = crate::gpu::shader::fnv64(&data2);
        
        // First page
        let count = dedup.page_hashes.entry(hash1).or_insert(0);
        *count += 1;
        dedup.total_pages += 1;
        dedup.total_bytes += data1.len() as u64;
        
        // Second page (different)
        let count = dedup.page_hashes.entry(hash2).or_insert(0);
        *count += 1;
        dedup.total_pages += 1;
        dedup.total_bytes += data2.len() as u64;
        
        assert_eq!(dedup.total_pages, 2);
        assert_eq!(dedup.total_bytes, 8);
        assert_eq!(dedup.duplicate_pages, 0);
        assert_eq!(dedup.duplicate_bytes, 0);
    }

    #[test]
    fn mip_chain_size_bc7() {
        // BC7 256x256 with full mip chain (9 levels: 256, 128, 64, 32, 16, 8, 4, 2, 1)
        let size = compute_block_compressed_size(256, 256, 1, 16, 9);
        
        // 256x256: 64x64 = 4096 blocks * 16 = 65536
        // 128x128: 32x32 = 1024 blocks * 16 = 16384
        // 64x64: 16x16 = 256 blocks * 16 = 4096
        // 32x32: 8x8 = 64 blocks * 16 = 1024
        // 16x16: 4x4 = 16 blocks * 16 = 256
        // 8x8: 2x2 = 4 blocks * 16 = 64
        // 4x4: 1x1 = 1 block * 16 = 16
        // 2x2: 1x1 = 1 block * 16 = 16 (ceil(2/4)=1)
        // 1x1: 1x1 = 1 block * 16 = 16 (ceil(1/4)=1)
        // Total: 65536 + 16384 + 4096 + 1024 + 256 + 64 + 16 + 16 + 16 = 87392
        assert_eq!(size, 87392, "BC7 256x256 with 9 mips should be 87392 bytes");
    }
}
