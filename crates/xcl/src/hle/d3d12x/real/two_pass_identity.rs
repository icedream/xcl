//! Stable driver blob identities for translated two-pass pipelines.
use super::{payload_shader, stream_subobjects};
use windows::Win32::Graphics::Direct3D12::*;

/// Bump on any change to two-pass cache key fields or ABI root-signature layout to invalidate stale disk entries.
const TWO_PASS_CACHE_SCHEMA: u32 = 3;
/// ABI root signature layout tag for two-pass pipelines (`graphics = true`: 6 root UAVs + directly indexed heaps + IA flag).
const TWO_PASS_ABI_RS_TAG: u32 = 0x5253_0001;

pub(super) struct Fnv64(u64);

impl Fnv64 {
    pub(super) fn new() -> Self {
        Self(0xcbf2_9ce4_8422_2325)
    }
    fn bytes(&mut self, b: &[u8]) {
        for &byte in b {
            self.0 = (self.0 ^ byte as u64).wrapping_mul(0x0000_0100_0000_01b3);
        }
    }
    pub(super) fn u32(&mut self, v: u32) {
        self.bytes(&v.to_le_bytes());
    }
    fn i32(&mut self, v: i32) {
        self.bytes(&v.to_le_bytes());
    }
    pub(super) fn u64(&mut self, v: u64) {
        self.bytes(&v.to_le_bytes());
    }
    fn bool(&mut self, v: bool) {
        self.bytes(&[v as u8]);
    }
    pub(super) fn finish(self) -> u64 {
        self.0
    }
}

fn samplers_hash_into(h: &mut Fnv64, samplers: &[crate::gpu::recomp::pso::StaticSampler]) {
    h.u64(samplers.len() as u64);
    for (w, cmp) in samplers {
        for word in w {
            h.u32(*word);
        }
        h.bool(*cmp);
    }
}

unsafe fn read_state<T: Copy>(payload: &[u8]) -> Option<T> {
    let payload = payload.get(..std::mem::size_of::<T>())?;
    Some(std::ptr::read_unaligned(payload.as_ptr().cast()))
}

/// `n` bytes at `p`, if that memory is readable (`n == 0` needs no memory).
unsafe fn slice_at<'a>(p: usize, n: usize) -> Option<&'a [u8]> {
    if n == 0 {
        return Some(&[]);
    }
    crate::gpu::shader::readable(p, n).then(|| unsafe { std::slice::from_raw_parts(p as *const u8, n) })
}

/// A NUL-terminated semantic name behind a pointer: hashed by its bytes, never by its address.
unsafe fn hash_name(h: &mut Fnv64, p: usize) -> Option<()> {
    if p == 0 {
        h.u64(0);
        return Some(());
    }
    let mut n = 0usize;
    while n < 256 && crate::gpu::shader::readable(p + n, 1) && unsafe { *(p as *const u8).add(n) } != 0 {
        n += 1;
    }
    if n >= 256 || !crate::gpu::shader::readable(p + n, 1) {
        return None;
    }
    h.u64(n as u64);
    h.bytes(unsafe { std::slice::from_raw_parts(p as *const u8, n) });
    Some(())
}

/// STREAM_OUTPUT (type 7): the declaration entries and buffer strides behind their pointers are
/// hashed by content (semantic-name addresses normalised); the counts and the rasterized stream
/// as-is. Returns None when a pointed-to array is not readable.
unsafe fn hash_stream_output(h: &mut Fnv64, p: &[u8]) -> Option<()> {
    let d = read_state::<D3D12_STREAM_OUTPUT_DESC>(p)?;
    // an empty descriptor carries only inert pointers: hash nothing, so a stream with one
    // matches the same stream without it
    if d.NumEntries == 0 && d.NumStrides == 0 && d.RasterizedStream == 0 {
        return Some(());
    }
    h.u32(7);
    h.u32(d.NumEntries);
    h.u32(d.NumStrides);
    h.u32(d.RasterizedStream);
    if d.NumEntries > D3D12_SO_OUTPUT_COMPONENT_COUNT || d.NumStrides > D3D12_SO_STREAM_COUNT {
        return None;
    }
    let esz = std::mem::size_of::<D3D12_SO_DECLARATION_ENTRY>();
    let decl = slice_at(d.pSODeclaration as usize, d.NumEntries as usize * esz)?;
    for i in 0..d.NumEntries as usize {
        let e = read_state::<D3D12_SO_DECLARATION_ENTRY>(&decl[i * esz..])?;
        h.u32(e.Stream);
        hash_name(h, e.SemanticName.0 as usize)?;
        h.u32(e.SemanticIndex);
        h.bytes(&[e.StartComponent, e.ComponentCount, e.OutputSlot]);
    }
    let strides = slice_at(d.pBufferStrides as usize, d.NumStrides as usize * 4)?;
    for w in strides.chunks_exact(4) {
        h.u32(u32::from_le_bytes(w.try_into().ok()?));
    }
    Some(())
}

/// INPUT_LAYOUT (type 12): the element descriptions behind the pointer are hashed by content
/// (semantic-name addresses normalised). Returns None when the array is not readable.
unsafe fn hash_input_layout(h: &mut Fnv64, p: &[u8]) -> Option<()> {
    let d = read_state::<D3D12_INPUT_LAYOUT_DESC>(p)?;
    if d.NumElements == 0 {
        return Some(());
    }
    h.u32(12);
    h.u32(d.NumElements);
    if d.NumElements > 128 {
        return None;
    }
    let esz = std::mem::size_of::<D3D12_INPUT_ELEMENT_DESC>();
    let els = slice_at(d.pInputElementDescs as usize, d.NumElements as usize * esz)?;
    for i in 0..d.NumElements as usize {
        let e = read_state::<D3D12_INPUT_ELEMENT_DESC>(&els[i * esz..])?;
        hash_name(h, e.SemanticName.0 as usize)?;
        h.u32(e.SemanticIndex);
        h.i32(e.Format.0);
        h.u32(e.InputSlot);
        h.u32(e.AlignedByteOffset);
        h.i32(e.InputSlotClass.0);
        h.u32(e.InstanceDataStepRate);
    }
    Some(())
}

/// VIEW_INSTANCING (type 22): the view instance locations behind the pointer are hashed raw (they
/// hold no pointers); the count and flags as-is. Returns None when the array is not readable.
unsafe fn hash_view_instancing(h: &mut Fnv64, p: &[u8]) -> Option<()> {
    let d = read_state::<D3D12_VIEW_INSTANCING_DESC>(p)?;
    if d.ViewInstanceCount == 0 && d.Flags.0 == 0 {
        return Some(());
    }
    h.u32(22);
    h.u32(d.ViewInstanceCount);
    h.i32(d.Flags.0);
    if d.ViewInstanceCount > 1024 {
        return None;
    }
    let locs = slice_at(d.pViewInstanceLocations as usize, d.ViewInstanceCount as usize * std::mem::size_of::<D3D12_VIEW_INSTANCE_LOCATION>())?;
    h.bytes(locs);
    Some(())
}

unsafe fn hash_blend(h: &mut Fnv64, p: &[u8]) -> Option<()> {
    let b = read_state::<D3D12_BLEND_DESC>(p)?;
    h.u32(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND.0 as u32);
    let indep = b.IndependentBlendEnable.0 != 0;
    h.bool(b.AlphaToCoverageEnable.0 != 0);
    h.bool(indep);
    let count = if indep { 8 } else { 1 };
    for rt in &b.RenderTarget[..count] {
        h.bool(rt.BlendEnable.0 != 0);
        h.bool(rt.LogicOpEnable.0 != 0);
        h.i32(rt.SrcBlend.0);
        h.i32(rt.DestBlend.0);
        h.i32(rt.BlendOp.0);
        h.i32(rt.SrcBlendAlpha.0);
        h.i32(rt.DestBlendAlpha.0);
        h.i32(rt.BlendOpAlpha.0);
        h.i32(rt.LogicOp.0);
        h.bytes(&[rt.RenderTargetWriteMask]);
    }
    Some(())
}

unsafe fn hash_rast(h: &mut Fnv64, ty: u32, p: &[u8]) -> Option<()> {
    h.u32(ty);
    match ty {
        10 => {
            let r = read_state::<D3D12_RASTERIZER_DESC>(p)?;
            h.i32(r.FillMode.0);
            h.i32(r.CullMode.0);
            h.bool(r.FrontCounterClockwise.0 != 0);
            h.i32(r.DepthBias);
            h.u32(r.DepthBiasClamp.to_bits());
            h.u32(r.SlopeScaledDepthBias.to_bits());
            h.bool(r.DepthClipEnable.0 != 0);
            h.bool(r.MultisampleEnable.0 != 0);
            h.bool(r.AntialiasedLineEnable.0 != 0);
            h.u32(r.ForcedSampleCount);
            h.i32(r.ConservativeRaster.0);
        }
        27 => {
            let r = read_state::<D3D12_RASTERIZER_DESC1>(p)?;
            h.i32(r.FillMode.0);
            h.i32(r.CullMode.0);
            h.bool(r.FrontCounterClockwise.0 != 0);
            h.u32(r.DepthBias.to_bits());
            h.u32(r.DepthBiasClamp.to_bits());
            h.u32(r.SlopeScaledDepthBias.to_bits());
            h.bool(r.DepthClipEnable.0 != 0);
            h.bool(r.MultisampleEnable.0 != 0);
            h.bool(r.AntialiasedLineEnable.0 != 0);
            h.u32(r.ForcedSampleCount);
            h.i32(r.ConservativeRaster.0);
        }
        28 => {
            let r = read_state::<D3D12_RASTERIZER_DESC2>(p)?;
            h.i32(r.FillMode.0);
            h.i32(r.CullMode.0);
            h.bool(r.FrontCounterClockwise.0 != 0);
            h.u32(r.DepthBias.to_bits());
            h.u32(r.DepthBiasClamp.to_bits());
            h.u32(r.SlopeScaledDepthBias.to_bits());
            h.bool(r.DepthClipEnable.0 != 0);
            h.i32(r.LineRasterizationMode.0);
            h.u32(r.ForcedSampleCount);
            h.i32(r.ConservativeRaster.0);
        }
        _ => return None,
    }
    Some(())
}

unsafe fn hash_depth_stencil(h: &mut Fnv64, ty: u32, p: &[u8]) -> Option<()> {
    let op = |h: &mut Fnv64, o: D3D12_DEPTH_STENCILOP_DESC| {
        h.i32(o.StencilFailOp.0);
        h.i32(o.StencilDepthFailOp.0);
        h.i32(o.StencilPassOp.0);
        h.i32(o.StencilFunc.0);
    };
    let op1 = |h: &mut Fnv64, o: D3D12_DEPTH_STENCILOP_DESC1| {
        h.i32(o.StencilFailOp.0);
        h.i32(o.StencilDepthFailOp.0);
        h.i32(o.StencilPassOp.0);
        h.i32(o.StencilFunc.0);
        h.bytes(&[o.StencilReadMask, o.StencilWriteMask]);
    };
    h.u32(ty);
    match ty {
        11 => {
            let d = read_state::<D3D12_DEPTH_STENCIL_DESC>(p)?;
            h.bool(d.DepthEnable.0 != 0);
            h.i32(d.DepthWriteMask.0);
            h.i32(d.DepthFunc.0);
            h.bool(d.StencilEnable.0 != 0);
            h.bytes(&[d.StencilReadMask, d.StencilWriteMask]);
            op(h, d.FrontFace);
            op(h, d.BackFace);
        }
        21 => {
            let d = read_state::<D3D12_DEPTH_STENCIL_DESC1>(p)?;
            h.bool(d.DepthEnable.0 != 0);
            h.i32(d.DepthWriteMask.0);
            h.i32(d.DepthFunc.0);
            h.bool(d.StencilEnable.0 != 0);
            h.bytes(&[d.StencilReadMask, d.StencilWriteMask]);
            op(h, d.FrontFace);
            op(h, d.BackFace);
            h.bool(d.DepthBoundsTestEnable.0 != 0);
        }
        26 => {
            let d = read_state::<D3D12_DEPTH_STENCIL_DESC2>(p)?;
            h.bool(d.DepthEnable.0 != 0);
            h.i32(d.DepthWriteMask.0);
            h.i32(d.DepthFunc.0);
            h.bool(d.StencilEnable.0 != 0);
            op1(h, d.FrontFace);
            op1(h, d.BackFace);
            h.bool(d.DepthBoundsTestEnable.0 != 0);
        }
        _ => return None,
    }
    Some(())
}

/// Canonical cache key `(hash, total_shader_bytes)` for a two-pass mesh/PS pipeline stream,
/// hashing referenced shader bytecodes and normalized fixed-function state (never pointers or
/// struct padding). Shader pointers in the stream must refer to live shader bytecode; the
/// STREAM_OUTPUT, INPUT_LAYOUT and VIEW_INSTANCING arrays behind their pointers must be readable
/// and are hashed by content with the semantic-name addresses normalised. Returns `None` if an
/// unsupported, duplicate, unknown or unreadable subobject is active.
pub(super) unsafe fn two_pass_stream_key(stream: &[u8], ps_samplers: &[crate::gpu::recomp::pso::StaticSampler], driver_id: u64, mem_windows: u32) -> Option<(u64, usize)> {
    let mut subs = stream_subobjects(stream)?;
    let total = subs.iter().map(|(ty, payload)| {
        let (_, align) = super::super::null::subobject_payload(*ty).unwrap();
        (align + payload.len() + 7) & !7
    }).sum::<usize>();
    if total != stream.len() {
        return None;
    }
    subs.sort_by_key(|s| s.0);
    if subs.windows(2).any(|w| w[0].0 == w[1].0) {
        return None;
    }
    let mut h = Fnv64::new();
    h.u32(TWO_PASS_CACHE_SCHEMA);
    h.u32(1); // stream kind
    h.u32(TWO_PASS_ABI_RS_TAG);
    h.u32(mem_windows);
    h.u64(driver_id);
    samplers_hash_into(&mut h, ps_samplers);
    let mut shader_bytes = 0usize;
    for (ty, payload) in subs {
        match ty {
            0 | 19 => {} // RS replaced by ABI RS via ps_samplers; CACHED_PSO is the lookup input
            // These descriptors carry live pointers and padding in a title stream, so hash
            // their content field-wise and drop the address-bearing bytes.
            7 => hash_stream_output(&mut h, payload)?,
            12 => hash_input_layout(&mut h, payload)?,
            22 => hash_view_instancing(&mut h, payload)?,
            1..=6 | 24 | 25 => {
                let state = read_state::<D3D12_SHADER_BYTECODE>(payload)?;
                if state.BytecodeLength == 0 && ty != 25 {
                    continue;
                }
                let code = payload_shader(payload)?;
                shader_bytes = shader_bytes.checked_add(code.len())?;
                h.u32(ty);
                h.u64(code.len() as u64);
                h.bytes(code);
            }
            8 => hash_blend(&mut h, payload)?,
            9 | 13 | 14 | 16 | 17 | 18 | 20 => {
                h.u32(ty);
                h.bytes(payload);
            }
            10 | 27 | 28 => hash_rast(&mut h, ty, payload)?,
            11 | 21 | 26 => hash_depth_stencil(&mut h, ty, payload)?,
            15 => {
                let rt = read_state::<D3D12_RT_FORMAT_ARRAY>(payload)?;
                if rt.NumRenderTargets > 8 {
                    return None;
                }
                let n = rt.NumRenderTargets as usize;
                h.u32(ty);
                h.u32(rt.NumRenderTargets);
                for f in &rt.RTFormats[..n] {
                    h.i32(f.0);
                }
            }
            _ => return None,
        }
    }
    Some((h.finish(), shader_bytes))
}

pub(super) fn two_pass_cs_key(cs: &[u8], samplers: &[crate::gpu::recomp::pso::StaticSampler], driver_id: u64, mem_windows: u32) -> u64 {
    let mut h = Fnv64::new();
    h.u32(TWO_PASS_CACHE_SCHEMA);
    h.u32(2); // compute-pass kind
    h.u32(TWO_PASS_ABI_RS_TAG);
    h.u32(mem_windows);
    h.u64(driver_id);
    h.u64(cs.len() as u64);
    h.bytes(cs);
    samplers_hash_into(&mut h, samplers);
    h.finish()
}

#[cfg(test)]
mod tests {
    use super::*;
    use super::super::StreamOut;
    use std::ffi::c_void;
    use windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT;
    use windows::core::PCSTR;

    #[test]
    fn canonical_stream_key_distinguishes_state_samplers_and_driver() {
        let ms = [1u8, 2, 3, 4];
        let ps = [5u8, 6, 7, 8, 9];
        let rs_a: *mut c_void = 0x1000 as *mut c_void;
        let rs_b: *mut c_void = 0x2000 as *mut c_void;
        let mut s1 = StreamOut::default();
        s1.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs_a);
        s1.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
        s1.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &ps);
        s1.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &u32::MAX);
        s1.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &45i32);

        let mut s2 = StreamOut::default();
        // Same shaders/state, different RS pointer and extra CACHED_PSO: must keep the same key.
        s2.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &45i32);
        s2.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, &ps);
        s2.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
        s2.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &rs_b);
        s2.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &u32::MAX);
        let blob = [0xAAu8; 8];
        s2.put(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO,
            &D3D12_CACHED_PIPELINE_STATE { pCachedBlob: blob.as_ptr() as _, CachedBlobSizeInBytes: blob.len() },
        );

        unsafe {
            let k1 = two_pass_stream_key(&s1.0, &[], 0x1234, 1).unwrap();
            let k2 = two_pass_stream_key(&s2.0, &[], 0x1234, 1).unwrap();
            assert_eq!(k1, k2);
            assert_eq!(k1.1, ms.len() + ps.len());
            assert_ne!(k1, two_pass_stream_key(&s1.0, &[([1, 2, 3, 4], false)], 0x1234, 1).unwrap());
            assert_ne!(k1, two_pass_stream_key(&s1.0, &[], 0x5678, 1).unwrap());
        }
    }

    #[test]
    fn canonical_stream_key_normalizes_padding_and_rejects_unreadable_pointers() {
        let mut s1 = StreamOut::default();
        let mut rt1 = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
        rt1.RTFormats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(28);
        rt1.RTFormats[7] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(999);
        let mut blend1 = D3D12_BLEND_DESC::default();
        blend1.IndependentBlendEnable = false.into();
        blend1.RenderTarget[3].RenderTargetWriteMask = 7; // unused when IndependentBlendEnable is false
        s1.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt1);
        s1.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend1);

        let mut s2 = StreamOut::default();
        let mut rt2 = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 1, ..Default::default() };
        rt2.RTFormats[0] = windows::Win32::Graphics::Dxgi::Common::DXGI_FORMAT(28);
        let mut blend2 = D3D12_BLEND_DESC::default();
        blend2.IndependentBlendEnable = false.into();
        s2.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt2);
        s2.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend2);

        let mut ds_a = [0; std::mem::size_of::<D3D12_DEPTH_STENCIL_DESC1>()];
        let mut ds_b = ds_a;
        let after_masks = std::mem::offset_of!(D3D12_DEPTH_STENCIL_DESC1, StencilWriteMask) + 1;
        let first_face = std::mem::offset_of!(D3D12_DEPTH_STENCIL_DESC1, FrontFace);
        assert!(after_masks < first_face);
        ds_a[after_masks..first_face].fill(0xaa);
        ds_b[after_masks..first_face].fill(0x55);
        let depth = D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL1.0 as u32;
        s1.raw(depth, &ds_a, false);
        s2.raw(depth, &ds_b, false);

        unsafe {
            assert_eq!(two_pass_stream_key(&s1.0, &[], 1, 1), two_pass_stream_key(&s2.0, &[], 1, 1));
            // An unreadable pointer in a STREAM_OUTPUT/INPUT_LAYOUT/VIEW_INSTANCING subobject fails closed.
            let mut with_il = StreamOut::default();
            let il = D3D12_INPUT_LAYOUT_DESC { pInputElementDescs: 0x1234 as *const _, NumElements: 1 };
            with_il.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, &il);
            assert_eq!(two_pass_stream_key(&with_il.0, &[], 1, 1), None);
        }
    }

    #[test]
    fn empty_stream_descriptors_do_not_change_the_key() {
        let ms = [1u8, 2, 3, 4];
        let mut base = StreamOut::default();
        base.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
        let mut with_empty = StreamOut(base.0.clone(), base.1);
        with_empty.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT, &D3D12_STREAM_OUTPUT_DESC {
            pSODeclaration: 0x2000 as *const _,
            pBufferStrides: 0x3000 as *const _,
            ..Default::default()
        });
        let mut input = [0u8; std::mem::size_of::<D3D12_INPUT_LAYOUT_DESC>()];
        input[..8].copy_from_slice(&0x4000u64.to_le_bytes());
        let padding = std::mem::offset_of!(D3D12_INPUT_LAYOUT_DESC, NumElements) + std::mem::size_of::<u32>();
        input[padding..].fill(0xaa);
        with_empty.raw(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT.0 as u32, &input, true);
        with_empty.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING, &D3D12_VIEW_INSTANCING_DESC {
            pViewInstanceLocations: 0x5000 as *const _,
            ..Default::default()
        });
        let empty_shader = D3D12_SHADER_BYTECODE { pShaderBytecode: 0x6000 as *const _, BytecodeLength: 0 };
        for stage in [
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VS,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DS,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_HS,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_GS,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CS,
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_AS,
        ] {
            with_empty.put(stage, &empty_shader);
        }

        unsafe {
            let key = two_pass_stream_key(&base.0, &[], 1, 4).expect("base mesh key");
            assert_eq!(two_pass_stream_key(&with_empty.0, &[], 1, 4), Some(key));

            let mut empty_ms = StreamOut::default();
            empty_ms.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &empty_shader);
            assert_eq!(two_pass_stream_key(&empty_ms.0, &[], 1, 4), None);
        }
    }

    #[test]
    fn two_pass_keys_cover_mem_windows_invalid_counts_and_duplicates() {
        let mut s = StreamOut::default();
        let ms = [1u8, 2, 3, 4];
        s.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &ms);
        s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0);
        unsafe {
            assert_ne!(two_pass_stream_key(&s.0, &[], 1, 1), two_pass_stream_key(&s.0, &[], 1, 4 | crate::gpu::recomp::HEAP_WINDOWS));
            // Duplicate subobjects fail closed.
            let mut dup = StreamOut(s.0.clone(), s.1);
            dup.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0);
            assert_eq!(two_pass_stream_key(&dup.0, &[], 1, 1), None);
            // Invalid NumRenderTargets > 8 fails closed.
            let mut bad_rt = StreamOut::default();
            let rt = D3D12_RT_FORMAT_ARRAY { NumRenderTargets: 9, ..Default::default() };
            bad_rt.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt);
            assert_eq!(two_pass_stream_key(&bad_rt.0, &[], 1, 1), None);
        }
        let cs = [10u8; 64];
        assert_ne!(two_pass_cs_key(&cs, &[], 1, 1), two_pass_cs_key(&cs, &[], 1, 4));
    }

    #[test]
    fn two_pass_cs_key_changes_with_samplers_and_driver() {
        let cs = [10u8; 64];
        let k0 = two_pass_cs_key(&cs, &[], 1, 1);
        assert_eq!(k0, two_pass_cs_key(&cs, &[], 1, 1));
        assert_ne!(k0, two_pass_cs_key(&cs, &[([0, 1, 2, 3], false)], 1, 1));
        assert_ne!(k0, two_pass_cs_key(&cs, &[([0, 1, 2, 3], true)], 1, 1));
        assert_ne!(k0, two_pass_cs_key(&cs, &[], 2, 1));
        assert_ne!(k0, two_pass_cs_key(&[11u8; 64], &[], 1, 1));
    }

    // One game-like MS pipeline stream: every subobject type that can carry an address, plus
    // state. Two creations of the same logical pipeline differ only in the pointer fields.
    #[derive(Clone)]
    struct StreamParams {
        rs: usize,
        ms: Vec<u8>,
        ps: Option<Vec<u8>>,
        name: *const u8,
        so_stream: u32,
        stride: u32,
        il_format: i32,
        vi_count: u32,
        vi_rtai: u32,
        sample_mask: u32,
        ds_format: i32,
        rt_count: u32,
        rt0: i32,
        blend_src: i32,
        cull: i32,
        depth_func: i32,
        topology: i32,
        blob: Vec<u8>,
    }

    impl Default for StreamParams {
        fn default() -> Self {
            Self {
                rs: 0x1000,
                ms: vec![1, 2, 3, 4, 5],
                ps: Some(vec![9, 8, 7]),
                name: b"POSITION\0".as_ptr(),
                so_stream: 0,
                stride: 16,
                il_format: 28,
                vi_count: 2,
                vi_rtai: 0,
                sample_mask: u32::MAX,
                ds_format: 45,
                rt_count: 2,
                rt0: 28,
                blend_src: D3D12_BLEND_SRC_ALPHA.0,
                cull: D3D12_CULL_MODE_BACK.0,
                depth_func: D3D12_COMPARISON_FUNC_LESS.0,
                topology: D3D12_PRIMITIVE_TOPOLOGY_TYPE_TRIANGLE.0,
                blob: vec![0xAA; 8],
            }
        }
    }

    // The stream plus the arrays its pointers refer to, kept alive for the key computation.
    struct Built {
        stream: StreamOut,
        // heap buffers: their addresses stay valid while Built is moved around
        _so_decl: Vec<D3D12_SO_DECLARATION_ENTRY>,
        _strides: Vec<u32>,
        _il: Vec<D3D12_INPUT_ELEMENT_DESC>,
        _vi: Vec<D3D12_VIEW_INSTANCE_LOCATION>,
    }

    // Serialize `v` into the stream with its struct padding explicitly zeroed, so two
    // constructions of the same value produce the same bytes.
    fn put_zeroed<T>(s: &mut StreamOut, ty: D3D12_PIPELINE_STATE_SUBOBJECT_TYPE, v: &T, pads: &[std::ops::Range<usize>]) {
        let mut b = unsafe { std::slice::from_raw_parts(v as *const T as *const u8, std::mem::size_of::<T>()) }.to_vec();
        for p in pads {
            b[p.clone()].fill(0);
        }
        s.raw(ty.0 as u32, &b, std::mem::align_of::<T>() == 8);
    }

    fn game_like_stream(p: &StreamParams) -> Built {
        let mut s = StreamOut::default();
        s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_ROOT_SIGNATURE, &(p.rs as *mut c_void));
        s.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_MS, &p.ms);
        if let Some(ps) = &p.ps {
            s.shader(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PS, ps);
        }
        s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_PRIMITIVE_TOPOLOGY, &p.topology);
        s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_SAMPLE_MASK, &p.sample_mask);
        s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL_FORMAT, &p.ds_format);
        let mut rt: D3D12_RT_FORMAT_ARRAY = unsafe { std::mem::zeroed() };
        rt.NumRenderTargets = p.rt_count;
        rt.RTFormats[0] = DXGI_FORMAT(p.rt0);
        s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RENDER_TARGET_FORMATS, &rt);
        let mut blend: D3D12_BLEND_DESC = unsafe { std::mem::zeroed() };
        blend.RenderTarget[0].SrcBlend = D3D12_BLEND(p.blend_src);
        let blend_pads: Vec<std::ops::Range<usize>> = (0..8).map(|i| 8 + 40 * i + 37..8 + 40 * i + 40).collect();
        put_zeroed(&mut s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_BLEND, &blend, &blend_pads);
        let mut rast: D3D12_RASTERIZER_DESC = unsafe { std::mem::zeroed() };
        rast.CullMode = D3D12_CULL_MODE(p.cull);
        s.put(D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_RASTERIZER, &rast);
        let mut ds: D3D12_DEPTH_STENCIL_DESC = unsafe { std::mem::zeroed() };
        ds.DepthFunc = D3D12_COMPARISON_FUNC(p.depth_func);
        put_zeroed(&mut s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_DEPTH_STENCIL, &ds, &[18..20]);
        let so_decl = vec![D3D12_SO_DECLARATION_ENTRY {
            Stream: p.so_stream,
            SemanticName: PCSTR(p.name),
            SemanticIndex: 0,
            StartComponent: 0,
            ComponentCount: 3,
            OutputSlot: 0,
        }];
        let strides = vec![p.stride, 0, 0, 0];
        let mut so: D3D12_STREAM_OUTPUT_DESC = unsafe { std::mem::zeroed() };
        so.pSODeclaration = so_decl.as_ptr();
        so.NumEntries = so_decl.len() as u32;
        so.pBufferStrides = strides.as_ptr();
        so.NumStrides = strides.len() as u32;
        put_zeroed(&mut s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_STREAM_OUTPUT, &so, &[12..16, 28..32]);
        let il = vec![D3D12_INPUT_ELEMENT_DESC {
            SemanticName: PCSTR(p.name),
            SemanticIndex: 0,
            Format: DXGI_FORMAT(p.il_format),
            InputSlot: 0,
            AlignedByteOffset: 0,
            InputSlotClass: D3D12_INPUT_CLASSIFICATION_PER_VERTEX_DATA,
            InstanceDataStepRate: 0,
        }];
        let mut ill: D3D12_INPUT_LAYOUT_DESC = unsafe { std::mem::zeroed() };
        ill.pInputElementDescs = il.as_ptr();
        ill.NumElements = il.len() as u32;
        put_zeroed(&mut s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_INPUT_LAYOUT, &ill, &[12..16]);
        let vi = vec![D3D12_VIEW_INSTANCE_LOCATION { ViewportArrayIndex: 0, RenderTargetArrayIndex: p.vi_rtai }; 2];
        let mut vid: D3D12_VIEW_INSTANCING_DESC = unsafe { std::mem::zeroed() };
        vid.ViewInstanceCount = p.vi_count;
        vid.pViewInstanceLocations = vi.as_ptr();
        put_zeroed(&mut s, D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_VIEW_INSTANCING, &vid, &[4..8, 20..24]);
        s.put(
            D3D12_PIPELINE_STATE_SUBOBJECT_TYPE_CACHED_PSO,
            &D3D12_CACHED_PIPELINE_STATE { pCachedBlob: p.blob.as_ptr() as _, CachedBlobSizeInBytes: p.blob.len() },
        );
        Built { stream: s, _so_decl: so_decl, _strides: strides, _il: il, _vi: vi }
    }

    // The absolute byte ranges of a stream that hold an address: the only bytes that may differ
    // between two creations of the same logical pipeline.
    fn pointer_field_ranges(buf: &[u8]) -> Vec<std::ops::Range<usize>> {
        let mut out = Vec::new();
        let mut off = 0usize;
        while off + 8 <= buf.len() {
            let ty = u32::from_le_bytes(buf[off..off + 4].try_into().unwrap());
            let Some((psize, palign)) = crate::hle::d3d12x::null::subobject_payload(ty) else { break };
            let poff = if palign == 8 { 8 } else { 4 };
            let p = off + poff;
            match ty {
                0 | 1..=6 | 12 | 19 | 24 | 25 => out.push(p..p + 8),
                7 => {
                    out.push(p..p + 8);
                    out.push(p + 16..p + 24);
                }
                22 => out.push(p + 8..p + 16),
                _ => {}
            }
            off += (poff + psize + 7) & !7;
        }
        out
    }

    #[test]
    fn same_logical_pipeline_two_creations_share_the_key() {
        let name_b = b"POSITION\0".to_vec();
        let mut a = StreamParams::default();
        let mut b = StreamParams { name: name_b.as_ptr(), ..StreamParams::default() };
        b.rs = 0x2000;
        b.ms = vec![1, 2, 3, 4, 5]; // same content, different allocation
        b.ps = Some(vec![9, 8, 7]);
        b.blob = vec![0xAA; 8];
        let sa = game_like_stream(&a);
        let sb = game_like_stream(&b);
        assert_ne!(sa.stream.0, sb.stream.0);
        let ranges = pointer_field_ranges(&sa.stream.0);
        let diffs: Vec<usize> = (0..sa.stream.0.len()).filter(|&i| sa.stream.0[i] != sb.stream.0[i]).collect();
        assert!(!diffs.is_empty());
        for d in &diffs {
            assert!(ranges.iter().any(|r| r.contains(d)), "byte {d:#x} varies outside the pointer fields");
        }
        let mut za = sa.stream.0.clone();
        let mut zb = sb.stream.0.clone();
        for r in &ranges {
            za[r.clone()].fill(0);
            zb[r.clone()].fill(0);
        }
        assert_eq!(za, zb);
        unsafe {
            assert_eq!(
                two_pass_stream_key(&sa.stream.0, &[], 0x1234, 1),
                two_pass_stream_key(&sb.stream.0, &[], 0x1234, 1)
            );
        }
    }

    #[test]
    fn stream_key_changes_with_each_distinguishing_field() {
        let base = StreamParams::default();
        let k = |p: &StreamParams| {
            let b = game_like_stream(p);
            unsafe { two_pass_stream_key(&b.stream.0, &[], 0x1234, 1).unwrap() }
        };
        let k0 = k(&base);
        fn flip(base: &StreamParams, k: impl Fn(&StreamParams) -> (u64, usize), k0: (u64, usize), what: &str, f: impl Fn(&mut StreamParams)) {
            let mut p = base.clone();
            f(&mut p);
            assert_ne!(k0, k(&p), "the key must change when {what} changes");
        }
        flip(&base, k, k0, "MS bytecode", |p| p.ms = vec![1, 2, 3, 4, 6]);
        flip(&base, k, k0, "PS bytecode", |p| p.ps = Some(vec![9, 8, 8]));
        flip(&base, k, k0, "PS presence", |p| p.ps = None);
        flip(&base, k, k0, "semantic name behind the SO/IL pointers", |p| p.name = b"TEXCOORD\0".as_ptr());
        flip(&base, k, k0, "stream-output entry", |p| p.so_stream = 1);
        flip(&base, k, k0, "stream-output buffer stride", |p| p.stride = 32);
        flip(&base, k, k0, "input element format", |p| p.il_format = 45);
        flip(&base, k, k0, "view instance count", |p| p.vi_count = 1);
        flip(&base, k, k0, "view instance location content", |p| p.vi_rtai = 1);
        flip(&base, k, k0, "sample mask", |p| p.sample_mask = 0x0FFF_FFFF);
        flip(&base, k, k0, "depth-stencil format", |p| p.ds_format = 50);
        flip(&base, k, k0, "render-target count", |p| p.rt_count = 1);
        flip(&base, k, k0, "render-target format", |p| p.rt0 = 45);
        flip(&base, k, k0, "blend src", |p| p.blend_src = D3D12_BLEND_ONE.0);
        flip(&base, k, k0, "cull mode", |p| p.cull = D3D12_CULL_MODE_NONE.0);
        flip(&base, k, k0, "depth func", |p| p.depth_func = D3D12_COMPARISON_FUNC_ALWAYS.0);
        flip(&base, k, k0, "topology", |p| p.topology = D3D12_PRIMITIVE_TOPOLOGY_TYPE_LINE.0);
        // samplers, driver id and memory-window count stay in the key
        let s = game_like_stream(&base);
        unsafe {
            assert_ne!(k0, two_pass_stream_key(&s.stream.0, &[([1, 2, 3, 4], false)], 0x1234, 1).unwrap());
            assert_ne!(k0, two_pass_stream_key(&s.stream.0, &[], 0x5678, 1).unwrap());
            assert_ne!(k0, two_pass_stream_key(&s.stream.0, &[], 0x1234, 4 | crate::gpu::recomp::HEAP_WINDOWS).unwrap());
        }
    }
}

