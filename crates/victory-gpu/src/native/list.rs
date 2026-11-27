//! A command list on the native host: the calls a title makes on its list (`SetPipelineState`,
//! `SetComputeRoot*`, `Dispatch`, `CopyBufferRegion`, `ClearRenderTargetView`, `DrawInstanced`, ...)
//! recorded as they arrive and turned into backend passes at submission ([`super::host::NativeHost::execute`]).
//!
//! Title addresses are kept as addresses: they resolve to GPU buffers when the list is submitted, by
//! which time the pool observer has told us about every mapping the title made.

use std::collections::BTreeMap;
use std::rc::Rc;
use victory_backend::backend::{RootLayout, RootParam};
use victory_backend::resources::{
    ClearRect, DrawCall, DrawState, GraphicsBackend, HeapKind, Topology, Viewport,
};

/// `(record, pool_generation, resources_epoch)` of a heap entry whose mirror attempt has not completed yet.
type PendingMirror = ([u8; 12], u64, u64);

/// Per-entry bitsets of buffer views written by `NativeHost::mirror` that have not yet materialized
/// a backend `texture_buffer` view (`typed`: all typed and 4-, 8-, or 16-byte structured views;
/// `explicit`: `stride == 0` typed views). Two bits per entry (~10 KiB for a 40,000-entry heap).
#[derive(Default)]
pub struct PendingTyped {
    typed: Vec<u64>,
    explicit: Vec<u64>,
    explicit_count: usize,
}

impl PendingTyped {
    pub fn set(&mut self, index: usize, is_typed: bool, is_explicit: bool) {
        let word = index >> 6;
        if !is_typed && word >= self.typed.len() {
            return;
        }
        if word >= self.typed.len() {
            self.typed.resize(word + 1, 0);
            self.explicit.resize(word + 1, 0);
        }
        let bit = 1u64 << (index & 63);
        if is_typed {
            self.typed[word] |= bit;
        } else {
            self.typed[word] &= !bit;
        }
        let was_explicit = (self.explicit[word] & bit) != 0;
        let now_explicit = is_typed && is_explicit;
        if now_explicit && !was_explicit {
            self.explicit[word] |= bit;
            self.explicit_count += 1;
        } else if !now_explicit && was_explicit {
            self.explicit[word] &= !bit;
            self.explicit_count -= 1;
        }
    }

    pub fn contains(&self, index: usize) -> bool {
        self.typed
            .get(index >> 6)
            .is_some_and(|w| (w >> (index & 63)) & 1 != 0)
    }

    pub fn count(&self) -> usize {
        self.typed.iter().map(|w| w.count_ones() as usize).sum()
    }

    pub fn has_explicit(&self) -> bool {
        self.explicit_count > 0
    }

    pub fn drain_range(&mut self, start: usize, end: usize, out: &mut Vec<usize>) {
        let end = end.min(self.typed.len() * 64);
        if start >= end {
            return;
        }
        let (first_w, last_w) = (start >> 6, (end - 1) >> 6);
        for w in first_w..=last_w {
            let lo = if w == first_w { start & 63 } else { 0 };
            let hi = if w == last_w { ((end - 1) & 63) + 1 } else { 64 };
            let hi_mask = if hi == 64 { u64::MAX } else { (1u64 << hi) - 1 };
            let mask = hi_mask & !((1u64 << lo) - 1);
            let mut bits = self.typed[w] & mask;
            if bits == 0 {
                continue;
            }
            self.typed[w] &= !mask;
            let ex = self.explicit[w] & mask;
            if ex != 0 {
                self.explicit[w] &= !mask;
                self.explicit_count -= ex.count_ones() as usize;
            }
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                out.push(w * 64 + b);
            }
        }
    }

    pub fn drain_explicit(&mut self, out: &mut Vec<usize>) {
        if self.explicit_count == 0 {
            return;
        }
        self.explicit_count = 0;
        for (w, slot) in self.explicit.iter_mut().enumerate() {
            let mut bits = std::mem::take(slot);
            if bits == 0 {
                continue;
            }
            self.typed[w] &= !bits;
            while bits != 0 {
                let b = bits.trailing_zeros() as usize;
                bits &= bits - 1;
                out.push(w * 64 + b);
            }
        }
    }
}

/// A heap the title wrote views into: `start` is its descriptor memory (title addresses).
pub struct NativeHeap<B: GraphicsBackend> {
    pub heap: B::Heap,
    pub kind: HeapKind,
    pub start: u64,
    pub count: usize,
    /// View pairs of the pool-offset contract ([`super::addr`]) written into the heap's reserved
    /// entries so far, and the pairs' epoch they were written in.
    pub pairs: std::cell::Cell<(u64, usize)>,
    /// The page table built for this heap (its view indices are the heap's own) and the pool-table
    /// generation it is current for.
    pub table: std::cell::RefCell<Option<(u64, Rc<super::addr::Pages<B>>)>>,
    /// The view-stride table (the translator's root SRV `t255`: a dword an entry, see
    /// `NativeHost::view_strides`), made the first time a shader needs it and kept current by
    /// every mirrored write after that.
    pub strides: std::cell::RefCell<Option<Rc<super::addr::Pages<B>>>>,
    /// The view addresses (the translator's root SRV `t1` of the memory space: four dwords an
    /// entry, see `NativeHost::view_addresses`), made the first time a shader reads them and kept
    /// current by every mirrored write after that.
    pub addresses: std::cell::RefCell<Option<Rc<super::addr::Pages<B>>>>,
    /// The title record (tag, view index, generation) each entry was last mirrored from.
    pub mirrored: std::cell::RefCell<std::collections::HashMap<usize, [u8; 12]>>,
    /// Entries whose last mirror attempt could not complete yet, keyed by `(record, pool_gen, res_gen)`
    /// so unchanged entries are retried only once the pool table or registered resources change.
    pub pending_mirror: std::cell::RefCell<std::collections::HashMap<usize, PendingMirror>>,
    pub images: std::cell::RefCell<Shown>,
    /// With `VICTORY_TRACE_HEAP_SET`: the frame (+1; 0: never) each entry was last inside a table a
    /// draw or dispatch bound (diagnostics).
    pub used: std::cell::RefCell<Vec<u32>>,
    /// Buffer views over title memory, by entry: their range and the signature of the title's
    /// mappings under it when the entry was written ([`super::memory::PoolTable::signature`]):
    /// `Some` for a view one pool run holds or a wrap of the title's own pages, `None` for the
    /// null view a gap left. The host mirrors such an entry again when that changes
    /// (`NativeHost::follow_title_pages`).
    pub title_views: std::cell::RefCell<std::collections::HashMap<usize, TitleView>>,
    /// Unmaterialized `texture_buffer` entry bits set by `NativeHost::mirror` and promoted lazily
    /// at draw or dispatch when a bound shader uses DXIL `TypedBuffer` operations.
    pub typed_pending: std::cell::RefCell<PendingTyped>,
}

/// A heap buffer view over title memory: its address, length and the signature of the mappings
/// under it when written (see [`NativeHeap::title_views`]).
pub type TitleView = (u64, u64, Option<u64>);

/// The title texture each heap entry shows (`of`), and each texture's entries (`entries`; those
/// may be stale: an entry counts only while `of` still names the texture).
#[derive(Default)]
pub struct Shown {
    of: std::collections::HashMap<usize, usize>,
    entries: std::collections::HashMap<usize, Vec<usize>>,
    /// The entries cleared when a texture was released, by its object's address.
    gone: std::collections::HashMap<usize, Vec<usize>>,
}

impl<B: GraphicsBackend> NativeHeap<B> {
    /// Entry `index` now shows title texture `resource` (`None`: a buffer or a sampler).
    pub fn shows(&self, index: usize, resource: Option<usize>) {
        let Shown { of, entries, .. } = &mut *self.images.borrow_mut();
        match resource {
            Some(r) => {
                if of.insert(index, r) != Some(r) {
                    entries.entry(r).or_default().push(index);
                }
            }
            None => {
                of.remove(&index);
            }
        }
    }

    /// Whether an entry shows title texture `resource`.
    pub fn shows_texture(&self, resource: usize) -> bool {
        let Shown { of, entries, .. } = &*self.images.borrow();
        entries.get(&resource).is_some_and(|e| e.iter().any(|i| of.get(i) == Some(&resource)))
    }

    /// The entries that show title texture `resource`, which they stop showing.
    pub fn take_entries(&self, resource: usize) -> Vec<usize> {
        let Shown { of, entries, gone } = &mut *self.images.borrow_mut();
        let mut listed = entries.remove(&resource).unwrap_or_default();
        listed.retain(|i| of.get(i) == Some(&resource));
        listed.sort_unstable();
        listed.dedup();
        listed.iter().for_each(|i| {
            of.remove(i);
        });
        if !listed.is_empty() {
            gone.insert(resource, listed.clone());
        }
        listed
    }

    /// The entries [`Self::take_entries`] took when the texture last at `resource` was released,
    /// once: the title has made a texture at that address again.
    pub fn take_gone(&self, resource: usize) -> Vec<usize> {
        self.images.borrow_mut().gone.remove(&resource).unwrap_or_default()
    }
}

/// How a translated title shader's arguments are laid out (`victory_shader::emit::HostShader`):
/// one block of root constants for the user data, then the root constant, shader resource and
/// unordered access views it reads, as addresses. Without it, the title's root parameters go to the
/// shader as they are, in order (shaders compiled for the native host).
#[derive(Clone, Debug, Default)]
pub struct Binding {
    /// Dwords of the user-data block (a multiple of 4).
    pub dwords: usize,
    pub user_data: Vec<victory_shader::emit::UserData>,
    /// Title root parameter indices of the root CBVs, SRVs and UAVs, in this order, with their
    /// kind ([`View`]).
    pub views: Vec<(u8, View)>,
    /// The shader finds title memory through the page table (`t0` of the memory space), which
    /// follows the views in the argument buffer, and the heap's reserved view pairs.
    pub memory: bool,
    /// The shader reads views' title addresses or texture sizes from their descriptors: the
    /// heap's view addresses (`t1` of the memory space) follow the page table, if any.
    pub view_addresses: bool,
    /// The pixel stage's own arguments, when the pipeline has a pixel shader that reads any
    /// (a stage has an argument block of its own; this one is the vertex stage's).
    pub ps: Option<Rc<Binding>>,
    /// Some user data takes the heap index of a view of a root view's buffer (and the address's
    /// offset in it): the heap needs the view pairs.
    pub root_views: bool,
    /// The shader's immediate samplers, as [`RootLayout::static_samplers`] declares them.
    pub static_samplers: Vec<[u32; 10]>,
    /// Minimum descriptor-table entries needed per `RootType::Table` root slot: `(slot, count)`.
    pub table_entries: Vec<(u8, usize)>,
    /// Whether the shader's DXIL uses `TypedBuffer` operations (`bufferLoad`, `bufferStore`, or
    /// `getDimensions` on a `TypedBuffer` handle).
    pub typed_buffers: bool,
    /// Whether the shader both uses `TypedBuffer` operations and indexes
    /// `ResourceDescriptorHeap` directly (`HostShader::heap`).
    pub heap_typed_buffers: bool,
    /// Resource-heap entries this stage may dereference when residency sets are off.
    pub heap_access: StageHeapAccess,
    /// Pixel stage's resource-heap access when the render pipeline has a pixel shader (kept even
    /// when `ps` is `None` because the pixel shader had no user-data or root views of its own).
    pub ps_heap_access: Option<StageHeapAccess>,
    /// The pipeline's name (its shaders' hashes), for diagnostics.
    pub name: Rc<str>,
}

/// One statically bounded resource-heap access of a translated shader stage.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum StageHeapElem {
    /// Entry `offset` (in 32-byte resource descriptors) past root parameter `slot`'s `Root::Table`.
    Table { slot: u8, offset: u32, writes: bool },
    /// Pool-offset contract view pairs `[heap.count .. heap.count + 2 * pairs)`.
    Pairs { writes: bool },
    /// Fixed constant heap index `index`.
    Fixed { index: u32, writes: bool },
}

/// Resource-heap access summary of a single shader stage (`Binding::of`).
#[derive(Clone, Debug, PartialEq, Eq)]
pub struct StageHeapAccess {
    /// Whether this stage has any unbounded resource-heap read or write.
    pub unbounded: bool,
    /// Whether any unbounded resource-heap handle in this stage is an unordered-access view.
    pub unbounded_writes: bool,
    /// Deduplicated statically-bounded resource-heap accesses of this stage.
    pub bounded: Vec<StageHeapElem>,
}

impl Default for StageHeapAccess {
    fn default() -> Self {
        StageHeapAccess {
            unbounded: true,
            unbounded_writes: true,
            bounded: Vec::new(),
        }
    }
}

/// A root view's kind.
#[derive(Clone, Copy, Debug, PartialEq, Eq)]
pub enum View {
    Cbv,
    Srv,
    Uav,
}

impl Binding {
    pub fn of(shader: &victory_shader::emit::HostShader) -> Binding {
        let info = analyze_dxil(&shader.dxil, &shader.user_data);
        let typed_buffers = info.typed_buffers;
        Binding {
            dwords: shader.user_data_dwords() as usize,
            user_data: shader.user_data.clone(),
            views: shader
                .root_cbvs
                .iter()
                .map(|&r| (r, View::Cbv))
                .chain(shader.root_srvs.iter().map(|&r| (r, View::Srv)))
                .chain(shader.root_uavs.iter().map(|&r| (r, View::Uav)))
                .collect(),
            memory: shader.memory,
            view_addresses: shader.view_addresses,
            ps: None,
            root_views: shader.user_data.iter().any(|u| {
                u.root == victory_shader::RootType::Descriptor
                    && u.dword >= victory_shader::emit::ROOT_SRV_VIEW
            }),
            static_samplers: shader
                .static_samplers
                .iter()
                .map(|s| {
                    // Anisotropic filtering with a point mip filter becomes linear-mip, as on
                    // D3D12 devices without it.
                    let [u, v, w] = s.address;
                    [s.filter_for_host(false), u, v, w, s.mip_lod_bias.to_bits(), s.max_anisotropy, s.comparison, s.border, s.min_lod.to_bits(), s.max_lod.to_bits()]
                })
                .collect(),
            table_entries: shader
                .table_entries
                .iter()
                .map(|&(slot, count)| (slot, count as usize))
                .collect(),
            typed_buffers,
            heap_typed_buffers: typed_buffers && shader.heap,
            heap_access: info.heap_access,
            ps_heap_access: None,
            name: "".into(),
        }
    }

    /// The root signature the converter lays the argument buffer out by, in the order
    /// `NativeHost::translated_args` fills it: the user-data block, the views, the page table, the
    /// view addresses.
    /// `None` when the shader has no arguments (it takes its own signature, if it has one).
    pub fn layout(&self) -> Option<RootLayout> {
        use victory_shader::emit::{MEMORY_SPACE, ROOT_CBV_SPACE, USER_DATA_SPACE};
        let mut params = Vec::new();
        if self.dwords > 0 {
            params.push(RootParam::Constants {
                register: 0,
                space: USER_DATA_SPACE,
                dwords: self.dwords as u32,
            });
        }
        for &(slot, kind) in &self.views {
            let (register, space) = (u32::from(slot), ROOT_CBV_SPACE);
            params.push(match kind {
                View::Cbv => RootParam::Cbv { register, space },
                View::Srv => RootParam::Srv { register, space },
                View::Uav => RootParam::Uav { register, space },
            });
        }
        if self.memory {
            params.push(RootParam::Srv {
                register: 0,
                space: MEMORY_SPACE,
            });
        }
        if self.view_addresses {
            params.push(RootParam::Srv {
                register: 1,
                space: MEMORY_SPACE,
            });
        }
        (!params.is_empty() || !self.static_samplers.is_empty()).then(|| RootLayout {
            params,
            // Memory no mapped pool holds is read through a view of size 0.
            bounds_checked: true,
            static_samplers: self.static_samplers.clone(),
        })
    }
}

struct DxilInfo {
    typed_buffers: bool,
    heap_access: StageHeapAccess,
}

/// Inspects a DXBC-wrapped DXIL module emitted by `victory-shader` for:
/// 1. `TypedBuffer` operations (`dx.op.bufferLoad.*`, `dx.op.bufferStore.*`, or `dx.op.getDimensions`
///    on `ResourceProperties` kind `10`), and
/// 2. Every resource `dx.op.createHandleFromHeap` / `dx.op.annotateHandle` call, proving which heap
///    entries are statically bounded (`Table`, `Pairs`, `Fixed`) vs unbounded (`unbounded`).
///
/// Defaults conservatively on unrecognized or abbreviated bitcode.
fn analyze_dxil(dxil: &[u8], user_data: &[victory_shader::emit::UserData]) -> DxilInfo {
    dxil_bitcode(dxil)
        .and_then(|bc| scan_dxil_info(bc, user_data))
        .unwrap_or(DxilInfo {
            typed_buffers: true,
            heap_access: StageHeapAccess::default(),
        })
}

fn dxil_bitcode(dxil: &[u8]) -> Option<&[u8]> {
    if dxil.len() < 32 || &dxil[..4] != b"DXBC" {
        return None;
    }
    let parts = u32::from_le_bytes(dxil[28..32].try_into().ok()?) as usize;
    if 32usize.checked_add(parts.checked_mul(4)?)? > dxil.len() {
        return None;
    }
    for i in 0..parts {
        let off = u32::from_le_bytes(dxil[32 + i * 4..36 + i * 4].try_into().ok()?) as usize;
        let hdr = dxil.get(off..off.checked_add(8)?)?;
        if &hdr[..4] != b"DXIL" {
            continue;
        }
        let part_len = u32::from_le_bytes(hdr[4..8].try_into().ok()?) as usize;
        let part = dxil.get(off + 8..(off + 8).checked_add(part_len)?)?;
        if part.len() < 24 || &part[8..12] != b"DXIL" {
            return None;
        }
        let bc_off = 8usize.checked_add(u32::from_le_bytes(part[16..20].try_into().ok()?) as usize)?;
        let bc_len = u32::from_le_bytes(part[20..24].try_into().ok()?) as usize;
        let bc = part.get(bc_off..bc_off.checked_add(bc_len)?)?;
        return (bc.len() >= 4 && &bc[..4] == b"BC\xC0\xDE" && bc.len().is_multiple_of(4))
            .then(|| &bc[4..]);
    }
    None
}

fn decode_signed(raw: u64) -> i64 {
    if raw & 1 == 0 {
        (raw >> 1) as i64
    } else {
        -((raw >> 1) as i64)
    }
}

fn dedup_stage_elems(elems: &mut Vec<StageHeapElem>) {
    let key = |e: &StageHeapElem| match *e {
        StageHeapElem::Table { slot, offset, writes } => (0u8, slot, offset, !writes),
        StageHeapElem::Pairs { writes } => (1, 0, 0, !writes),
        StageHeapElem::Fixed { index, writes } => (2, 0, index, !writes),
    };
    elems.sort_unstable_by_key(key);
    elems.dedup_by(|b, a| {
        let ka = key(a);
        let kb = key(b);
        (ka.0, ka.1, ka.2) == (kb.0, kb.1, kb.2)
    });
}

fn scan_dxil_info(
    bc: &[u8],
    user_data: &[victory_shader::emit::UserData],
) -> Option<DxilInfo> {
    use victory_shader::RootType;
    use victory_shader::emit::{
        HEAP_VIEWS, MEMORY_SPACE, ROOT_SRV_VIEW, ROOT_UAV_VIEW, USER_DATA_SPACE,
    };

    #[derive(Clone, Copy)]
    enum PendingStruct {
        None,
        Props,
        Bind,
    }

    #[derive(Clone, Copy)]
    enum Node {
        Other,
        UserCbv,
        MemoryTable,
        UserRow(u32),
        UserDword(u32),
        MemLoadVec,
        MemLoadF32,
        PairIndex,
        Add(u32, u32),
        Sub(u32, u32),
        LShr(u32, u32),
        Select(u32, u32),
        Phi(u32, u16),
    }

    struct BitReader<'a> {
        words: &'a [u8],
        bit: usize,
    }
    impl BitReader<'_> {
        fn bits(&mut self, width: u32, end: usize) -> Option<u64> {
            let mut left = width as usize;
            if self.bit.checked_add(left)? > end {
                return None;
            }
            let (mut out, mut shift) = (0u64, 0usize);
            while left > 0 {
                let word_idx = self.bit >> 5;
                let bit_in_word = self.bit & 31;
                let bytes: [u8; 4] = self.words.get(word_idx * 4..word_idx * 4 + 4)?.try_into().ok()?;
                let word = u32::from_le_bytes(bytes);
                let take = left.min(32 - bit_in_word);
                let mask = if take == 32 { u64::from(u32::MAX) } else { (1u64 << take) - 1 };
                out |= ((u64::from(word) >> bit_in_word) & mask) << shift;
                self.bit += take;
                shift += take;
                left -= take;
            }
            Some(out)
        }
        fn vbr(&mut self, width: u32, end: usize) -> Option<u64> {
            let step = width - 1;
            let hi = 1u64 << step;
            let (mut out, mut shift) = (0u64, 0u32);
            loop {
                let chunk = self.bits(width, end)?;
                if shift >= 64 {
                    return None;
                }
                out |= (chunk & (hi - 1)) << shift;
                if chunk & hi == 0 {
                    return Some(out);
                }
                shift += step;
            }
        }
        fn align32(&mut self) {
            self.bit = (self.bit + 31) & !31;
        }
    }

    let total_bits = bc.len().checked_mul(8)?;
    let mut r = BitReader { words: bc, bit: 0 };
    if r.bits(2, total_bits)? != 1 || r.vbr(8, total_bits)? != 8 {
        return None;
    }
    let mod_w = r.vbr(4, total_bits)? as u32;
    r.align32();
    let mod_words = r.bits(32, total_bits)? as usize;
    let mod_end = r.bit.checked_add(mod_words.checked_mul(32)?)?;
    if mod_end > total_bits {
        return None;
    }

    let mut void_ty: Option<u64> = None;
    let mut props_ty: Option<u64> = None;
    let mut bind_ty: Option<u64> = None;
    let mut func_types: Vec<Option<(bool, u32)>> = Vec::new();
    let mut defined_func_tys: Vec<u64> = Vec::new();
    let mut func_idx = 0usize;
    let mut value_base = 0u64;
    let mut const_int: Vec<Option<u32>> = Vec::new();
    let mut res_props: Vec<Option<u32>> = Vec::new();
    let mut res_bind: Vec<Option<(u32, u32, u8)>> = Vec::new();
    let mut has_typed_props = false;
    let mut has_buf_load_store = false;
    let mut has_get_dims = false;
    let mut seen_symtab = false;
    let mut fn_create_from_binding: Option<u64> = None;
    let mut fn_create_from_heap: Option<u64> = None;
    let mut fn_annotate: Option<u64> = None;
    let mut fn_cbuf_load_i32: Option<u64> = None;
    let mut fn_raw_buf_load_f32: Option<u64> = None;
    let mut stage_unbounded = false;
    let mut stage_unbounded_writes = false;
    let mut stage_bounded: Vec<StageHeapElem> = Vec::new();
    let mut scratch = Vec::new();
    let mut ops = Vec::new();

    while r.bit < mod_end {
        match r.bits(mod_w, mod_end)? {
            0 => break,
            3 => {
                let code = r.vbr(6, mod_end)?;
                let n = r.vbr(6, mod_end)? as usize;
                ops.clear();
                for _ in 0..n {
                    ops.push(r.vbr(6, mod_end)?);
                }
                if code == 7 {
                    value_base += 1;
                } else if code == 8 {
                    value_base += 1;
                    if ops.len() >= 3 && ops[2] == 0 {
                        defined_func_tys.push(ops[0]);
                    }
                }
            }
            1 => {
                let sub_id = r.vbr(8, mod_end)?;
                let sub_w = r.vbr(4, mod_end)? as u32;
                r.align32();
                let sub_words = r.bits(32, mod_end)? as usize;
                let sub_end = r.bit.checked_add(sub_words.checked_mul(32)?)?;
                if sub_end > mod_end {
                    return None;
                }
                match sub_id {
                    // TYPE_BLOCK: locate `void`, `ResourceProperties`, `ResBind`, and function types.
                    17 => {
                        let (mut ty_idx, mut pending) = (0u64, PendingStruct::None);
                        while r.bit < sub_end {
                            match r.bits(sub_w, sub_end)? {
                                0 => break,
                                3 => {
                                    let code = r.vbr(6, sub_end)?;
                                    let n = r.vbr(6, sub_end)? as usize;
                                    if code == 19 {
                                        scratch.clear();
                                        for _ in 0..n {
                                            scratch.push(r.vbr(6, sub_end)? as u8);
                                        }
                                        pending = if scratch == b"dx.types.ResourceProperties" {
                                            PendingStruct::Props
                                        } else if scratch == b"dx.types.ResBind" {
                                            PendingStruct::Bind
                                        } else {
                                            PendingStruct::None
                                        };
                                    } else {
                                        ops.clear();
                                        for _ in 0..n {
                                            ops.push(r.vbr(6, sub_end)?);
                                        }
                                        if code != 1 {
                                            let fty = if code == 2 {
                                                void_ty = Some(ty_idx);
                                                None
                                            } else if code == 20 {
                                                match pending {
                                                    PendingStruct::Props => props_ty = Some(ty_idx),
                                                    PendingStruct::Bind => bind_ty = Some(ty_idx),
                                                    PendingStruct::None => {}
                                                }
                                                None
                                            } else if code == 21 && ops.len() >= 2 {
                                                Some((Some(ops[1]) == void_ty, (ops.len() - 2) as u32))
                                            } else {
                                                None
                                            };
                                            func_types.push(fty);
                                            pending = PendingStruct::None;
                                            ty_idx += 1;
                                        }
                                    }
                                }
                                _ => return None,
                            }
                        }
                        r.bit = sub_end;
                    }
                    // CONSTANTS_BLOCK: record integer constants, `ResourceProperties`, and `ResBind`.
                    11 => {
                        let mut cur_ty = None;
                        while r.bit < sub_end {
                            match r.bits(sub_w, sub_end)? {
                                0 => break,
                                3 => {
                                    let code = r.vbr(6, sub_end)?;
                                    let n = r.vbr(6, sub_end)? as usize;
                                    ops.clear();
                                    for _ in 0..n {
                                        ops.push(r.vbr(6, sub_end)?);
                                    }
                                    if code == 1 && !ops.is_empty() {
                                        cur_ty = Some(ops[0]);
                                    } else if code == 2 {
                                        const_int.push(Some(0));
                                        res_props.push((cur_ty == props_ty && props_ty.is_some()).then_some(0));
                                        res_bind.push((cur_ty == bind_ty && bind_ty.is_some()).then_some((0, 0, 0)));
                                    } else if code == 4 && !ops.is_empty() {
                                        let val = decode_signed(ops[0]) as u32;
                                        const_int.push(Some(val));
                                        res_props.push(None);
                                        res_bind.push(None);
                                    } else if code == 7 {
                                        let int_at = |id: u64, ci: &[Option<u32>]| {
                                            id.checked_sub(value_base)
                                                .and_then(|k| ci.get(k as usize).copied().flatten())
                                        };
                                        let p = if cur_ty == props_ty && props_ty.is_some() && !ops.is_empty() {
                                            int_at(ops[0], &const_int)
                                        } else {
                                            None
                                        };
                                        if let Some(p0) = p && (p0 & 0xFF) == 10 {
                                            has_typed_props = true;
                                        }
                                        let b = if cur_ty == bind_ty && bind_ty.is_some() && ops.len() >= 4 {
                                            match (int_at(ops[0], &const_int), int_at(ops[2], &const_int), int_at(ops[3], &const_int)) {
                                                (Some(lo), Some(sp), Some(cl)) => Some((lo, sp, cl as u8)),
                                                _ => None,
                                            }
                                        } else {
                                            None
                                        };
                                        const_int.push(None);
                                        res_props.push(p);
                                        res_bind.push(b);
                                    } else {
                                        const_int.push(None);
                                        res_props.push(None);
                                        res_bind.push(None);
                                    }
                                }
                                _ => return None,
                            }
                        }
                        r.bit = sub_end;
                    }
                    // VALUE_SYMTAB_BLOCK: inspect kept `dx.op.*` declarations.
                    14 => {
                        seen_symtab = true;
                        while r.bit < sub_end {
                            match r.bits(sub_w, sub_end)? {
                                0 => break,
                                3 => {
                                    let code = r.vbr(6, sub_end)?;
                                    let n = r.vbr(6, sub_end)? as usize;
                                    if code == 1 && n >= 1 {
                                        let val_id = r.vbr(6, sub_end)?;
                                        scratch.clear();
                                        for _ in 1..n {
                                            scratch.push(r.vbr(6, sub_end)? as u8);
                                        }
                                        if scratch.starts_with(b"dx.op.bufferLoad.")
                                            || scratch.starts_with(b"dx.op.bufferStore.")
                                        {
                                            has_buf_load_store = true;
                                        } else if scratch == b"dx.op.getDimensions" {
                                            has_get_dims = true;
                                        } else if scratch == b"dx.op.createHandleFromBinding" {
                                            fn_create_from_binding = Some(val_id);
                                        } else if scratch == b"dx.op.createHandleFromHeap" {
                                            fn_create_from_heap = Some(val_id);
                                        } else if scratch == b"dx.op.annotateHandle" {
                                            fn_annotate = Some(val_id);
                                        } else if scratch == b"dx.op.cbufferLoadLegacy.i32" {
                                            fn_cbuf_load_i32 = Some(val_id);
                                        } else if scratch == b"dx.op.rawBufferLoad.f32" {
                                            fn_raw_buf_load_f32 = Some(val_id);
                                        }
                                    } else {
                                        for _ in 0..n {
                                            r.vbr(6, sub_end)?;
                                        }
                                    }
                                }
                                _ => return None,
                            }
                        }
                        r.bit = sub_end;
                    }
                    // FUNCTION_BLOCK: trace every resource `createHandleFromHeap` index expression.
                    12 => {
                        if !seen_symtab {
                            return None;
                        }
                        let fty = *defined_func_tys.get(func_idx)?;
                        func_idx += 1;
                        if fn_create_from_heap.is_none() {
                            r.bit = sub_end;
                            continue;
                        }
                        let (_, n_params) = (*func_types.get(fty as usize)?)?;
                        let inst_base = value_base
                            .checked_add(const_int.len() as u64)?
                            .checked_add(u64::from(n_params))?;
                        let mut inst_id = inst_base;
                        let mut nodes: Vec<Node> = Vec::new();
                        let mut phi_args: Vec<u32> = Vec::new();
                        // `(handle_val_id, raw_index_val_id, annotated, writes)`
                        let mut heap_handles: Vec<(u32, u32, bool, bool)> = Vec::new();

                        let const_raw = |id: u64| -> Option<u32> {
                            id.checked_sub(value_base)
                                .and_then(|k| const_int.get(k as usize).copied().flatten())
                        };
                        let props_raw = |id: u64| -> Option<u32> {
                            id.checked_sub(value_base)
                                .and_then(|k| res_props.get(k as usize).copied().flatten())
                        };
                        let bind_raw = |id: u64| -> Option<(u32, u32, u8)> {
                            id.checked_sub(value_base)
                                .and_then(|k| res_bind.get(k as usize).copied().flatten())
                        };

                        while r.bit < sub_end {
                            match r.bits(sub_w, sub_end)? {
                                0 => break,
                                3 => {
                                    let code = r.vbr(6, sub_end)?;
                                    let n = r.vbr(6, sub_end)? as usize;
                                    ops.clear();
                                    for _ in 0..n {
                                        ops.push(r.vbr(6, sub_end)?);
                                    }
                                    let node_at = |id: u32, nodes: &[Node]| -> Option<Node> {
                                        u64::from(id)
                                            .checked_sub(inst_base)
                                            .and_then(|k| nodes.get(k as usize).copied())
                                    };
                                    match code {
                                        1 | 10 | 11 | 44 => {}
                                        2 => {
                                            if ops.len() < 3 {
                                                return None;
                                            }
                                            let a = u32::try_from(inst_id.checked_sub(ops[0])?).ok()?;
                                            let b = u32::try_from(inst_id.checked_sub(ops[1])?).ok()?;
                                            let node = match ops[2] {
                                                0 => Node::Add(a, b),
                                                1 => Node::Sub(a, b),
                                                8 => Node::LShr(a, b),
                                                _ => Node::Other,
                                            };
                                            nodes.push(node);
                                            inst_id += 1;
                                        }
                                        3 => {
                                            if ops.is_empty() {
                                                return None;
                                            }
                                            let v = u32::try_from(inst_id.checked_sub(ops[0])?).ok()?;
                                            let node = if matches!(node_at(v, &nodes), Some(Node::MemLoadF32)) {
                                                Node::PairIndex
                                            } else {
                                                Node::Other
                                            };
                                            nodes.push(node);
                                            inst_id += 1;
                                        }
                                        26 => {
                                            if ops.len() < 2 {
                                                return None;
                                            }
                                            let agg = u32::try_from(inst_id.checked_sub(ops[0])?).ok()?;
                                            let idx = ops[1];
                                            let node = match node_at(agg, &nodes) {
                                                Some(Node::UserRow(row)) if idx < 4 => {
                                                    Node::UserDword(row * 4 + idx as u32)
                                                }
                                                Some(Node::MemLoadVec) if idx == 0 => Node::MemLoadF32,
                                                _ => Node::Other,
                                            };
                                            nodes.push(node);
                                            inst_id += 1;
                                        }
                                        29 => {
                                            if ops.len() < 3 {
                                                return None;
                                            }
                                            let t = u32::try_from(inst_id.checked_sub(ops[0])?).ok()?;
                                            let f = u32::try_from(inst_id.checked_sub(ops[1])?).ok()?;
                                            nodes.push(Node::Select(t, f));
                                            inst_id += 1;
                                        }
                                        16 => {
                                            if ops.is_empty() || !(ops.len() - 1).is_multiple_of(2) {
                                                return None;
                                            }
                                            let pairs = (ops.len() - 1) / 2;
                                            let start = u32::try_from(phi_args.len()).ok()?;
                                            let len = u16::try_from(pairs).ok()?;
                                            for k in 0..pairs {
                                                let srel = decode_signed(ops[1 + 2 * k]);
                                                let target = i64::try_from(inst_id).ok()?.checked_sub(srel)?;
                                                phi_args.push(u32::try_from(target).ok()?);
                                            }
                                            nodes.push(Node::Phi(start, len));
                                            inst_id += 1;
                                        }
                                        20 | 28 | 38 | 43 | 46 => {
                                            nodes.push(Node::Other);
                                            inst_id += 1;
                                        }
                                        34 => {
                                            if ops.len() < 4 || ops[1] & (1 << 15) == 0 {
                                                return None;
                                            }
                                            let (is_void, _) = (*func_types.get(ops[2] as usize)?)?;
                                            let callee = inst_id.checked_sub(ops[3])?;
                                            if is_void {
                                                continue;
                                            }
                                            let arg = |k: usize| -> Option<u64> {
                                                inst_id.checked_sub(*ops.get(4 + k)?)
                                            };
                                            let node = if Some(callee) == fn_create_from_binding && ops.len() == 8 {
                                                match bind_raw(arg(1)?) {
                                                    Some((0, USER_DATA_SPACE, 2)) => Node::UserCbv,
                                                    Some((0, MEMORY_SPACE, 0)) => Node::MemoryTable,
                                                    _ => Node::Other,
                                                }
                                            } else if Some(callee) == fn_annotate && ops.len() == 7 {
                                                let h_val = u32::try_from(arg(1)?).ok()?;
                                                let p_val = arg(2)?;
                                                let is_uav = props_raw(p_val).is_none_or(|p| p & 0x1000 != 0);
                                                for hh in &mut heap_handles {
                                                    if hh.0 == h_val {
                                                        hh.2 = true;
                                                        hh.3 |= is_uav;
                                                    }
                                                }
                                                match node_at(h_val, &nodes) {
                                                    Some(Node::UserCbv) => Node::UserCbv,
                                                    Some(Node::MemoryTable) => Node::MemoryTable,
                                                    _ => Node::Other,
                                                }
                                            } else if Some(callee) == fn_cbuf_load_i32 && ops.len() == 7 {
                                                let cbv_h = u32::try_from(arg(1)?).ok()?;
                                                match (node_at(cbv_h, &nodes), const_raw(arg(2)?)) {
                                                    (Some(Node::UserCbv), Some(row)) => Node::UserRow(row),
                                                    _ => Node::Other,
                                                }
                                            } else if Some(callee) == fn_raw_buf_load_f32 && ops.len() == 10 {
                                                let srv_h = u32::try_from(arg(1)?).ok()?;
                                                if matches!(node_at(srv_h, &nodes), Some(Node::MemoryTable)) {
                                                    Node::MemLoadVec
                                                } else {
                                                    Node::Other
                                                }
                                            } else if Some(callee) == fn_create_from_heap && ops.len() == 8 {
                                                let idx_val = u32::try_from(arg(1)?).ok()?;
                                                match const_raw(arg(2)?) {
                                                    Some(1) => {}
                                                    Some(0) => {
                                                        let h_id = u32::try_from(inst_id).ok()?;
                                                        heap_handles.push((h_id, idx_val, false, false));
                                                    }
                                                    _ => {
                                                        let h_id = u32::try_from(inst_id).ok()?;
                                                        heap_handles.push((h_id, idx_val, true, true));
                                                    }
                                                }
                                                Node::Other
                                            } else {
                                                Node::Other
                                            };
                                            nodes.push(node);
                                            inst_id += 1;
                                        }
                                        _ => return None,
                                    }
                                }
                                _ => return None,
                            }
                        }
                        r.bit = sub_end;

                        let node_of = |id: u32| -> Option<Node> {
                            u64::from(id)
                                .checked_sub(inst_base)
                                .and_then(|k| nodes.get(k as usize).copied())
                        };
                        fn eval_const(
                            v: u32,
                            depth: u8,
                            value_base: u64,
                            const_int: &[Option<u32>],
                            inst_base: u64,
                            nodes: &[Node],
                        ) -> Option<u32> {
                            if depth > 16 {
                                return None;
                            }
                            if let Some(c) = u64::from(v)
                                .checked_sub(value_base)
                                .and_then(|k| const_int.get(k as usize).copied().flatten())
                            {
                                return Some(c);
                            }
                            let n = u64::from(v)
                                .checked_sub(inst_base)
                                .and_then(|k| nodes.get(k as usize).copied())?;
                            match n {
                                Node::Add(a, b) => Some(
                                    eval_const(a, depth + 1, value_base, const_int, inst_base, nodes)?
                                        .wrapping_add(eval_const(b, depth + 1, value_base, const_int, inst_base, nodes)?),
                                ),
                                Node::Sub(a, b) => Some(
                                    eval_const(a, depth + 1, value_base, const_int, inst_base, nodes)?
                                        .wrapping_sub(eval_const(b, depth + 1, value_base, const_int, inst_base, nodes)?),
                                ),
                                Node::LShr(a, b) => {
                                    let s = eval_const(b, depth + 1, value_base, const_int, inst_base, nodes)?;
                                    (s < 32).then(|| {
                                        eval_const(a, depth + 1, value_base, const_int, inst_base, nodes)
                                            .map(|x| x >> s)
                                    })?
                                }
                                _ => None,
                            }
                        }
                        fn eval_table_bytes(
                            v: u32,
                            depth: u8,
                            user_data: &[victory_shader::emit::UserData],
                            value_base: u64,
                            const_int: &[Option<u32>],
                            inst_base: u64,
                            nodes: &[Node],
                        ) -> Option<(u8, u32)> {
                            if depth > 16 {
                                return None;
                            }
                            let n = u64::from(v)
                                .checked_sub(inst_base)
                                .and_then(|k| nodes.get(k as usize).copied())?;
                            match n {
                                Node::UserDword(ud_idx) => user_data
                                    .iter()
                                    .find(|u| u.index == ud_idx && u.root == RootType::Table && u.dword == 0)
                                    .map(|u| (u.slot, 0)),
                                Node::Add(a, b) => {
                                    if let (Some((slot, off)), Some(k)) = (
                                        eval_table_bytes(a, depth + 1, user_data, value_base, const_int, inst_base, nodes),
                                        eval_const(b, depth + 1, value_base, const_int, inst_base, nodes),
                                    ) {
                                        Some((slot, off.wrapping_add(k)))
                                    } else if let (Some(k), Some((slot, off))) = (
                                        eval_const(a, depth + 1, value_base, const_int, inst_base, nodes),
                                        eval_table_bytes(b, depth + 1, user_data, value_base, const_int, inst_base, nodes),
                                    ) {
                                        Some((slot, off.wrapping_add(k)))
                                    } else {
                                        None
                                    }
                                }
                                Node::Sub(a, b) => {
                                    let (slot, off) = eval_table_bytes(
                                        a, depth + 1, user_data, value_base, const_int, inst_base, nodes,
                                    )?;
                                    let k = eval_const(b, depth + 1, value_base, const_int, inst_base, nodes)?;
                                    Some((slot, off.wrapping_sub(k)))
                                }
                                _ => None,
                            }
                        }
                        #[allow(clippy::too_many_arguments)]
                        fn resolve_index(
                            v: u32,
                            writes: bool,
                            user_data: &[victory_shader::emit::UserData],
                            value_base: u64,
                            const_int: &[Option<u32>],
                            inst_base: u64,
                            nodes: &[Node],
                            phi_args: &[u32],
                            visiting: &mut Vec<u32>,
                            seen: &mut Vec<u32>,
                            out: &mut Vec<StageHeapElem>,
                        ) -> bool {
                            if seen.contains(&v) {
                                return true;
                            }
                            if out.len() > 128 {
                                return false;
                            }
                            if let Some(c) = eval_const(v, 0, value_base, const_int, inst_base, nodes) {
                                out.push(StageHeapElem::Fixed { index: c, writes });
                                seen.push(v);
                                return true;
                            }
                            let Some(n) = u64::from(v)
                                .checked_sub(inst_base)
                                .and_then(|k| nodes.get(k as usize).copied())
                            else {
                                return false;
                            };
                            let ok = match n {
                                Node::PairIndex => {
                                    out.push(StageHeapElem::Pairs { writes });
                                    true
                                }
                                Node::Add(a, b) => {
                                    let node_a = u64::from(a).checked_sub(inst_base).and_then(|k| nodes.get(k as usize).copied());
                                    let node_b = u64::from(b).checked_sub(inst_base).and_then(|k| nodes.get(k as usize).copied());
                                    let is_pair_plus_one = (matches!(node_a, Some(Node::PairIndex))
                                        && eval_const(b, 0, value_base, const_int, inst_base, nodes) == Some(1))
                                        || (eval_const(a, 0, value_base, const_int, inst_base, nodes) == Some(1)
                                            && matches!(node_b, Some(Node::PairIndex)));
                                    if is_pair_plus_one {
                                        out.push(StageHeapElem::Pairs { writes });
                                    }
                                    is_pair_plus_one
                                }
                                Node::UserDword(ud_idx) => {
                                    let is_root_pair = user_data.iter().any(|u| {
                                        u.index == ud_idx
                                            && u.root == RootType::Descriptor
                                            && matches!(u.dword, ROOT_SRV_VIEW | ROOT_UAV_VIEW)
                                    });
                                    if is_root_pair {
                                        out.push(StageHeapElem::Pairs { writes });
                                    }
                                    is_root_pair
                                }
                                Node::LShr(addr, shift)
                                    if eval_const(shift, 0, value_base, const_int, inst_base, nodes) == Some(5) =>
                                {
                                    if let Some((slot, byte_off)) = eval_table_bytes(
                                        addr, 0, user_data, value_base, const_int, inst_base, nodes,
                                    ) && (byte_off as i32) >= 0
                                    {
                                        out.push(StageHeapElem::Table {
                                            slot,
                                            offset: byte_off >> 5,
                                            writes,
                                        });
                                        true
                                    } else {
                                        false
                                    }
                                }
                                Node::Select(t, f) => {
                                    resolve_index(t, writes, user_data, value_base, const_int, inst_base, nodes, phi_args, visiting, seen, out)
                                        && resolve_index(f, writes, user_data, value_base, const_int, inst_base, nodes, phi_args, visiting, seen, out)
                                }
                                Node::Phi(start, len) => {
                                    if len == 0 || visiting.contains(&v) {
                                        return false;
                                    }
                                    visiting.push(v);
                                    let args = &phi_args[start as usize..start as usize + usize::from(len)];
                                    let all_ok = args.iter().all(|&arg| {
                                        resolve_index(arg, writes, user_data, value_base, const_int, inst_base, nodes, phi_args, visiting, seen, out)
                                    });
                                    visiting.pop();
                                    all_ok
                                }
                                _ => false,
                            };
                            if ok {
                                seen.push(v);
                            }
                            ok
                        }

                        let mut visiting = Vec::new();
                        let mut seen = Vec::new();
                        let mut local = Vec::new();
                        for &(_h_id, mut idx_id, annotated, mut writes) in &heap_handles {
                            if !annotated {
                                writes = true;
                            }
                            if let Some(Node::Select(t, f)) = node_of(idx_id)
                                && eval_const(t, 0, value_base, &const_int, inst_base, &nodes)
                                    == Some(HEAP_VIEWS - 1)
                            {
                                idx_id = f;
                            }
                            visiting.clear();
                            seen.clear();
                            local.clear();
                            if resolve_index(
                                idx_id,
                                writes,
                                user_data,
                                value_base,
                                &const_int,
                                inst_base,
                                &nodes,
                                &phi_args,
                                &mut visiting,
                                &mut seen,
                                &mut local,
                            ) {
                                stage_bounded.extend_from_slice(&local);
                            } else {
                                stage_unbounded = true;
                                stage_unbounded_writes |= writes;
                            }
                        }
                    }
                    _ => r.bit = sub_end,
                }
            }
            _ => return None,
        }
    }
    dedup_stage_elems(&mut stage_bounded);
    Some(DxilInfo {
        typed_buffers: has_buf_load_store || (has_get_dims && has_typed_props),
        heap_access: StageHeapAccess {
            unbounded: stage_unbounded,
            unbounded_writes: stage_unbounded_writes,
            bounded: stage_bounded,
        },
    })
}

/// A compute pipeline and the group size its shader was written with (`numthreads`).
pub struct ComputePso<B: GraphicsBackend> {
    pub pso: B::Pipeline,
    pub threads: [usize; 3],
    pub binding: Option<Binding>,
}

/// One root parameter as the title set it.
pub enum Root<B: GraphicsBackend> {
    Constants(Vec<u32>),
    /// A root SRV, UAV or CBV: a title address.
    View(u64),
    /// A descriptor table: heap and the index of its first entry.
    Table(Rc<NativeHeap<B>>, usize),
}

impl<B: GraphicsBackend> Clone for Root<B> {
    fn clone(&self) -> Self {
        match self {
            Root::Constants(c) => Root::Constants(c.clone()),
            Root::View(a) => Root::View(*a),
            Root::Table(h, i) => Root::Table(h.clone(), *i),
        }
    }
}

type Heaps<B> = (Option<Rc<NativeHeap<B>>>, Option<Rc<NativeHeap<B>>>);

pub struct Targets<B: GraphicsBackend> {
    pub color: Vec<Rc<B::Texture>>,
    pub depth: Option<Rc<B::Texture>>,
}

impl<B: GraphicsBackend> Clone for Targets<B> {
    fn clone(&self) -> Self {
        Targets {
            color: self.color.clone(),
            depth: self.depth.clone(),
        }
    }
}

impl<B: GraphicsBackend> Targets<B> {
    pub(super) fn same(&self, o: &Targets<B>) -> bool {
        self.color.len() == o.color.len()
            && self
                .color
                .iter()
                .zip(&o.color)
                .all(|(a, b)| Rc::ptr_eq(a, b))
            && match (&self.depth, &o.depth) {
                (Some(a), Some(b)) => Rc::ptr_eq(a, b),
                (None, None) => true,
                _ => false,
            }
    }
}

/// Target and independent depth/stencil clear values (`None` preserves a plane).
type DepthClear<T> = (Rc<T>, Option<f32>, Option<u8>);

pub enum Cmd<B: GraphicsBackend> {
    Dispatch {
        pso: Rc<ComputePso<B>>,
        root: Vec<Option<Root<B>>>,
        heaps: Heaps<B>,
        groups: [usize; 3],
        /// The thread counts of a dispatch that gives them (`DispatchX` with thread dimensions),
        /// for the pipeline's lanes to start EXEC by; `u32::MAX` for any other dispatch.
        threads: [u32; 3],
        /// `ExecuteIndirect`: the title address of the group counts, read on the GPU.
        indirect: Option<u64>,
        /// `ExecuteIndirect` with a count buffer: its title address and this record's index.
        count: Option<(u64, u32)>,
        /// `ExecuteIndirect`: `(stride, n)`, this record and the `n - 1` that follow it `stride`
        /// bytes apart, each run as its own dispatch (count index one more each); `(0, 1)` else.
        records: (u32, u32),
    },
    Copy {
        dst: u64,
        src: u64,
        len: usize,
    },
    Fill {
        dst: u64,
        len: usize,
        value: u8,
    },
    CopyTexture {
        src: Rc<B::Texture>,
        src_mip: usize,
        src_slice: usize,
        src_origin: [usize; 3],
        size: [usize; 3],
        dst: Rc<B::Texture>,
        dst_mip: usize,
        dst_slice: usize,
        dst_origin: [usize; 3],
    },
    CopyBufferToTexture {
        src: u64,
        len: usize,
        row_pitch: usize,
        slice_pitch: usize,
        size: [usize; 3],
        dst: Rc<B::Texture>,
        dst_mip: usize,
        dst_slice: usize,
        dst_origin: [usize; 3],
    },
    CopyTextureToBuffer {
        src: Rc<B::Texture>,
        src_mip: usize,
        src_slice: usize,
        src_origin: [usize; 3],
        size: [usize; 3],
        dst: u64,
        len: usize,
        row_pitch: usize,
        slice_pitch: usize,
    },
    Clear {
        color: Option<(Rc<B::Texture>, [f32; 4])>,
        depth: Option<DepthClear<B::Texture>>,
        rects: Option<Rc<[ClearRect]>>,
    },
    Draw {
        pso: Rc<B::RenderPipeline>,
        binding: Option<Rc<Binding>>,
        topology: Topology,
        root: Vec<Option<Root<B>>>,
        heaps: Heaps<B>,
        targets: Targets<B>,
        vertex: Vec<(usize, u64)>,
        index: Option<(u64, bool)>,
        state: DrawState,
        call: DrawCall,
        /// `ExecuteIndirect`: the title address of the argument record, read on the GPU (`call`
        /// gives only its kind).
        indirect: Option<u64>,
        /// `ExecuteIndirect` with a count buffer: its title address and this record's index.
        count: Option<(u64, u32)>,
        /// As [`Cmd::Dispatch`]'s: the records of one `ExecuteIndirect`, each its own draw.
        records: (u32, u32),
    },
}

impl<B: GraphicsBackend> Clone for Cmd<B> {
    fn clone(&self) -> Self {
        match self {
            Cmd::Dispatch {
                pso,
                root,
                heaps,
                groups,
                threads,
                indirect,
                count,
                records,
            } => Cmd::Dispatch {
                pso: pso.clone(),
                root: root.clone(),
                heaps: heaps.clone(),
                groups: *groups,
                threads: *threads,
                indirect: *indirect,
                count: *count,
                records: *records,
            },
            Cmd::Copy { dst, src, len } => Cmd::Copy {
                dst: *dst,
                src: *src,
                len: *len,
            },
            Cmd::Fill { dst, len, value } => Cmd::Fill {
                dst: *dst,
                len: *len,
                value: *value,
            },
            Cmd::CopyTexture {
                src,
                src_mip,
                src_slice,
                src_origin,
                size,
                dst,
                dst_mip,
                dst_slice,
                dst_origin,
            } => Cmd::CopyTexture {
                src: src.clone(),
                src_mip: *src_mip,
                src_slice: *src_slice,
                src_origin: *src_origin,
                size: *size,
                dst: dst.clone(),
                dst_mip: *dst_mip,
                dst_slice: *dst_slice,
                dst_origin: *dst_origin,
            },
            Cmd::CopyBufferToTexture {
                src,
                len,
                row_pitch,
                slice_pitch,
                size,
                dst,
                dst_mip,
                dst_slice,
                dst_origin,
            } => Cmd::CopyBufferToTexture {
                src: *src,
                len: *len,
                row_pitch: *row_pitch,
                slice_pitch: *slice_pitch,
                size: *size,
                dst: dst.clone(),
                dst_mip: *dst_mip,
                dst_slice: *dst_slice,
                dst_origin: *dst_origin,
            },
            Cmd::CopyTextureToBuffer {
                src,
                src_mip,
                src_slice,
                src_origin,
                size,
                dst,
                len,
                row_pitch,
                slice_pitch,
            } => Cmd::CopyTextureToBuffer {
                src: src.clone(),
                src_mip: *src_mip,
                src_slice: *src_slice,
                src_origin: *src_origin,
                size: *size,
                dst: *dst,
                len: *len,
                row_pitch: *row_pitch,
                slice_pitch: *slice_pitch,
            },
            Cmd::Clear { color, depth, rects } => Cmd::Clear {
                color: color.clone(),
                depth: depth.clone(),
                rects: rects.clone(),
            },
            Cmd::Draw {
                pso,
                binding,
                topology,
                root,
                heaps,
                targets,
                vertex,
                index,
                state,
                call,
                indirect,
                count,
                records,
            } => Cmd::Draw {
                pso: pso.clone(),
                binding: binding.clone(),
                topology: *topology,
                root: root.clone(),
                heaps: heaps.clone(),
                targets: targets.clone(),
                vertex: vertex.clone(),
                index: *index,
                state: *state,
                call: *call,
                indirect: *indirect,
                count: *count,
                records: *records,
            },
        }
    }
}

/// The state a title sets between commands.
struct State<B: GraphicsBackend> {
    compute: Option<Rc<ComputePso<B>>>,
    graphics: Option<(Rc<B::RenderPipeline>, Topology)>,
    graphics_binding: Option<Rc<Binding>>,
    compute_root: Vec<Option<Root<B>>>,
    graphics_root: Vec<Option<Root<B>>>,
    heaps: Heaps<B>,
    targets: Targets<B>,
    vertex: BTreeMap<usize, u64>,
    index: Option<(u64, bool)>,
    draw: DrawState,
}

pub struct NativeList<B: GraphicsBackend> {
    pub(super) cmds: Vec<Cmd<B>>,
    /// Pass-trace lines (`VICTORY_TRACE_PASSES`), each with the command count when it was recorded;
    /// empty unless the trace's window is open.
    pub(super) notes: Vec<(usize, String)>,
    state: State<B>,
}

fn set<B: GraphicsBackend>(root: &mut Vec<Option<Root<B>>>, slot: usize, value: Root<B>) {
    if root.len() <= slot {
        root.resize_with(slot + 1, || None);
    }
    root[slot] = Some(value);
}

impl<B: GraphicsBackend> Default for NativeList<B> {
    fn default() -> Self {
        NativeList {
            cmds: Vec::new(),
            notes: Vec::new(),
            state: State {
                compute: None,
                graphics: None,
                graphics_binding: None,
                compute_root: Vec::new(),
                graphics_root: Vec::new(),
                heaps: (None, None),
                targets: Targets {
                    color: Vec::new(),
                    depth: None,
                },
                vertex: BTreeMap::new(),
                index: None,
                draw: DrawState::default(),
            },
        }
    }
}

impl<B: GraphicsBackend> NativeList<B> {
    /// The graphics root parameters and binding the next draw takes (for traces).
    #[cfg(target_os = "macos")]
    pub(super) fn graphics_args(&self) -> (&[Option<Root<B>>], Option<&Binding>) {
        (&self.state.graphics_root, self.state.graphics_binding.as_deref())
    }

    /// The compute root parameters the next dispatch takes (for traces).
    #[cfg(target_os = "macos")]
    pub(super) fn compute_args(&self) -> &[Option<Root<B>>] {
        &self.state.compute_root
    }

    /// `Reset`: forget everything recorded and set.
    pub fn reset(&mut self) {
        *self = NativeList::default();
    }

    pub fn is_empty(&self) -> bool {
        self.cmds.is_empty()
    }

    /// What was recorded so far as a list of its own, leaving this one empty with its state kept
    /// (the title goes on recording after a wait with the pipeline and root arguments it had set).
    pub fn cut(&mut self) -> NativeList<B> {
        NativeList {
            cmds: std::mem::take(&mut self.cmds),
            notes: std::mem::take(&mut self.notes),
            ..NativeList::default()
        }
    }

    /// A copy of the recorded commands (a title may reset and re-record a list it submitted).
    pub fn snapshot(&self) -> NativeList<B> {
        NativeList {
            cmds: self.cmds.clone(),
            notes: self.notes.clone(),
            ..NativeList::default()
        }
    }

    /// A write of `value`'s low `bytes` (little endian) at title address `at`, in order with the
    /// rest of the list on the GPU (bottom-of-pipe value writes): byte fills.
    pub fn write_value(&mut self, at: u64, value: u64, bytes: usize) {
        for (i, b) in value.to_le_bytes().iter().take(bytes).enumerate() {
            self.fill(at + i as u64, 1, *b);
        }
    }

    /// `SetDescriptorHeaps`.
    pub fn set_heaps(
        &mut self,
        resource: Option<&Rc<NativeHeap<B>>>,
        sampler: Option<&Rc<NativeHeap<B>>>,
    ) {
        self.state.heaps = (resource.cloned(), sampler.cloned());
    }

    pub fn set_compute_pipeline(&mut self, pso: &Rc<ComputePso<B>>) {
        self.state.compute = Some(pso.clone());
    }

    pub fn set_graphics_pipeline(&mut self, pso: &Rc<B::RenderPipeline>, topology: Topology) {
        self.state.graphics = Some((pso.clone(), topology));
        self.state.graphics_binding = None;
    }

    /// The argument layout of the graphics pipeline set last (a translated title shader's).
    pub fn set_graphics_binding(&mut self, binding: Option<Rc<Binding>>) {
        self.state.graphics_binding = binding;
    }

    /// `SetComputeRoot32BitConstants` (the whole parameter).
    pub fn set_compute_root_constants(&mut self, slot: usize, words: &[u32]) {
        set(
            &mut self.state.compute_root,
            slot,
            Root::Constants(words.to_vec()),
        );
    }

    /// `SetComputeRoot32BitConstant(s)` / `SetGraphicsRoot32BitConstant(s)` at `offset` words into
    /// the parameter: the words before it keep what was set (zero if nothing was).
    pub fn set_root_constants_at(&mut self, compute: bool, slot: usize, offset: usize, words: &[u32]) {
        let root = if compute {
            &mut self.state.compute_root
        } else {
            &mut self.state.graphics_root
        };
        let mut all = match root.get(slot) {
            Some(Some(Root::Constants(c))) => c.clone(),
            _ => Vec::new(),
        };
        if all.len() < offset + words.len() {
            all.resize(offset + words.len(), 0);
        }
        all[offset..offset + words.len()].copy_from_slice(words);
        set(root, slot, Root::Constants(all));
    }

    /// `SetComputeRootShaderResourceView` / `UnorderedAccessView` / `ConstantBufferView`.
    pub fn set_compute_root_view(&mut self, slot: usize, address: u64) {
        set(&mut self.state.compute_root, slot, Root::View(address));
    }

    /// `SetComputeRootDescriptorTable`.
    pub fn set_compute_root_table(&mut self, slot: usize, heap: &Rc<NativeHeap<B>>, index: usize) {
        set(
            &mut self.state.compute_root,
            slot,
            Root::Table(heap.clone(), index),
        );
    }

    pub fn set_graphics_root_constants(&mut self, slot: usize, words: &[u32]) {
        set(
            &mut self.state.graphics_root,
            slot,
            Root::Constants(words.to_vec()),
        );
    }

    pub fn set_graphics_root_view(&mut self, slot: usize, address: u64) {
        set(&mut self.state.graphics_root, slot, Root::View(address));
    }

    pub fn set_graphics_root_table(&mut self, slot: usize, heap: &Rc<NativeHeap<B>>, index: usize) {
        set(
            &mut self.state.graphics_root,
            slot,
            Root::Table(heap.clone(), index),
        );
    }

    /// `Dispatch`. Needs the compute pipeline set.
    pub fn dispatch(&mut self, x: u32, y: u32, z: u32) -> Result<(), String> {
        self.dispatch_from([x as usize, y as usize, z as usize], [u32::MAX; 3], None, None)
    }

    /// `DispatchX` with thread counts, of a pipeline translated for them
    /// (`victory_shader::emit::Options::thread_counts`): whole groups, rounded up, whose threads
    /// past the counts start with EXEC off.
    pub fn dispatch_threads(&mut self, threads: [u32; 3]) -> Result<(), String> {
        let per = self.state.compute.as_ref().ok_or("dispatch without a compute pipeline")?.threads;
        let groups = std::array::from_fn(|d| (threads[d] as usize).div_ceil(per[d]));
        self.dispatch_from(groups, threads, None, None)
    }

    /// One dispatch of `ExecuteIndirect`: its group counts are the three words at title address
    /// `record`, read on the GPU.
    pub fn dispatch_indirect(&mut self, record: u64) -> Result<(), String> {
        self.dispatch_from([0; 3], [u32::MAX; 3], Some(record), None)
    }

    /// Record `index` of an `ExecuteIndirect` of dispatches with a count buffer at title address
    /// `count`: it runs only if `index` is below the count, read on the GPU when it runs.
    pub fn dispatch_indirect_counted(
        &mut self,
        record: u64,
        count: u64,
        index: u32,
    ) -> Result<(), String> {
        self.dispatch_from([0; 3], [u32::MAX; 3], Some(record), Some((count, index)))
    }

    fn dispatch_from(
        &mut self,
        groups: [usize; 3],
        threads: [u32; 3],
        indirect: Option<u64>,
        count: Option<(u64, u32)>,
    ) -> Result<(), String> {
        let pso = self
            .state
            .compute
            .clone()
            .ok_or("dispatch without a compute pipeline")?;
        if indirect.is_none() && groups.contains(&0) {
            return Ok(());
        }
        self.cmds.push(Cmd::Dispatch {
            pso,
            root: self.state.compute_root.clone(),
            heaps: self.state.heaps.clone(),
            groups,
            threads,
            indirect,
            count,
            records: (0, 1),
        });
        Ok(())
    }

    /// `CopyBufferRegion` between title addresses. Runs on the GPU, in order with the rest.
    pub fn copy_buffer_region(&mut self, dst: u64, src: u64, len: usize) {
        if len > 0 {
            self.cmds.push(Cmd::Copy { dst, src, len });
        }
    }

    /// A byte fill of title memory (the console's `FillMemoryX`).
    pub fn fill(&mut self, dst: u64, len: usize, value: u8) {
        if len > 0 {
            self.cmds.push(Cmd::Fill { dst, len, value });
        }
    }

    /// Texture-to-texture region copy, ordered on the GPU with the rest of the list.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_texture_region(
        &mut self,
        src: &Rc<B::Texture>,
        src_mip: usize,
        src_slice: usize,
        src_origin: [usize; 3],
        size: [usize; 3],
        dst: &Rc<B::Texture>,
        dst_mip: usize,
        dst_slice: usize,
        dst_origin: [usize; 3],
    ) {
        if size.contains(&0) {
            return;
        }
        self.cmds.push(Cmd::CopyTexture {
            src: src.clone(),
            src_mip,
            src_slice,
            src_origin,
            size,
            dst: dst.clone(),
            dst_mip,
            dst_slice,
            dst_origin,
        });
    }

    /// Linear buffer footprint to texture region copy, ordered on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_buffer_to_texture(
        &mut self,
        src: u64,
        len: usize,
        row_pitch: usize,
        slice_pitch: usize,
        size: [usize; 3],
        dst: &Rc<B::Texture>,
        dst_mip: usize,
        dst_slice: usize,
        dst_origin: [usize; 3],
    ) {
        if len == 0 || size.contains(&0) {
            return;
        }
        self.cmds.push(Cmd::CopyBufferToTexture {
            src,
            len,
            row_pitch,
            slice_pitch,
            size,
            dst: dst.clone(),
            dst_mip,
            dst_slice,
            dst_origin,
        });
    }

    /// Texture region to linear buffer footprint copy, ordered on the GPU.
    #[allow(clippy::too_many_arguments)]
    pub fn copy_texture_to_buffer(
        &mut self,
        src: &Rc<B::Texture>,
        src_mip: usize,
        src_slice: usize,
        src_origin: [usize; 3],
        size: [usize; 3],
        dst: u64,
        len: usize,
        row_pitch: usize,
        slice_pitch: usize,
    ) {
        if len == 0 || size.contains(&0) {
            return;
        }
        self.cmds.push(Cmd::CopyTextureToBuffer {
            src: src.clone(),
            src_mip,
            src_slice,
            src_origin,
            size,
            dst,
            len,
            row_pitch,
            slice_pitch,
        });
    }

    /// `OMSetRenderTargets`.
    pub fn set_render_targets(&mut self, color: &[Rc<B::Texture>], depth: Option<&Rc<B::Texture>>) {
        self.state.targets = Targets {
            color: color.to_vec(),
            depth: depth.cloned(),
        };
    }

    #[cfg(all(test, target_os = "macos"))]
    pub(super) fn color_targets(&self) -> &[Rc<B::Texture>] {
        &self.state.targets.color
    }

    /// `ClearRenderTargetView`. shortcut: each clear is a render pass of its own (load action
    /// clear); folding it into the following draws' pass saves one pass per clear.
    pub fn clear_render_target(&mut self, target: &Rc<B::Texture>, color: [f32; 4]) {
        self.cmds.push(Cmd::Clear {
            color: Some((target.clone(), color)),
            depth: None,
            rects: None,
        });
    }

    pub fn clear_render_target_rects(
        &mut self,
        target: &Rc<B::Texture>,
        color: [f32; 4],
        rects: &[ClearRect],
    ) {
        self.cmds.push(Cmd::Clear {
            color: Some((target.clone(), color)),
            depth: None,
            rects: Some(Rc::from(rects)),
        });
    }

    pub fn clear_depth(&mut self, target: &Rc<B::Texture>, depth: f32) {
        self.clear_depth_stencil(target, Some(depth), None);
    }

    /// Clear only the requested planes; absent values preserve their contents.
    pub fn clear_depth_stencil(
        &mut self,
        target: &Rc<B::Texture>,
        depth: Option<f32>,
        stencil: Option<u8>,
    ) {
        if depth.is_some() || stencil.is_some() {
            self.cmds.push(Cmd::Clear {
                color: None,
                depth: Some((target.clone(), depth, stencil)),
                rects: None,
            });
        }
    }

    /// `RSSetViewports` (the first viewport; the host draws into one).
    pub fn set_viewport(&mut self, viewport: Option<Viewport>) {
        self.state.draw.viewport = viewport;
    }

    /// `RSSetScissorRects` (the first rectangle): left, top, right, bottom.
    pub fn set_scissor(&mut self, rect: Option<[i32; 4]>) {
        self.state.draw.scissor = rect;
    }

    /// `OMSetStencilRef`.
    pub fn set_stencil_ref(&mut self, reference: u32) {
        self.state.draw.stencil_ref = reference;
    }

    /// `OMSetBlendFactor`.
    pub fn set_blend_factor(&mut self, factor: [f32; 4]) {
        self.state.draw.blend_factor = factor;
    }

    /// `IASetVertexBuffers` for one slot: the title address of its data (the stride is the pipeline's).
    pub fn set_vertex_buffer(&mut self, slot: usize, address: u64) {
        if address == 0 {
            self.state.vertex.remove(&slot);
        } else {
            self.state.vertex.insert(slot, address);
        }
    }

    /// The address bound as vertex buffer `slot`, if any (for the target trace).
    pub fn vertex_buffer_address(&self, slot: usize) -> Option<u64> {
        self.state.vertex.get(&slot).copied()
    }

    pub fn set_index_buffer(&mut self, address: u64, wide: bool) {
        self.state.index = Some((address, wide));
    }

    pub fn clear_index_buffer(&mut self) {
        self.state.index = None;
    }

    fn draw_call(&mut self, call: DrawCall) -> Result<(), String> {
        let empty = match call {
            DrawCall::Arrays { vertex_count, instance_count, .. } => {
                vertex_count == 0 || instance_count == 0
            }
            DrawCall::Indexed { index_count, instance_count, .. } => {
                index_count == 0 || instance_count == 0
            }
            DrawCall::Mesh { groups } => groups.contains(&0),
            DrawCall::Rects => false,
        };
        if empty {
            self.state
                .graphics
                .as_ref()
                .ok_or("draw without a graphics pipeline")?;
            return Ok(());
        }
        self.draw_from(call, None, None)
    }

    /// One draw of `ExecuteIndirect`: its counts are the record at title address `record`
    /// (`D3D12_DRAW_INDEXED_ARGUMENTS` when `indexed`, else `D3D12_DRAW_ARGUMENTS`), read on the GPU.
    pub fn draw_indirect(&mut self, record: u64, indexed: bool) -> Result<(), String> {
        self.draw_indirect_from(record, indexed, None)
    }

    /// Record `index` of an `ExecuteIndirect` of draws with a count buffer at title address
    /// `count`: it draws only if `index` is below the count, read on the GPU when it runs.
    pub fn draw_indirect_counted(
        &mut self,
        record: u64,
        indexed: bool,
        count: u64,
        index: u32,
    ) -> Result<(), String> {
        self.draw_indirect_from(record, indexed, Some((count, index)))
    }

    /// One rectangle-list draw of `ExecuteIndirect` (`DrawCall::Rects`): the record at title
    /// address `record`, with `count` as [`Self::draw_indirect_counted`]'s.
    pub fn draw_rects_indirect(&mut self, record: u64, count: Option<(u64, u32)>) -> Result<(), String> {
        self.draw_from(DrawCall::Rects, Some(record), count)
    }

    fn draw_indirect_from(
        &mut self,
        record: u64,
        indexed: bool,
        count: Option<(u64, u32)>,
    ) -> Result<(), String> {
        let call = if indexed {
            DrawCall::Indexed {
                index_count: 0,
                instance_count: 0,
                first_index: 0,
                base_vertex: 0,
                first_instance: 0,
                wide: self
                    .state
                    .index
                    .ok_or("indexed draw without an index buffer")?
                    .1,
            }
        } else {
            DrawCall::Arrays {
                vertex_count: 0,
                instance_count: 0,
                first_vertex: 0,
                first_instance: 0,
            }
        };
        self.draw_from(call, Some(record), count)
    }

    fn draw_from(
        &mut self,
        call: DrawCall,
        indirect: Option<u64>,
        count: Option<(u64, u32)>,
    ) -> Result<(), String> {
        let (pso, topology) = self
            .state
            .graphics
            .clone()
            .ok_or("draw without a graphics pipeline")?;
        self.cmds.push(Cmd::Draw {
            pso,
            binding: self.state.graphics_binding.clone(),
            topology,
            root: self.state.graphics_root.clone(),
            heaps: self.state.heaps.clone(),
            targets: self.state.targets.clone(),
            vertex: self.state.vertex.iter().map(|(&slot, &addr)| (slot, addr)).collect(),
            index: self.state.index,
            state: self.state.draw,
            call,
            indirect,
            count,
            records: (0, 1),
        });
        Ok(())
    }

    /// The `n` records of one `ExecuteIndirect`, at title address `first` and `stride` bytes apart,
    /// as one command: `record` records the first (with count index 0), and the command is
    /// widened to all `n`. The host prepares it once and runs each record as its own draw or
    /// dispatch, as `n` calls of `record` at each record (index `i`) would.
    pub fn indirect_records(
        &mut self,
        first: u64,
        stride: u32,
        n: u32,
        count: Option<u64>,
        record: impl FnOnce(&mut Self, u64, Option<(u64, u32)>) -> Result<(), String>,
    ) -> Result<(), String> {
        if n == 0 {
            return Ok(());
        }
        let before = self.cmds.len();
        record(self, first, count.map(|c| (c, 0)))?;
        if self.cmds.len() > before
            && let Some(Cmd::Draw { records, .. } | Cmd::Dispatch { records, .. }) = self.cmds.last_mut()
        {
            *records = (stride, n);
        }
        Ok(())
    }

    pub fn draw_instanced(
        &mut self,
        vertices: u32,
        instances: u32,
        first_vertex: u32,
        first_instance: u32,
    ) -> Result<(), String> {
        self.draw_call(DrawCall::Arrays {
            vertex_count: vertices,
            instance_count: instances,
            first_vertex,
            first_instance,
        })
    }

    /// `DispatchMesh` of a mesh pipeline.
    pub fn dispatch_mesh(&mut self, x: u32, y: u32, z: u32) -> Result<(), String> {
        self.draw_call(DrawCall::Mesh { groups: [x, y, z] })
    }

    /// One `DispatchMesh` of `ExecuteIndirect`: its group counts are the record at title address
    /// `record`, read on the GPU; with `count`, only if the record's index is below the count.
    pub fn dispatch_mesh_indirect(&mut self, record: u64, count: Option<(u64, u32)>) -> Result<(), String> {
        self.draw_from(DrawCall::Mesh { groups: [0; 3] }, Some(record), count)
    }

    pub fn draw_indexed_instanced(
        &mut self,
        indices: u32,
        instances: u32,
        first_index: u32,
        base_vertex: i32,
        first_instance: u32,
    ) -> Result<(), String> {
        let wide = self
            .state
            .index
            .ok_or("indexed draw without an index buffer")?
            .1;
        self.draw_call(DrawCall::Indexed {
            index_count: indices,
            instance_count: instances,
            first_index,
            base_vertex,
            first_instance,
            wide,
        })
    }
}
