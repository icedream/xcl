//! Fence #12 of the newer console layout (a retail game): the address of the fence's completed value,
//! which the title reads and sleeps on (`monitorx` / `mwaitx`). On the console the GPU writes
//! that memory; a PC fence's value is only behind GetCompletedValue, so each fence asked gets a
//! cell of its own that a thread keeps equal to it: every Signal the title queues arms an event.

use std::collections::HashMap;
use std::ffi::c_void;
use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Mutex, OnceLock};
use std::time::{Duration, Instant};
use windows::core::Interface;
use windows::Win32::Foundation::HANDLE;
use windows::Win32::Graphics::Direct3D12::ID3D12Fence;
use windows_sys::Win32::System::Threading as th;

/// a cache line of its own: the title's monitors wake on writes to the line
#[repr(C, align(64))]
struct Cell(AtomicU64);

#[derive(Default)]
struct Fence {
    cell: Option<&'static Cell>,
    /// the highest value a queue was told to signal
    signalled: u64,
    /// the highest value a queue was told to wait for
    waited: u64,
}

/// (fences are never freed: XCL holds a reference to each)
static FENCES: Mutex<Option<HashMap<usize, Fence>>> = Mutex::new(None);

/// the event a completion sets (auto-reset); the first use starts the thread that follows it
fn event() -> usize {
    static E: OnceLock<usize> = OnceLock::new();
    *E.get_or_init(|| unsafe {
        let e = th::CreateEventW(std::ptr::null(), 0, 0, std::ptr::null()) as usize;
        let spawned = std::thread::Builder::new().name("xcl-fences".into()).spawn(move || loop {
            th::WaitForSingleObject(e as _, th::INFINITE);
            refresh();
        });
        if let Err(err) = spawned {
            crate::xlog_error!("D3D12.X: no thread for the fences' completed values: {err}");
        }
        e
    })
}

unsafe fn borrowed<R>(fence: usize, f: impl FnOnce(&ID3D12Fence) -> R) -> Option<R> {
    let p = fence as *mut c_void;
    ID3D12Fence::from_raw_borrowed(&p).map(f)
}

fn refresh() {
    let g = FENCES.lock().unwrap_or_else(|e| e.into_inner());
    for (&p, f) in g.iter().flatten() {
        if let Some(c) = f.cell {
            if let Some(v) = unsafe { borrowed(p, |f| f.GetCompletedValue()) } {
                c.0.store(v, Ordering::Release);
            }
        }
    }
}

/// How long an armed `mwaitx` wait may sit unmet before it is logged once.
const ARMED_WAIT_WARN: Duration = Duration::from_secs(2);

/// A wait armed on a fence cell: `arm` records it when a queue is told to signal the value, the
/// `mwaitx` emulation reports it when the cell has not advanced for `ARMED_WAIT_WARN`.
struct ArmedWait {
    fence: usize,
    cell: &'static Cell,
    /// the value the title waits for (the highest a queue was told to signal)
    wanted: u64,
    armed_at: Instant,
    warned: bool,
}

/// Armed waits, newest last; satisfied ones are dropped, the rest capped so a title arming
/// ever-higher values on a never-completing fence cannot grow it without bound.
static ARMED_WAITS: Mutex<Vec<ArmedWait>> = Mutex::new(Vec::new());
/// Armed waits logged as unmet for over `ARMED_WAIT_WARN` (the metrics line).
static LONG_WAITS: AtomicU64 = AtomicU64::new(0);

/// Armed waits logged as unmet for over 2 s (the metrics line).
pub(super) fn long_waits() -> u64 {
    LONG_WAITS.load(Ordering::Relaxed)
}

unsafe fn arm(fence: usize, value: u64) {
    if let Some(Err(e)) = borrowed(fence, |f| f.SetEventOnCompletion(value, HANDLE(event() as *mut c_void))) {
        crate::xlog_warn!("D3D12.X: fence {fence:#x}: SetEventOnCompletion({value:#x}): {e}");
    }
    // record the armed wait: the title's `mwaitx` on the cell reports it if the value never comes
    let cell = FENCES.lock().unwrap_or_else(|e| e.into_inner()).as_ref().and_then(|m| m.get(&fence)).and_then(|f| f.cell);
    if let Some(cell) = cell {
        let mut g = ARMED_WAITS.lock().unwrap_or_else(|e| e.into_inner());
        g.retain(|w| w.cell.0.load(Ordering::Acquire) < w.wanted);
        match g.iter_mut().find(|w| w.fence == fence && w.wanted == value) {
            Some(w) => w.armed_at = Instant::now(),
            None => {
                g.push(ArmedWait { fence, cell, wanted: value, armed_at: Instant::now(), warned: false });
                let excess = g.len().saturating_sub(64);
                if excess > 0 {
                    g.drain(..excess);
                }
            }
        }
    }
}

/// The title's `mwaitx` monitors `addr` (RAX): report an armed wait on that cell whose value has
/// not advanced for `ARMED_WAIT_WARN`, once per armed wait. Called from the instruction
/// emulation on every `mwaitx`, so it returns at once when nothing is armed.
pub(crate) fn note_mwaitx(addr: u64) {
    let mut g = ARMED_WAITS.lock().unwrap_or_else(|e| e.into_inner());
    if g.is_empty() {
        return;
    }
    let now = Instant::now();
    let mut i = 0;
    while i < g.len() {
        if g[i].cell.0.as_ptr() as u64 != addr {
            i += 1;
            continue;
        }
        let (fence, wanted, cur) = (g[i].fence, g[i].wanted, g[i].cell.0.load(Ordering::Acquire));
        if cur >= wanted {
            g.remove(i); // the wait is met: stop watching it
            continue;
        }
        let waited = now.duration_since(g[i].armed_at);
        if !g[i].warned && waited >= ARMED_WAIT_WARN {
            g[i].warned = true;
            LONG_WAITS.fetch_add(1, Ordering::Relaxed);
            let done = unsafe { borrowed(fence, |f| f.GetCompletedValue()) }.unwrap_or(cur);
            crate::xlog_warn!(
                "D3D12.X: mwaitx on {addr:#x} waits for {wanted:#x} and the cell has not advanced for {:.1} s (cell {cur:#x}, fence completed {done:#x})",
                waited.as_secs_f64()
            );
        }
        i += 1;
    }
}

/// A queue was told to signal `fence` (one of the title's) with `value`.
pub(super) unsafe fn note_queue_signal(fence: usize, value: u64) {
    let cell = {
        let mut g = FENCES.lock().unwrap_or_else(|e| e.into_inner());
        let f = g.get_or_insert_with(HashMap::new).entry(fence).or_default();
        f.signalled = f.signalled.max(value);
        f.cell.is_some()
    };
    if cell {
        arm(fence, value);
    }
}

/// A queue was told to wait for `fence` (one of the title's) to reach `value`.
pub(super) fn note_queue_wait(fence: usize, value: u64) {
    let mut g = FENCES.lock().unwrap_or_else(|e| e.into_inner());
    let f = g.get_or_insert_with(HashMap::new).entry(fence).or_default();
    f.waited = f.waited.max(value);
}

/// The CPU signalled `fence`.
pub(super) fn note_cpu_signal(fence: usize) {
    let known = FENCES.lock().unwrap_or_else(|e| e.into_inner()).as_ref().is_some_and(|m| m.get(&fence).is_some_and(|f| f.cell.is_some()));
    if known {
        refresh();
    }
}

/// Diagnostic summary of title fences whose completed value lags their signalled or waited value.
pub(super) fn pending_summary() -> String {
    let g = FENCES.lock().unwrap_or_else(|e| e.into_inner());
    let mut parts = Vec::new();
    for (&p, f) in g.iter().flatten() {
        let done = unsafe { borrowed(p, |f| f.GetCompletedValue()) }.unwrap_or(0);
        if done < f.signalled || done < f.waited {
            parts.push(format!("{p:#x}(done={done:#x}/sig={:#x}/wait={:#x},cell={})", f.signalled, f.waited, f.cell.is_some()));
        }
    }
    if parts.is_empty() {
        "none".into()
    } else {
        parts.join(", ")
    }
}

/// fence #12: the address of its completed value
pub(super) unsafe extern "system" fn completed_value_address(this: *mut c_void) -> *const u64 {
    let (cell, pending) = {
        let mut g = FENCES.lock().unwrap_or_else(|e| e.into_inner());
        let f = g.get_or_insert_with(HashMap::new).entry(this as usize).or_default();
        match f.cell {
            Some(c) => return c.0.as_ptr(),
            None => {
                let c: &'static Cell = Box::leak(Box::new(Cell(AtomicU64::new(0))));
                f.cell = Some(c);
                (c, f.signalled)
            }
        }
    };
    if pending > 0 {
        arm(this as usize, pending);
    }
    refresh();
    cell.0.as_ptr()
}

/// Remove a synthetic test fence from `FENCES` before it is dropped.
#[cfg(test)]
pub(super) fn forget_fence_for_test(fence: usize) {
    if let Some(m) = FENCES.lock().unwrap_or_else(|e| e.into_inner()).as_mut() {
        m.remove(&fence);
    }
}

#[cfg(test)]
mod tests {
    use super::*;
    use windows::Win32::Graphics::Direct3D::D3D_FEATURE_LEVEL_11_0;
    use windows::Win32::Graphics::Direct3D12::*;
    use windows::Win32::Graphics::Dxgi::*;

    /// The armed-wait watch list: an armed wait on a cell is reported once when it has not been
    /// met for 2 s, and dropped once the cell reaches the wanted value (counted per cell, so the
    /// GPU test's own armed waits cannot disturb the assertions).
    #[test]
    fn armed_waits_are_watched_per_cell_and_dropped_when_met() {
        let c: &'static Cell = Box::leak(Box::new(Cell(AtomicU64::new(0))));
        let addr = c.0.as_ptr() as u64;
        {
            let mut g = ARMED_WAITS.lock().unwrap();
            g.push(ArmedWait { fence: 1, cell: c, wanted: 7, armed_at: Instant::now() - Duration::from_secs(3), warned: true });
            g.push(ArmedWait { fence: 2, cell: c, wanted: 3, armed_at: Instant::now(), warned: false });
        }
        let mine = |g: &Vec<ArmedWait>| g.iter().filter(|w| w.cell.0.as_ptr() as u64 == addr).map(|w| w.wanted).collect::<Vec<_>>();
        note_mwaitx(addr);
        assert_eq!(mine(&ARMED_WAITS.lock().unwrap()), vec![7, 3], "nothing met, nothing newly warned (the 3-wait is under 2 s old)");
        c.0.store(3, Ordering::Release);
        note_mwaitx(addr);
        assert_eq!(mine(&ARMED_WAITS.lock().unwrap()), vec![7], "the met wait is dropped");
        c.0.store(7, Ordering::Release);
        note_mwaitx(addr);
        assert_eq!(mine(&ARMED_WAITS.lock().unwrap()), Vec::<u64>::new());
    }

    /// The cell follows the fence: CPU signals at once, queue signals when the GPU is there –
    /// also one queued before the address was asked for.
    #[test]
    fn the_cell_follows_the_fence() {
        unsafe {
            let mut dev: Option<ID3D12Device> = None;
            if D3D12CreateDevice(None, D3D_FEATURE_LEVEL_11_0, &mut dev).is_err() {
                let factory: IDXGIFactory4 = CreateDXGIFactory1().unwrap();
                let warp: IDXGIAdapter = factory.EnumWarpAdapter().unwrap();
                D3D12CreateDevice(&warp, D3D_FEATURE_LEVEL_11_0, &mut dev).expect("WARP D3D12 device");
            }
            let dev = dev.unwrap();
            let queue: ID3D12CommandQueue = dev.CreateCommandQueue(&D3D12_COMMAND_QUEUE_DESC { Type: D3D12_COMMAND_LIST_TYPE_DIRECT, ..Default::default() }).unwrap();
            let fence: ID3D12Fence = dev.CreateFence(5, D3D12_FENCE_FLAG_NONE).unwrap();
            let p = fence.as_raw();
            let wait = |cell: *const u64, v: u64| {
                let t = std::time::Instant::now();
                while (*(cell as *const AtomicU64)).load(Ordering::Acquire) != v {
                    assert!(t.elapsed().as_secs() < 5, "cell never became {v}");
                    std::thread::yield_now();
                }
            };
            queue.Signal(&fence, 6).unwrap();
            note_queue_signal(p as usize, 6);
            let cell = completed_value_address(p);
            assert_eq!(cell, completed_value_address(p));
            wait(cell, 6);
            queue.Signal(&fence, 7).unwrap();
            note_queue_signal(p as usize, 7);
            wait(cell, 7);
            fence.Signal(9).unwrap();
            note_cpu_signal(p as usize);
            assert_eq!(*cell, 9);
            std::mem::forget(fence); // the map keeps its address
        }
    }
}
