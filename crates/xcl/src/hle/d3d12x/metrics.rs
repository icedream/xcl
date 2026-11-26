//! What a run spent compiling pipelines and what that cost it in frames: one line every 60 s
//! (`host`), the same numbers in `report.json` (`d3d12.metrics`) and once at the exit with the
//! run's average frame rate. Always on and cheap: the counters are relaxed atomics, and the
//! present path takes no lock for them.
//!
//! A title that freezes while graphics pipelines compile: 28 gated driver compiles of
//! 262 904 – 299 996 B DXIL took 6.3 – 8.9 s each (218.9 s in the driver inside 162 s of wall,
//! 2 slots busy) and 2 frames were shown. The log had every compile and every frame in it, but
//! not that answer.

use std::sync::atomic::{AtomicU64, Ordering};
use std::sync::{Once, OnceLock};
use std::time::{Duration, Instant};

use serde::{Deserialize, Serialize};

/// The counters at one moment: `report.json`'s `d3d12.metrics` and the body of the log line.
#[derive(Debug, Default, Clone, Serialize, Deserialize)]
pub struct Metrics {
    /// driver compiles begun, returned (with the time they spent in the driver added up) and
    /// refused (a negative HRESULT)
    pub compiles_started: u64,
    pub compiles_done: u64,
    pub compiles_failed: u64,
    pub driver_ms: u64,
    /// compiles that waited for a huge-compile slot, and the time they waited for it
    pub gated_compiles: u64,
    pub slot_wait_ms: u64,
    /// most compiles in the driver at the same time
    pub compiles_peak: u64,
    /// frames shown, the longest time with none shown before one, and how many of the gaps
    /// between frames were over 1 s / over 10 s
    pub presents: u64,
    pub longest_present_gap_ms: u64,
    pub present_gaps_over_1s: u64,
    pub present_gaps_over_10s: u64,
    /// stand-ins handed to binds of a pipeline not built within their wait budget (`pso_async`
    /// `late`), and the draws / dispatches of console code those skipped
    pub standins: u64,
    pub skipped_draws: u64,
    pub skipped_dispatches: u64,
    /// since the counters began (`start`)
    pub wall_ms: u64,
}

/// A driver compile is in the driver right now.
static IN_FLIGHT: AtomicU64 = AtomicU64::new(0);
static COMPILES_STARTED: AtomicU64 = AtomicU64::new(0);
static COMPILES_DONE: AtomicU64 = AtomicU64::new(0);
static COMPILES_FAILED: AtomicU64 = AtomicU64::new(0);
static DRIVER_MICROS: AtomicU64 = AtomicU64::new(0);
static GATED_COMPILES: AtomicU64 = AtomicU64::new(0);
static SLOT_WAIT_MICROS: AtomicU64 = AtomicU64::new(0);
static COMPILES_PEAK: AtomicU64 = AtomicU64::new(0);
/// when the last frame was shown, in `since_epoch()` µs; 0 while none was
static LAST_PRESENT: AtomicU64 = AtomicU64::new(0);
static LONGEST_PRESENT_GAP: AtomicU64 = AtomicU64::new(0);
static PRESENT_GAPS_OVER_1S: AtomicU64 = AtomicU64::new(0);
static PRESENT_GAPS_OVER_10S: AtomicU64 = AtomicU64::new(0);
static STANDINS: AtomicU64 = AtomicU64::new(0);

const GAP_1S: u64 = 1_000_000;
const GAP_10S: u64 = 10_000_000;

static EPOCH: OnceLock<Instant> = OnceLock::new();

/// Begin counting the run's wall time from now (`host`, at the start).
pub fn start() {
    let _ = epoch();
}

/// When the counters count from: the run's start, or the first use when nothing called `start`.
fn epoch() -> Instant {
    *EPOCH.get_or_init(Instant::now)
}

fn micros(d: Duration) -> u64 {
    u64::try_from(d.as_micros()).unwrap_or(u64::MAX)
}

/// µs since `epoch()`, the clock every instant in here is kept in.
fn since_epoch() -> u64 {
    micros(epoch().elapsed())
}

/// A driver compile is about to enter the driver (`COMPILES_STARTED`, `COMPILES_PEAK`).
pub fn compile_started() {
    COMPILES_STARTED.fetch_add(1, Ordering::Relaxed);
    let now = IN_FLIGHT.fetch_add(1, Ordering::Relaxed) + 1;
    COMPILES_PEAK.fetch_max(now, Ordering::Relaxed);
}

/// It came back after `took` in the driver; `failed`: the driver refused it
/// (`COMPILES_DONE`, `COMPILES_FAILED`, `DRIVER_MICROS`).
pub fn compile_done(took: Duration, failed: bool) {
    COMPILES_DONE.fetch_add(1, Ordering::Relaxed);
    DRIVER_MICROS.fetch_add(micros(took), Ordering::Relaxed);
    if failed {
        COMPILES_FAILED.fetch_add(1, Ordering::Relaxed);
    }
    IN_FLIGHT.fetch_sub(1, Ordering::Relaxed);
}

/// A compile waited `waited` for a huge-compile slot (`GATED_COMPILES`, `SLOT_WAIT_MICROS`).
pub fn gate_wait(waited: Duration) {
    GATED_COMPILES.fetch_add(1, Ordering::Relaxed);
    SLOT_WAIT_MICROS.fetch_add(micros(waited), Ordering::Relaxed);
}

/// The counts a frame shown at `to` adds to the gap since the one at `from` (µs since
/// `epoch()`): the gap, and whether it was over 1 s and over 10 s. `None` for the run's first
/// frame – the wait before it is the run's startup, not a gap between two frames.
fn gap_counts(from: u64, to: u64) -> Option<(u64, bool, bool)> {
    (from != 0).then(|| {
        let gap = to.saturating_sub(from);
        (gap, gap > GAP_1S, gap > GAP_10S)
    })
}

/// A frame was shown: how long the last one was ago (`LONGEST_PRESENT_GAP`, the two gap counts).
pub fn present() {
    let now = since_epoch();
    let mut last = LAST_PRESENT.load(Ordering::Acquire);
    while let Some((gap, over_1s, over_10s)) = gap_counts(last, now) {
        match LAST_PRESENT.compare_exchange_weak(last, now, Ordering::AcqRel, Ordering::Acquire) {
            Ok(_) => {
                LONGEST_PRESENT_GAP.fetch_max(gap, Ordering::Relaxed);
                if over_1s {
                    PRESENT_GAPS_OVER_1S.fetch_add(1, Ordering::Relaxed);
                }
                if over_10s {
                    PRESENT_GAPS_OVER_10S.fetch_add(1, Ordering::Relaxed);
                }
                return;
            }
            Err(seen) => last = seen,
        }
    }
    // the run's first frame: the next one has something to measure from
    let _ = LAST_PRESENT.compare_exchange(0, now, Ordering::AcqRel, Ordering::Acquire);
}

/// A bind went on with the stand-in of a pipeline still being built (`STANDINS`).
pub fn standin() {
    STANDINS.fetch_add(1, Ordering::Relaxed);
}

/// The counters now, with the wall since `start`.
pub fn snapshot() -> Metrics {
    let (skipped_draws, skipped_dispatches) = super::bind::skipped();
    Metrics {
        compiles_started: COMPILES_STARTED.load(Ordering::Relaxed),
        compiles_done: COMPILES_DONE.load(Ordering::Relaxed),
        compiles_failed: COMPILES_FAILED.load(Ordering::Relaxed),
        driver_ms: DRIVER_MICROS.load(Ordering::Relaxed) / 1000,
        gated_compiles: GATED_COMPILES.load(Ordering::Relaxed),
        slot_wait_ms: SLOT_WAIT_MICROS.load(Ordering::Relaxed) / 1000,
        compiles_peak: COMPILES_PEAK.load(Ordering::Relaxed),
        presents: super::present::presents(),
        longest_present_gap_ms: LONGEST_PRESENT_GAP.load(Ordering::Relaxed) / 1000,
        present_gaps_over_1s: PRESENT_GAPS_OVER_1S.load(Ordering::Relaxed),
        present_gaps_over_10s: PRESENT_GAPS_OVER_10S.load(Ordering::Relaxed),
        standins: STANDINS.load(Ordering::Relaxed),
        skipped_draws,
        skipped_dispatches,
        wall_ms: since_epoch() / 1000,
    }
}

/// How much of the wall the driver compiles kept a slot busy: 100 when one slot was in the
/// driver all the time, 100 with 2 slots busy half the time each.
fn slots_busy_pct(m: &Metrics) -> u64 {
    100 * m.driver_ms / (m.wall_ms.max(1) * m.compiles_peak.max(1))
}

fn secs(ms: u64) -> f64 {
    ms as f64 / 1000.0
}

/// The metrics line's numbers, as `log` and `log_final` write them (the slots are
/// `compiles_peak` of them, `slots_busy_pct` of the wall):
/// `compiles 28 started/28 done/0 failed, driver 218.9 s over 162.0 s wall (slots 67% busy),
/// presents 2, longest present gap 41.2 s, gaps>1s 9, gaps>10s 3, stand-ins 41`
fn line(m: &Metrics) -> String {
    format!(
        "compiles {} started/{} done/{} failed, driver {:.1} s over {:.1} s wall (slots {}% busy), presents {}, longest present gap {:.1} s, gaps>1s {}, gaps>10s {}, stand-ins {}",
        m.compiles_started,
        m.compiles_done,
        m.compiles_failed,
        secs(m.driver_ms),
        secs(m.wall_ms),
        slots_busy_pct(m),
        m.presents,
        secs(m.longest_present_gap_ms),
        m.present_gaps_over_1s,
        m.present_gaps_over_10s,
        m.standins,
    )
}

/// The whole run's average frame rate.
fn average_fps(m: &Metrics) -> f64 {
    m.presents as f64 / secs(m.wall_ms).max(0.001)
}

/// The line every 60 s (`host`).
pub fn log() {
    crate::xlog_info!("D3D12.X: metrics: {}", line(&snapshot()));
}

/// The same numbers once at the exit (`report::write_final`), with the run's average frame rate.
pub fn log_final() {
    static ONCE: Once = Once::new();
    ONCE.call_once(|| {
        let m = snapshot();
        crate::xlog_info!("D3D12.X: metrics (final): {}, {:.1} fps over {:.1} s", line(&m), average_fps(&m), secs(m.wall_ms));
    });
}

#[cfg(test)]
mod tests {
    use super::*;

    fn metrics(started: u64, done: u64, failed: u64, driver_ms: u64, wall_ms: u64, peak: u64) -> Metrics {
        Metrics {
            compiles_started: started,
            compiles_done: done,
            compiles_failed: failed,
            driver_ms,
            compiles_peak: peak,
            wall_ms,
            ..Default::default()
        }
    }

    #[test]
    fn a_gap_between_frames_counts_from_the_second_frame_on() {
        assert_eq!(gap_counts(0, 5_000_000), None, "the run's first frame is not a gap");
        assert_eq!(gap_counts(1_000_000, 1_500_000), Some((500_000, false, false)));
        assert_eq!(gap_counts(1_000_000, 2_500_000), Some((1_500_000, true, false)));
        assert_eq!(gap_counts(1_000_000, 42_200_000), Some((41_200_000, true, true)));
        // exactly 1 s is not over 1 s
        assert_eq!(gap_counts(1, 1_000_001), Some((1_000_000, false, false)));
    }

    #[test]
    fn slots_are_busy_in_proportion_to_the_driver_time_and_how_many_ran_at_once() {
        // 218.9 s of driver time inside 162 s of wall with 2 compiles at once: 68% of both slots
        assert_eq!(slots_busy_pct(&metrics(28, 28, 0, 218_900, 162_000, 2)), 67);
        // one compile at a time in the driver throughout: both slots never ran together
        assert_eq!(slots_busy_pct(&metrics(1, 1, 0, 162_000, 162_000, 1)), 100);
        // nothing compiled, and a moment in which nothing was measured
        assert_eq!(slots_busy_pct(&metrics(0, 0, 0, 0, 162_000, 0)), 0);
        assert_eq!(slots_busy_pct(&metrics(0, 0, 0, 0, 0, 0)), 0);
    }

    #[test]
    fn the_line_reads_as_the_one_run_log() {
        let mut m = metrics(28, 28, 0, 218_900, 162_000, 2);
        m.presents = 2;
        m.longest_present_gap_ms = 41_200;
        m.present_gaps_over_1s = 9;
        m.present_gaps_over_10s = 3;
        m.standins = 41;
        assert_eq!(
            line(&m),
            "compiles 28 started/28 done/0 failed, driver 218.9 s over 162.0 s wall (slots 67% busy), presents 2, longest present gap 41.2 s, gaps>1s 9, gaps>10s 3, stand-ins 41"
        );
    }

    #[test]
    fn the_average_frame_rate_is_frames_over_the_whole_run() {
        let mut m = metrics(0, 0, 0, 0, 162_000, 0);
        m.presents = 2;
        assert!((average_fps(&m) - 0.012_345).abs() < 1e-6);
        // a run that presented nothing and barely ran does not divide by zero
        assert_eq!(average_fps(&Metrics::default()), 0.0);
    }
}
