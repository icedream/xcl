//! What the title is told about memory, and how often it asks.
//!
//! `XCL_MEM_CALLS=1` puts the counters on the 2 s activity line, so a loader stall can be matched
//! against the same timeline as the DirectStorage request counter. `QueryVideoMemoryInfo` needs no
//! counter: the title that polls DXGI gets the same numbers XCL prints for itself.

use std::sync::atomic::{AtomicU64, Ordering};

static CALLS_TITLE_STATUS: AtomicU64 = AtomicU64::new(0);
static CALLS_WORKING_SET: AtomicU64 = AtomicU64::new(0);
static LAST_TITLE_AVAIL: AtomicU64 = AtomicU64::new(0);
static LAST_TITLE_USED: AtomicU64 = AtomicU64::new(0);
static LAST_WS_AVAIL: AtomicU64 = AtomicU64::new(0);
static LAST_WS_USED: AtomicU64 = AtomicU64::new(0);

pub(crate) fn title_status(total: u64, used: u64) {
    CALLS_TITLE_STATUS.fetch_add(1, Ordering::Relaxed);
    LAST_TITLE_AVAIL.store(total - used, Ordering::Relaxed);
    LAST_TITLE_USED.store(used, Ordering::Relaxed);
}

pub(crate) fn working_set(limit: u64, used: u64) {
    CALLS_WORKING_SET.fetch_add(1, Ordering::Relaxed);
    LAST_WS_AVAIL.store(limit - used, Ordering::Relaxed);
    LAST_WS_USED.store(used, Ordering::Relaxed);
}

/// One line per 2 s sample: `enabled` is `XCL_MEM_CALLS=1`.
pub(crate) fn line(enabled: bool) -> Option<String> {
    if !enabled {
        return None;
    }
    let mb = |b: u64| b >> 20;
    Some(format!(
        "title memory view: TitleMemoryStatus used {}/{} MB ({} calls), XMemGetWorkingSetStatistics used {}/{} MB ({} calls)",
        mb(LAST_TITLE_USED.load(Ordering::Relaxed)),
        mb(LAST_TITLE_USED.load(Ordering::Relaxed) + LAST_TITLE_AVAIL.load(Ordering::Relaxed)),
        CALLS_TITLE_STATUS.load(Ordering::Relaxed),
        mb(LAST_WS_USED.load(Ordering::Relaxed)),
        mb(LAST_WS_USED.load(Ordering::Relaxed) + LAST_WS_AVAIL.load(Ordering::Relaxed)),
        CALLS_WORKING_SET.load(Ordering::Relaxed),
    ))
}

/// The switch, read once.
pub(crate) fn wanted() -> bool {
    static V: std::sync::OnceLock<bool> = std::sync::OnceLock::new();
    *V.get_or_init(|| std::env::var("XCL_MEM_CALLS").ok().as_deref() == Some("1"))
}

#[cfg(test)]
mod tests {
    use super::*;

    #[test]
    fn counters_track_the_values_handed_to_the_title() {
        let before = CALLS_TITLE_STATUS.load(Ordering::Relaxed);
        title_status(100 << 20, 60 << 20);
        assert_eq!(CALLS_TITLE_STATUS.load(Ordering::Relaxed), before + 1);
        assert_eq!(LAST_TITLE_AVAIL.load(Ordering::Relaxed), 40 << 20);
        let text = line(true).unwrap();
        assert!(text.contains("TitleMemoryStatus used 60/100 MB"), "{text}");
        assert!(line(true).is_some() && line(false).is_none());
    }
}
