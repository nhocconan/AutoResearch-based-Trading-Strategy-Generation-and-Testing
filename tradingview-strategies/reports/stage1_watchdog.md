# Stage 1 Watchdog

- updated_at: 2026-03-22T14:09:28.013441+00:00
- action: none
- next_check_at: 2026-03-22T14:24:28.273972+00:00
- runner_pids: 85177
- stage1_status: running
- batch_started_with: cached_ok=1528 pending=2073 errors=1802
- latest_manifest: cached_ok=1528 pending=2065 errors=1810 total_items=5403
- browsers: agent_browser=57 chrome_testing=49
- monitor_interval_seconds: 900
- stale_restart_minutes: 20

This file is rewritten by the 15-minute watchdog. If Stage 1 stalls or the runner exits while pending work remains, the watchdog kills leftover browser processes and restarts Stage 1.
