"""Thread-safe NVML hardware telemetry engine with psutil fallback."""

from __future__ import annotations

import logging
import threading
import time
from dataclasses import dataclass, field
from typing import Optional

logger = logging.getLogger(__name__)

_NVML_AVAILABLE = False
_nvml_handle = None

try:
    import pynvml
    pynvml.nvmlInit()
    _NVML_AVAILABLE = True
except Exception as _e:
    logger.warning(
        "pynvml init failed (%s) — GPU telemetry disabled, falling back to psutil.",
        _e,
    )

try:
    import psutil as _psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False
    logger.warning("psutil not installed — CPU/RAM fallback also disabled.")


@dataclass
class TelemetrySnapshot:
    timestamp_s: float = 0.0
    vram_used_mb: float = 0.0
    vram_total_mb: float = 0.0
    power_draw_w: float = 0.0
    gpu_util_pct: float = 0.0
    cpu_util_pct: float = 0.0
    ram_used_mb: float = 0.0


@dataclass
class TelemetrySummary:
    peak_vram_mb: float = 0.0
    avg_vram_mb: float = 0.0
    peak_power_w: float = 0.0
    avg_power_w: float = 0.0
    avg_gpu_util_pct: float = 0.0
    avg_cpu_util_pct: float = 0.0
    avg_ram_used_mb: float = 0.0
    sample_count: int = 0
    gpu_available: bool = False


class HardwareMonitor:
    """Background polling thread at a fixed 50 ms interval."""

    def __init__(self, device_index: int = 0, interval_ms: int = 50):
        self._interval = interval_ms / 1000.0
        self._device_index = device_index
        self._lock = threading.Lock()
        self._snapshots: list[TelemetrySnapshot] = []
        self._running = False
        self._thread: Optional[threading.Thread] = None
        self._gpu_ok = False
        self._handle = None

        if _NVML_AVAILABLE:
            try:
                self._handle = pynvml.nvmlDeviceGetHandleByIndex(device_index)
                self._gpu_ok = True
            except Exception as e:
                logger.warning(
                    "Cannot get NVML handle for device %d (%s) — disabling GPU sampling.",
                    device_index, e,
                )

    # ------------------------------------------------------------------
    # Context manager
    # ------------------------------------------------------------------

    def __enter__(self) -> "HardwareMonitor":
        self.start()
        return self

    def __exit__(self, *_) -> None:
        self.stop()

    # ------------------------------------------------------------------
    # Control
    # ------------------------------------------------------------------

    def start(self) -> None:
        self._running = True
        self._snapshots.clear()
        self._thread = threading.Thread(target=self._poll_loop, daemon=True, name="hw-monitor")
        self._thread.start()

    def stop(self) -> None:
        self._running = False
        if self._thread:
            self._thread.join(timeout=2.0)

    def reset(self) -> None:
        with self._lock:
            self._snapshots.clear()

    # ------------------------------------------------------------------
    # Polling
    # ------------------------------------------------------------------

    def _poll_loop(self) -> None:
        while self._running:
            snap = self._collect()
            with self._lock:
                self._snapshots.append(snap)
            time.sleep(self._interval)

    def _collect(self) -> TelemetrySnapshot:
        snap = TelemetrySnapshot(timestamp_s=time.monotonic())

        if self._gpu_ok:
            try:
                mem = pynvml.nvmlDeviceGetMemoryInfo(self._handle)
                snap.vram_used_mb = mem.used / 1024**2
                snap.vram_total_mb = mem.total / 1024**2
                snap.power_draw_w = pynvml.nvmlDeviceGetPowerUsage(self._handle) / 1000.0
                util = pynvml.nvmlDeviceGetUtilizationRates(self._handle)
                snap.gpu_util_pct = util.gpu
            except Exception as e:
                logger.debug("NVML read error: %s", e)

        if _PSUTIL_AVAILABLE:
            try:
                snap.cpu_util_pct = _psutil.cpu_percent(interval=None)
                vm = _psutil.virtual_memory()
                snap.ram_used_mb = vm.used / 1024**2
            except Exception as e:
                logger.debug("psutil read error: %s", e)

        return snap

    # ------------------------------------------------------------------
    # Summary
    # ------------------------------------------------------------------

    def summarize(self) -> TelemetrySummary:
        with self._lock:
            snaps = list(self._snapshots)

        if not snaps:
            return TelemetrySummary(gpu_available=self._gpu_ok)

        vrams  = [s.vram_used_mb  for s in snaps]
        powers = [s.power_draw_w  for s in snaps]
        gpus   = [s.gpu_util_pct  for s in snaps]
        cpus   = [s.cpu_util_pct  for s in snaps]
        rams   = [s.ram_used_mb   for s in snaps]

        return TelemetrySummary(
            peak_vram_mb     = max(vrams),
            avg_vram_mb      = sum(vrams) / len(vrams),
            peak_power_w     = max(powers),
            avg_power_w      = sum(powers) / len(powers),
            avg_gpu_util_pct = sum(gpus) / len(gpus),
            avg_cpu_util_pct = sum(cpus) / len(cpus),
            avg_ram_used_mb  = sum(rams) / len(rams),
            sample_count     = len(snaps),
            gpu_available    = self._gpu_ok,
        )

    def snapshots_as_dicts(self) -> list[dict]:
        with self._lock:
            return [s.__dict__.copy() for s in self._snapshots]
