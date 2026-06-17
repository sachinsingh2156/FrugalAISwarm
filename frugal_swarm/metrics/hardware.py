"""
Hardware profiling — low-watermark baseline for edge / institutional servers.

Supervisor recommendation: benchmark against a realistic x86 institutional
server spec rather than developer laptops or cloud VMs.

LOW_WATERMARK_MODE (set via env LOW_WATERMARK_MODE=1 or config.py) activates:
  - Reduced swarm size (N=2)
  - Reduced max_tokens per agent call
  - Serial execution (no threading concurrency within a round)
  - Conservative Ollama timeout

Detected platform:
  - "apple_silicon"   — M-series Mac (ARM, unified memory)
  - "x86_institutional" — x86-64 server or workstation (target baseline)
  - "x86_consumer"    — x86-64 laptop / desktop
  - "unknown"         — anything else

Baseline specification (x86 institutional — LOW_WATERMARK target):
  CPU: dual-socket Intel Xeon E5-2680 v4 (14 cores / 28 threads per socket)
  RAM: 64 GB DDR4 ECC
  Storage: 480 GB SSD
  GPU: none (CPU inference only)
  TDP: ~300 W (both sockets)
  OS: Ubuntu 22.04 LTS
"""
from __future__ import annotations

import os
import platform
import sys
from dataclasses import dataclass
from typing import Optional

try:
    import psutil
    _PSUTIL_AVAILABLE = True
except ImportError:
    _PSUTIL_AVAILABLE = False


# ── Platform detection ────────────────────────────────────────────────────────

def detect_platform() -> str:
    """
    Return a short string identifying the execution platform.

    Returns one of: "apple_silicon", "x86_institutional", "x86_consumer", "unknown"
    """
    machine = platform.machine().lower()
    proc = platform.processor().lower()
    system = platform.system()

    if machine in ("arm64", "aarch64") and system == "Darwin":
        return "apple_silicon"

    if machine in ("x86_64", "amd64"):
        # Heuristic: institutional = ≥ 16 logical CPUs or "xeon" in processor string
        if _PSUTIL_AVAILABLE:
            logical_cpus = psutil.cpu_count(logical=True) or 0
            ram_gb = psutil.virtual_memory().total / (1024 ** 3)
            if "xeon" in proc or logical_cpus >= 16 or ram_gb >= 32:
                return "x86_institutional"
        return "x86_consumer"

    return "unknown"


# ── Low-watermark mode ────────────────────────────────────────────────────────

def is_low_watermark_mode() -> bool:
    """
    Return True if LOW_WATERMARK_MODE is active.

    Enabled by:
      - Environment variable: LOW_WATERMARK_MODE=1
      - config.py: LOW_WATERMARK_MODE = True
    """
    if os.getenv("LOW_WATERMARK_MODE", "").strip() in ("1", "true", "yes"):
        return True
    try:
        sys.path.insert(0, os.path.dirname(os.path.dirname(os.path.dirname(__file__))))
        from config import LOW_WATERMARK_MODE
        return bool(LOW_WATERMARK_MODE)
    except (ImportError, AttributeError):
        return False


# ── Hardware snapshot ─────────────────────────────────────────────────────────

@dataclass
class HardwareSnapshot:
    platform_tag: str
    cpu_count_logical: Optional[int]
    cpu_count_physical: Optional[int]
    ram_total_gb: Optional[float]
    ram_available_gb: Optional[float]
    python_version: str
    low_watermark_mode: bool

    def as_dict(self) -> dict:
        return {
            "platform_tag":        self.platform_tag,
            "cpu_count_logical":   self.cpu_count_logical,
            "cpu_count_physical":  self.cpu_count_physical,
            "ram_total_gb":        round(self.ram_total_gb, 2) if self.ram_total_gb else None,
            "ram_available_gb":    round(self.ram_available_gb, 2) if self.ram_available_gb else None,
            "python_version":      self.python_version,
            "low_watermark_mode":  self.low_watermark_mode,
        }


def get_hardware_snapshot() -> HardwareSnapshot:
    """Collect and return a HardwareSnapshot for the current machine."""
    tag = detect_platform()
    lwm = is_low_watermark_mode()

    cpu_logical = cpu_physical = ram_total = ram_avail = None
    if _PSUTIL_AVAILABLE:
        cpu_logical = psutil.cpu_count(logical=True)
        cpu_physical = psutil.cpu_count(logical=False)
        mem = psutil.virtual_memory()
        ram_total = mem.total / (1024 ** 3)
        ram_avail = mem.available / (1024 ** 3)

    return HardwareSnapshot(
        platform_tag=tag,
        cpu_count_logical=cpu_logical,
        cpu_count_physical=cpu_physical,
        ram_total_gb=ram_total,
        ram_available_gb=ram_avail,
        python_version=sys.version,
        low_watermark_mode=lwm,
    )


# ── Low-watermark parameter overrides ─────────────────────────────────────────

LOW_WATERMARK_PARAMS: dict = {
    "swarm_size":    2,
    "max_tokens":    256,
    "max_rounds":    1,
    "ollama_timeout": 60,  # seconds
    "serial_rounds": True,
}

STANDARD_PARAMS: dict = {
    "swarm_size":    3,
    "max_tokens":    512,
    "max_rounds":    2,
    "ollama_timeout": 120,
    "serial_rounds": False,
}


def get_run_params() -> dict:
    """Return parameter overrides appropriate for the current hardware mode."""
    return LOW_WATERMARK_PARAMS if is_low_watermark_mode() else STANDARD_PARAMS


# ── Quick CLI report ──────────────────────────────────────────────────────────

if __name__ == "__main__":
    snap = get_hardware_snapshot()
    print("\n── Hardware Profile ────────────────────")
    for k, v in snap.as_dict().items():
        print(f"  {k:<22}: {v}")
    params = get_run_params()
    print("\n── Run Parameters ──────────────────────")
    for k, v in params.items():
        print(f"  {k:<22}: {v}")
    print()
