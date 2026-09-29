"""Low-overhead prompt-scoped resource telemetry for CK-Telemetry.

On Windows/AMD, it combines native WDDM memory performance counters via PDH, 
and system-level psutil telemetry under a single synchronized sampling lifecycle.
"""

from __future__ import annotations

import importlib
import math
import platform
import threading
import time
from dataclasses import dataclass, field
from typing import Any
import ctypes
from ctypes import wintypes

_SAMPLE_INTERVAL_SECONDS = 0.5

# --- Pure Native Win32 Structures for PDH ---
class DOUBLE_OR_LONG(ctypes.Union):
    _fields_ = [
        ("longValue", ctypes.c_long),
        ("doubleValue", ctypes.c_double),
        ("largeIntValue", ctypes.c_longlong),
        ("AnsiValue", ctypes.c_char_p),
        ("WideValue", ctypes.c_wchar_p)
    ]

class PDH_FMT_COUNTERVALUE(ctypes.Structure):
    _anonymous_ = ("u",)
    _fields_ = [
        ("CStatus", wintypes.DWORD),
        ("Format", wintypes.DWORD),
        ("u", DOUBLE_OR_LONG)
    ]

class DISPLAY_DEVICEW(ctypes.Structure):
    _fields_ = [
        ("cb", wintypes.DWORD),
        ("DeviceName", wintypes.WCHAR * 32),
        ("DeviceString", wintypes.WCHAR * 128),
        ("StateFlags", wintypes.DWORD),
        ("DeviceID", wintypes.WCHAR * 128),
        ("DeviceKey", wintypes.WCHAR * 128)
    ]


@dataclass
class ResourceStats:
    """Accumulated statistics for one prompt-scoped telemetry run."""

    sample_count: int = 0
    gpu_util_samples: list[float] = field(default_factory=list)
    vram_used_mb_samples: list[float] = field(default_factory=list)
    gpu_shared_memory_mb_samples: list[float] = field(default_factory=list)
    
    # Precise WDDM Native Metrics
    wddm_dedicated_mb_samples: list[float] = field(default_factory=list)
    wddm_shared_mb_samples: list[float] = field(default_factory=list)
    wddm_committed_mb_samples: list[float] = field(default_factory=list)

    cpu_util_samples: list[float] = field(default_factory=list)
    ram_used_mb_samples: list[float] = field(default_factory=list)
    commit_used_mb_samples: list[float] = field(default_factory=list)
    commit_limit_mb: float | None = None
    ram_total_mb: float | None = None
    vram_total_mb: float | None = None
    
    started_perf: float | None = None
    ended_perf: float | None = None
    provider: str | None = None
    provider_error: str | None = None
    wddm_status: str | None = None

    def add_sample(self, sample: dict[str, Any], now: float) -> None:

        self.sample_count += 1
        self.started_perf = self.started_perf if self.started_perf is not None else now
        self.ended_perf = now

        wddm_load = sample.get("wddm_gpu_util")
        _append_finite(self.gpu_util_samples, wddm_load if wddm_load is not None else sample.get("gpu_util_percent"))
        _append_finite(self.vram_used_mb_samples, sample.get("vram_used_mb"))
        _append_finite(self.gpu_shared_memory_mb_samples, sample.get("wddm_shared_mb"))
        
        # Unpack native WDDM arrays
        _append_finite(self.wddm_dedicated_mb_samples, sample.get("wddm_dedicated_mb"))
        _append_finite(self.wddm_shared_mb_samples, sample.get("wddm_shared_mb"))
        _append_finite(self.wddm_committed_mb_samples, sample.get("wddm_committed_mb"))

        _append_finite(self.cpu_util_samples, sample.get("cpu_util_percent"))
        _append_finite(self.ram_used_mb_samples, sample.get("ram_used_mb"))
        _append_finite(self.commit_used_mb_samples, sample.get("commit_used_mb"))

        if _is_number(sample.get("vram_total_mb")):
            self.vram_total_mb = float(sample["vram_total_mb"])
        if _is_number(sample.get("ram_total_mb")):
            self.ram_total_mb = float(sample["ram_total_mb"])
        if _is_number(sample.get("commit_limit_mb")):
            self.commit_limit_mb = float(sample["commit_limit_mb"])

    def to_dict(self) -> dict[str, Any]:
        data: dict[str, Any] = {
            "samples": self.sample_count,
            "sample_interval_seconds": _SAMPLE_INTERVAL_SECONDS,
            "provider": self.provider,
            "wddm_status": self.wddm_status,
            "gpu_utilization_percent": _summary(self.gpu_util_samples),
            "vram_used_mb": _summary(self.vram_used_mb_samples),
            
            # Formatted clean native outputs for output processing
            "wddm_dedicated_gpu_memory_mb": _summary(self.wddm_dedicated_mb_samples),
            "wddm_shared_gpu_memory_mb": _summary(self.wddm_shared_mb_samples),
            "wddm_total_gpu_committed_mb": _summary(self.wddm_committed_mb_samples),
            
            "gpu_shared_memory_mb": _summary(self.gpu_shared_memory_mb_samples),
            "cpu_utilization_percent": _summary(self.cpu_util_samples),
            "ram_used_mb": _summary(self.ram_used_mb_samples),
            "commit_used_mb": _summary(self.commit_used_mb_samples),
            "ram_total_mb": round(self.ram_total_mb, 2) if self.ram_total_mb is not None else None,
            "vram_total_mb": round(self.vram_total_mb, 2) if self.vram_total_mb is not None else None,
            "commit_limit_mb": round(self.commit_limit_mb, 2) if self.commit_limit_mb is not None else None,
        }
        if self.provider_error:
            data["provider_error"] = self.provider_error
        return data

class _WDDMProvider:
    """Zero-dependency Telemetry Provider utilizing Windows Performance Data Helpers (PDH).

    Dynamically matches host process IDs and hardware LUID signatures to preserve 
    absolute open-source portability across Windows AMD rigs.
    """
    def __init__(self, target_gpu_name="unknown"):
        self.target_gpu_name = target_gpu_name or "unknown"
        self.pdh = ctypes.windll.pdh
        self.user32 = ctypes.windll.user32
        
        self.hQuery = wintypes.HANDLE()
        self.hDec = wintypes.HANDLE()
        self.hShr = wintypes.HANDLE()
        self.hCom = wintypes.HANDLE()
        
        self.target_luid_str = None
        self.active_luid_hex = ""
        self.registered_util_handles = []
        self.is_ready = False
        self.vram_total_mb = None

    def start(self):
        
        pcchCounterLength = wintypes.DWORD(0)
        pcchInstanceLength = wintypes.DWORD(0)
        
        self.pdh.PdhEnumObjectItemsW(None, None, "GPU Adapter Memory", None, ctypes.byref(pcchCounterLength), None, ctypes.byref(pcchInstanceLength), 1, 0)
        if pcchInstanceLength.value == 0:
            raise RuntimeError("WDDM memory objects not exposed by OS configuration.")
            
        mszCounters = ctypes.create_unicode_buffer(pcchCounterLength.value)
        mszInstances = ctypes.create_unicode_buffer(pcchInstanceLength.value)
        
        status = self.pdh.PdhEnumObjectItemsW(None, None, "GPU Adapter Memory", mszCounters, ctypes.byref(pcchCounterLength), mszInstances, ctypes.byref(pcchInstanceLength), 1, 0)
        if status != 0:
            raise RuntimeError("Failed executing dynamic instance namespace enumeration block.")

        instances = []
        offset = 0
        while offset < pcchInstanceLength.value:
            curr_str = ctypes.wstring_at(ctypes.addressof(mszInstances) + (offset * 2))
            if not curr_str:
                break
            instances.append(curr_str)
            offset += len(curr_str) + 1

        dev = DISPLAY_DEVICEW()
        dev.cb = ctypes.sizeof(dev)
        idx = 0
        while self.user32.EnumDisplayDevicesW(None, idx, ctypes.byref(dev), 0):
            if dev.StateFlags & 0x00000004:
                friendly = dev.DeviceString
                if "AMD" in friendly.upper() or "RADEON" in friendly.upper() or self.target_gpu_name.lower() in friendly.lower():
                    for inst in instances:
                        if (len(instances) == 1) or (idx == 0 and "phys" in inst):
                            self.target_luid_str = inst
                            break
            idx += 1

        if not self.target_luid_str and instances:
            self.target_luid_str = instances

        if not self.target_luid_str:
            raise RuntimeError("Could not establish a verified hardware instance path link.")

        try:
            parts = self.target_luid_str.split('_')
            for part in parts:
                if part.lower().startswith("0x") or len(part) == 8:
                    self.active_luid_hex = part.upper()
                    if self.active_luid_hex.startswith("0X"):
                        self.active_luid_hex = self.active_luid_hex[2:]
                    break
        except Exception:
            pass

        if self.pdh.PdhOpenQueryW(None, 0, ctypes.byref(self.hQuery)) != 0:
            raise RuntimeError("Failed configuring performance tracking query channel.")

        p_dec = f"\\GPU Adapter Memory({self.target_luid_str})\\Dedicated Usage"
        p_shr = f"\\GPU Adapter Memory({self.target_luid_str})\\Shared Usage"
        p_com = f"\\GPU Adapter Memory({self.target_luid_str})\\Total Committed"

        self.pdh.PdhAddCounterW(self.hQuery, p_dec, 0, ctypes.byref(self.hDec))
        self.pdh.PdhAddCounterW(self.hQuery, p_shr, 0, ctypes.byref(self.hShr))
        self.pdh.PdhAddCounterW(self.hQuery, p_com, 0, ctypes.byref(self.hCom))

        pcchEngCounters = wintypes.DWORD(0)
        pcchEngInstances = wintypes.DWORD(0)
        self.pdh.PdhEnumObjectItemsW(None, None, "GPU Engine", None, ctypes.byref(pcchEngCounters), None, ctypes.byref(pcchEngInstances), 1, 0)
        
        mszEngCounters = ctypes.create_unicode_buffer(pcchEngCounters.value)
        mszEngInstances = ctypes.create_unicode_buffer(pcchEngInstances.value)
        status_eng = self.pdh.PdhEnumObjectItemsW(None, None, "GPU Engine", mszEngCounters, ctypes.byref(pcchEngCounters), mszEngInstances, ctypes.byref(pcchEngInstances), 1, 0)

        if status_eng == 0 and self.active_luid_hex:
            offset = 0
            while offset < pcchEngInstances.value:
                curr_eng = ctypes.wstring_at(ctypes.addressof(mszEngInstances) + (offset * 2))
                if not curr_eng:
                    break

                if self.active_luid_hex in curr_eng.upper() and ("ENGTYPE_3D" in curr_eng.upper() or "ENG_0" in curr_eng.upper()):
                    hTmpUtil = wintypes.HANDLE()
                    p_util_path = f"\\GPU Engine({curr_eng})\\Utilization Percentage"
                    if self.pdh.PdhAddCounterW(self.hQuery, p_util_path, 0, ctypes.byref(hTmpUtil)) == 0:
                        self.registered_util_handles.append(hTmpUtil)
                offset += len(curr_eng) + 1

        try:
            import comfy.model_management as mm
            self.vram_total_mb = mm.get_total_memory() / (1024 * 1024)
        except Exception:
            self.vram_total_mb = None

        self.pdh.PdhCollectQueryData(self.hQuery)
        self.is_ready = True


    def sample(self):
        if not self.is_ready or self.pdh.PdhCollectQueryData(self.hQuery) != 0:
            return {}

        val = PDH_FMT_COUNTERVALUE()
        PDH_FMT_DOUBLE = 0x00000200
        res = {"vram_total_mb": self.vram_total_mb}  # Fixed handshake key name

        if self.pdh.PdhGetFormattedCounterValue(self.hDec, PDH_FMT_DOUBLE, None, ctypes.byref(val)) == 0:
            res["wddm_dedicated_mb"] = val.u.doubleValue / (1024 * 1024)
        if self.pdh.PdhGetFormattedCounterValue(self.hShr, PDH_FMT_DOUBLE, None, ctypes.byref(val)) == 0:
            res["wddm_shared_mb"] = val.u.doubleValue / (1024 * 1024)
        if self.pdh.PdhGetFormattedCounterValue(self.hCom, PDH_FMT_DOUBLE, None, ctypes.byref(val)) == 0:
            res["wddm_committed_mb"] = val.u.doubleValue / (1024 * 1024)

        accumulated_load = 0.0
        for hCounter in self.registered_util_handles:
            if self.pdh.PdhGetFormattedCounterValue(hCounter, PDH_FMT_DOUBLE, None, ctypes.byref(val)) == 0:
                accumulated_load += val.u.doubleValue
        
        res["wddm_gpu_util"] = min(100.0, max(0.0, accumulated_load))
        return res


    def close(self):
        self.is_ready = False
        if self.hQuery:
            try:
                self.pdh.PdhCloseQuery(self.hQuery)
            except Exception:
                pass
        self.hQuery = None
        self.registered_util_handles = []

class ResourceSampler:
    """Collect low-frequency prompt-scoped hardware/system metrics."""

    def __init__(self, gpu_name: str = "unknown", interval_seconds: float = _SAMPLE_INTERVAL_SECONDS) -> None:
        self.interval_seconds = max(0.25, float(interval_seconds))
        self.gpu_name = gpu_name
        self.stats = ResourceStats()
        self._stop = threading.Event()
        self._thread: threading.Thread | None = None
        self._wddm_provider: _WDDMProvider | None = None
        self._psutil: Any | None = None

    def start(self) -> None:
        try:
            self._psutil = importlib.import_module("psutil")
        except Exception as exc:
            self._psutil = None
            self.stats.provider_error = f"psutil unavailable: {type(exc).__name__}: {exc}"

        self._thread = threading.Thread(target=self._run, name="CKResourceTelemetry", daemon=True)
        self._thread.start()

    def stop(self) -> dict[str, Any]:
        self._stop.set()
        if self._thread is not None:
            self._thread.join(timeout=max(2.0, self.interval_seconds * 4.0))
        return self.stats.to_dict()

    def _run(self) -> None:
        wddm = None
        try:
            if platform.system() == "Windows":
                try:
                    wddm = _WDDMProvider(self.gpu_name)
                    wddm.start()
                    self._wddm_provider = wddm
                    self.stats.wddm_status = "Active (Native PDH)"
                except Exception as exc:
                    self.stats.wddm_status = f"Unavailable: {type(exc).__name__}: {exc}"

            if self._psutil is not None:
                try:
                    self._psutil.cpu_percent(interval=None)
                except Exception:
                    pass

            # Orchestrate sample loop pass targeting only the active WDDM provider wrapper
            self._take_sample(wddm)
            while not self._stop.wait(self.interval_seconds):
                self._take_sample(wddm)
        finally:
            if wddm is not None:
                wddm.close()

    def _take_sample(self, wddm: _WDDMProvider | None) -> None:
        sample: dict[str, Any] = {}

        if wddm is not None:
            try:
                wddm_data = wddm.sample()
                sample.update(wddm_data)
                if "wddm_gpu_util" in wddm_data:
                    sample["gpu_util_percent"] = wddm_data["wddm_gpu_util"]
            except Exception as exc:
                if "Active" in str(self.stats.wddm_status):
                    self.stats.wddm_status = f"Sample loop exception: {type(exc).__name__}: {exc}"

        if self._psutil is not None:
            try:
                virtual = self._psutil.virtual_memory()
                sample["ram_used_mb"] = float(virtual.used) / (1024 * 1024)
                sample["ram_total_mb"] = float(virtual.total) / (1024 * 1024)
            except Exception:
                pass
            try:
                cpu = self._psutil.cpu_percent(interval=None)
                sample["cpu_util_percent"] = float(cpu)
            except Exception:
                pass
            try:
                swap = self._psutil.swap_memory()
                sample["commit_used_mb"] = max(0.0, float(swap.used) / (1024 * 1024))
                sample["commit_limit_mb"] = max(0.0, float(swap.total) / (1024 * 1024))
            except Exception:
                pass

        if sample:
            now = time.perf_counter()
            self.stats.add_sample(sample, now)
            

def _append_finite(target: list[float], value: Any) -> None:
    if _is_number(value):
        number = float(value)
        if math.isfinite(number):
            target.append(number)


def _is_number(value: Any) -> bool:
    return isinstance(value, (int, float)) and not isinstance(value, bool) and math.isfinite(float(value))


def _safe_method(obj: Any, name: str) -> float | None:
    try:
        value = getattr(obj, name)()
    except Exception:
        return None
    return float(value) if _is_number(value) else None


def _summary(values: list[float]) -> dict[str, float] | None:
    if not values:
        return None
    return {
        "peak": round(max(values), 3),
        "average": round(sum(values) / len(values), 3),
        "first": round(values[0], 3),
        "last": round(values[-1], 3),
    }