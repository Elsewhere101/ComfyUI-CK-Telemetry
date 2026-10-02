"""
CK-Telemetry (Comfy Kitchen Telemetry)

ComfyUI diagnostic utility for ROCm/HIP Comfy Kitchen builds.

Features:
- Instruments Comfy Kitchen's registry-level backend selection instead of parsing
  human-readable log strings.
- Automatically detects the active PyTorch device name and gfx architecture.
- Ability to keep per-prompt hardware/stack performance metrics & dispatch counts for backend/function selections.
- Prints compact or detailed terminal report when enabled.
- Provides one universal pass-through node that can:
    * attach reports to EXTRA_PNGINFO;
    * expose the report as a STRING output for Preview as Text or other text nodes.

The instrumentation itself is armed only for prompts that either:
- request global telemetry through COMFY_KITCHEN_TELEMETRY; or utilize the versatile telemetry node.

Environment variable:
    COMFY_KITCHEN_TELEMETRY=0      (default) disabled
    COMFY_KITCHEN_TELEMETRY=1      per session compact terminal summary for every prompt without need for the custom node
    COMFY_KITCHEN_TELEMETRY=2      per session detailed terminal report for every prompt without need for the custom node

The node can opt a single workflow into detailed file metadata without requiring a global environment-variable setting.
"""

from __future__ import annotations

import importlib.metadata
import logging
import os
import platform
import re
import sys
import threading
import time
from collections import Counter, defaultdict
from dataclasses import dataclass, field
from datetime import datetime, timezone
from pathlib import Path
from typing import Any

from typing_extensions import override

from comfy_api.latest import ComfyExtension, io, ui

from .resource_telemetry import ResourceSampler


# -----------------------------------------------------------------------------
# Constants / configuration
# -----------------------------------------------------------------------------

TELEMETRY_VERSION = 5
PRIMARY_NODE_ID = "CK-Telemetry"
LEGACY_NODE_IDS = {"CKDispatchReport", "ComfyKitchenTelemetryImage", "ComfyKitchenTelemetryVideo"}
NODE_IDS = {PRIMARY_NODE_ID, *LEGACY_NODE_IDS}
REPORT_METADATA_KEY = "comfy_kitchen_dispatch"
REPORT_TEXT_METADATA_KEY = "comfy_kitchen_dispatch_report"

_LOG_LOCK = threading.Lock()
_TLS = threading.local()


# -----------------------------------------------------------------------------
# Runtime state
# -----------------------------------------------------------------------------

@dataclass
class TelemetrySettings:
    active: bool = False
    show_terminal_report: bool = False
    terminal_report_level: str = "summary"
    embed_metadata: bool = False
    node_report_level: str = "summary"

    def merge_node_inputs(self, inputs: dict[str, Any]) -> None:
        self.active = True
        self.show_terminal_report = self.show_terminal_report or bool(inputs.get("show_terminal_report", False))
        self.embed_metadata = self.embed_metadata or bool(inputs.get("embed_metadata", True))

        level = str(inputs.get("node_report_level", "summary")).lower()
        if level == "detailed":
            self.node_report_level = "detailed"

        terminal_level = str(inputs.get("terminal_report_level", "summary")).lower()
        if terminal_level == "detailed":
            self.terminal_report_level = "detailed"


@dataclass
class TelemetryState:
    prompt_id: str = "unknown"
    settings: TelemetrySettings = field(default_factory=TelemetrySettings)
    started_at: datetime = field(default_factory=lambda: datetime.now(timezone.utc))
    started_perf: float = field(default_factory=time.perf_counter)
    backend_counts: Counter = field(default_factory=Counter)
    operation_counts: dict[str, Counter] = field(default_factory=lambda: defaultdict(Counter))
    dispatch_failures: Counter = field(default_factory=Counter)
    explicit_backend_failures: Counter = field(default_factory=Counter)
    resource_sampler: ResourceSampler | None = None
    resource_telemetry: dict[str, Any] | None = None

    def record_selection(self, backend: str, function_name: str) -> None:
        self.backend_counts[backend] += 1
        self.operation_counts[backend][function_name] += 1

    def record_failure(self, function_name: str, failures: dict[str, str] | None) -> None:
        if failures:
            for backend, reason in failures.items():
                self.dispatch_failures[(backend, function_name, str(reason))] += 1
        else:
            self.dispatch_failures[("unknown", function_name, "no backend available")] += 1

    def record_explicit_failure(self, backend: str, function_name: str, reason: str) -> None:
        self.explicit_backend_failures[(backend, function_name, reason)] += 1


# -----------------------------------------------------------------------------
# Helpers
# -----------------------------------------------------------------------------


def _get_state() -> TelemetryState | None:
    return getattr(_TLS, "state", None)


def _set_state(state: TelemetryState | None) -> None:
    if state is None:
        if hasattr(_TLS, "state"):
            delattr(_TLS, "state")
    else:
        _TLS.state = state


def _parse_env_level() -> int:
    raw = os.environ.get("COMFY_KITCHEN_TELEMETRY", "0").strip().lower()
    if raw in {"2", "detailed", "verbose"}:
        return 2
    if raw in {"1", "summary", "compact", "true", "yes", "on"}:
        return 1
    return 0


def _discover_prompt_settings(prompt: Any) -> TelemetrySettings:
    settings = TelemetrySettings()

    env_level = _parse_env_level()
    if env_level:
        settings.active = True
        settings.show_terminal_report = True
        settings.terminal_report_level = "detailed" if env_level >= 2 else "summary"
        settings.node_report_level = "detailed" if env_level >= 2 else "summary"

    if not isinstance(prompt, dict):
        return settings

    for node in prompt.values():
        if not isinstance(node, dict):
            continue
        if node.get("class_type") not in NODE_IDS:
            continue
        inputs = node.get("inputs")
        if isinstance(inputs, dict):
            settings.merge_node_inputs(inputs)

    return settings


def _normalize_gfx_arch(value: Any) -> str:
    if value is None:
        return "unknown"
    text = str(value)
    match = re.search(r"gfx\d+", text, flags=re.IGNORECASE)
    return match.group(0).lower() if match else text


def _safe_distribution_version() -> str:
    for name in ("comfy-kitchen", "comfy_kitchen"):
        try:
            return importlib.metadata.version(name)
        except importlib.metadata.PackageNotFoundError:
            pass
        except Exception:
            break
    return "unknown"


def _runtime_info() -> dict[str, Any]:
    info: dict[str, Any] = {
        "gpu_name": "unknown",
        "gfx_arch": "unknown",
        "torch_version": "unknown",
        "hip_version": "unknown",
        "python_version": platform.python_version(),
        "comfy_kitchen_version": _safe_distribution_version(),
        "comfy_kitchen_path": "unknown",
        "comfy_kitchen_backends": {},
        "metadata_disabled": False,
    }

    try:
        import torch

        info["torch_version"] = getattr(torch, "__version__", "unknown")
        info["hip_version"] = getattr(torch.version, "hip", None) or "unknown"

        if torch.cuda.is_available():
            device = torch.cuda.current_device()
            info["gpu_name"] = torch.cuda.get_device_name(device)
            props = torch.cuda.get_device_properties(device)
            info["gfx_arch"] = _normalize_gfx_arch(getattr(props, "gcnArchName", None))
            info["device_index"] = int(device)
    except Exception as exc:
        info["gpu_error"] = f"{type(exc).__name__}: {exc}"

    try:
        import comfy_kitchen

        info["comfy_kitchen_path"] = str(Path(comfy_kitchen.__file__).resolve())
        registry = getattr(comfy_kitchen, "registry", None)
        if registry is not None:
            try:
                info["comfy_kitchen_backends"] = registry.list_backends()
            except Exception as exc:
                info["comfy_kitchen_backends_error"] = f"{type(exc).__name__}: {exc}"
    except Exception as exc:
        info["comfy_kitchen_error"] = f"{type(exc).__name__}: {exc}"

    try:
        from comfy.cli_args import args
        info["metadata_disabled"] = bool(getattr(args, "disable_metadata", False))
    except Exception:
        pass

    return info


def _backend_status_line(name: str, status: Any) -> str:
    if not isinstance(status, dict):
        return f"  {name}: {status}"
    available = bool(status.get("available", False))
    disabled = bool(status.get("disabled", False))
    state = "available" if available and not disabled else "disabled" if disabled else "unavailable"
    capability_count = len(status.get("capabilities", [])) if isinstance(status.get("capabilities"), list) else 0
    return f"  {name:<8} {state:<10} ({capability_count} advertised capabilities)"


def _performance_snapshot(
    state: TelemetryState,
    elapsed_seconds: float | None = None,
    measurement: str = "to_report_node",
) -> dict[str, Any]:
    if elapsed_seconds is None:
        elapsed_seconds = max(0.0, time.perf_counter() - state.started_perf)

    return {
        "elapsed_seconds": round(elapsed_seconds, 3),
        "measurement": measurement,
        "resources": state.resource_telemetry,
    }


def _format_resource_lines(resources: dict[str, Any] | None, detailed: bool) -> list[str]:
    if not isinstance(resources, dict):
        return []

    # Safely pulling aggregated telemetry data maps
    gpu_data = resources.get("gpu_utilization_percent") or resources.get("wddm_gpu_util") or {}
    ram_sys = resources.get("ram_used_mb") or {}
    cpu_util = resources.get("cpu_utilization_percent") or {}
    
    wddm_dec = resources.get("wddm_dedicated_gpu_memory_mb") or {}
    wddm_shr = resources.get("wddm_shared_gpu_memory_mb") or {}
    wddm_com = resources.get("wddm_total_gpu_committed_mb") or {}
    
    ram_total_mb = resources.get("ram_total_mb")
    vram_total_mb = resources.get("vram_total_mb")
    commit_used_mb = resources.get("commit_used_mb") or {}
    commit_limit_mb = resources.get("commit_limit_mb")

    raw_rows = []

    # CPU Utilization Row Builder
    if cpu_util.get("peak") is not None:
        cpu_p = f"{cpu_util['peak']:.0f}%"
        cpu_a = f"{cpu_util['average']:.0f}% Avg" if cpu_util.get("average") is not None else "-"
        raw_rows.append(("CPU Load", cpu_p, cpu_a))

    # GPU Utilization Row Builder (Using unmanaged WDDM collection layers)
    if isinstance(gpu_data, dict) and gpu_data.get("peak") is not None:
        gpu_p = f"{gpu_data['peak']:.0f}%"
        gpu_a = f"{gpu_util_avg_string(gpu_data)}"
        raw_rows.append(("GPU Load", gpu_p, gpu_a))
    else:
        raw_rows.append(("GPU Load", "-", "-"))

    if wddm_dec.get("peak") is not None:
        v_dec_p = f"{_mb_to_gb(wddm_dec['peak']):.2f} GB"
        
        resolved_vram_mb = vram_total_mb
        if not resolved_vram_mb:
            try:
                import comfy.model_management as mm
                # Dynamically calculates the hardware byte matrix ceilings straight from active silicon registers
                resolved_vram_mb = mm.get_total_memory() / (1024 * 1024)
            except Exception:
                # Gracefully drops down to an unknown state instead of risking a false assumption
                resolved_vram_mb = None
                
        v_dec_t = f"{_mb_to_gb(resolved_vram_mb):.1f} GB" if resolved_vram_mb else "-"
        raw_rows.append(("VRAM", v_dec_p, v_dec_t))

    # System Shared RAM Graphics Page Overflow Track
    if wddm_shr.get("peak") is not None:
        v_shr_p = f"{_mb_to_gb(wddm_shr['peak']):.2f} GB"
        raw_rows.append(("RAM Spill", v_shr_p, "-"))

    # Host System Physical RAM Footprint Track
    if ram_sys.get("peak") is not None:
        ram_p = f"{_mb_to_gb(ram_sys['peak']):.1f} GB"
        ram_t = f"{_mb_to_gb(ram_total_mb):.1f} GB" if ram_total_mb else "-"
        raw_rows.append(("RAM", ram_p, ram_t))

    # Host System Pagefile Virtual Memory Allocation Bounds
    if commit_used_mb.get("peak") is not None:
        page_p = f"{_mb_to_gb(commit_used_mb['peak']):.1f} GB"
        page_t = f"{_mb_to_gb(commit_limit_mb):.1f} GB" if commit_limit_mb else "-"
        raw_rows.append(("Pagefile", page_p, page_t))

    if not raw_rows:
        return []

    lines = [
        "  ___________________________________________",
        "   Hardware     |    Peak     |   Avg/Total  ",
        "  --------------|-------------|--------------"
    ]

    for label, peak, total_avg in raw_rows:
        lines.append(f"  {label:<13} |  {peak:<10} | {total_avg:<11}")

    lines.append("  ___________________________________________")
    return lines


def _mb_to_gb(value: float) -> float:
    return float(value) / 1024.0


def _build_report(
    state: TelemetryState,
    level: str | None = None,
    elapsed_seconds: float | None = None,
    measurement: str = "to_report_node",
) -> tuple[dict[str, Any], str]:
    level = (level or state.settings.node_report_level).lower()
    detailed = level == "detailed"
    runtime = _runtime_info()
    timestamp = state.started_at.astimezone().isoformat(timespec="seconds")
    performance = _performance_snapshot(state, elapsed_seconds=elapsed_seconds, measurement=measurement)

    backend_counts = dict(sorted(state.backend_counts.items(), key=lambda item: (-item[1], item[0])))
    operations: dict[str, dict[str, int]] = {}
    for backend in sorted(state.operation_counts):
        operations[backend] = dict(
            sorted(
                state.operation_counts[backend].items(),
                key=lambda item: (-item[1], item[0]),
            )
        )

    failures = [
        {
            "backend": backend,
            "operation": function_name,
            "reason": reason,
            "count": count,
        }
        for (backend, function_name, reason), count in sorted(
            state.dispatch_failures.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    explicit_failures = [
        {
            "backend": backend,
            "operation": function_name,
            "reason": reason,
            "count": count,
        }
        for (backend, function_name, reason), count in sorted(
            state.explicit_backend_failures.items(),
            key=lambda item: (-item[1], item[0]),
        )
    ]

    structured: dict[str, Any] = {
        "telemetry_version": TELEMETRY_VERSION,
        "timestamp": timestamp,
        "prompt_id": state.prompt_id,
        "gpu": {
            "name": runtime.get("gpu_name", "unknown"),
            "gfx_arch": runtime.get("gfx_arch", "unknown"),
            "device_index": runtime.get("device_index"),
        },
        "software": {
            "python": runtime.get("python_version", "unknown"),
            "torch": runtime.get("torch_version", "unknown"),
            "hip": runtime.get("hip_version", "unknown"),
            "comfy_kitchen_version": runtime.get("comfy_kitchen_version", "unknown"),
            "comfy_kitchen_path": runtime.get("comfy_kitchen_path", "unknown"),
        },
        "performance": performance,
        "backend_counts": backend_counts,
        "operations": operations,
        "dispatch_failures": failures,
        "explicit_backend_failures": explicit_failures,
    }

    wddm_status = state.resource_telemetry.get("wddm_status") if isinstance(state.resource_telemetry, dict) else None
    if wddm_status:
        structured["wddm_status"] = wddm_status

    if detailed:
        backend_status = runtime.get("comfy_kitchen_backends", {})
        if isinstance(backend_status, dict):
            structured["backend_status"] = {
                name: {
                    "available": bool(status.get("available", False)) if isinstance(status, dict) else None,
                    "disabled": bool(status.get("disabled", False)) if isinstance(status, dict) else None,
                    "capability_count": len(status.get("capabilities", [])) if isinstance(status, dict) and isinstance(status.get("capabilities"), list) else 0,
                    "unavailable_reason": status.get("unavailable_reason") if isinstance(status, dict) else None,
                }
                for name, status in sorted(backend_status.items())
            }

    # -------------------------------------------------------------------------
    # Twin-Track Visual Layout Renderer Engine
    # -------------------------------------------------------------------------
    if not detailed:
        # TRACK A: Dynamic Summary Mode Layout Configuration Pass
        total_selections = sum(backend_counts.values())
        elapsed_label = "PROMPT EXECUTOR ELAPSED" if measurement == "prompt_executor" else "PROMPT RUNTIME ELAPSED"
        
        lines = [
            "COMFY-KITCHEN DISPATCH SUMMARY",
            "============================================================",
            f"  GPU: {runtime.get('gpu_name', 'unknown')} | {runtime.get('gfx_arch', 'unknown')}",
            f"  {elapsed_label}: {performance['elapsed_seconds']:.2f} s",
            f"  TOTAL DISPATCH SELECTIONS: {total_selections}",
            "  OPERATION                                    | CALLS",
            "  ---------------------------------------------|-----------"
        ]
        
        if operations:
            for backend, op_counts in sorted(operations.items()):
                for function_name, count in op_counts.items():
                    full_op_string = f"[{backend}] {function_name}"
                    lines.append(f"  {full_op_string:<44} | {count:>9}")
        else:
            lines.append("  No Comfy Kitchen backend selections observed.")
            
        lines.append("============================================================")
        
    else:
        # TRACK B: Comprehensive Monolithic Detailed Document Pass
        lines = [
            "ROCm/HIP COMFY-KITCHEN RUNTIME REPORT",
            "========================================================================",
            "",
            f"Timestamp      : {timestamp}",
            f"Prompt ID      : {state.prompt_id}",
            f"GPU            : {runtime.get('gpu_name', 'unknown')}",
            f"Architecture   : {runtime.get('gfx_arch', 'unknown')}",
            f"PyTorch        : {runtime.get('torch_version', 'unknown')}",
            f"HIP/ROCm       : {runtime.get('hip_version', 'unknown')}",
            f"CK package     : {runtime.get('comfy_kitchen_version', 'unknown')}",
            f"CK module path : {runtime.get('comfy_kitchen_path', 'unknown')}",
            "________________________________________________________________________",
            "",
            "PERFORMANCE",
            "________________________________________________________________________",
            "",
        ]

        if measurement == "prompt_executor":
            lines.append(f"  Elapsed prompt executor : {performance['elapsed_seconds']:.2f} s")
        else:
            lines.append(f"  Elapsed to report node : {performance['elapsed_seconds']:.2f} s")

        resource_lines = _format_resource_lines(state.resource_telemetry, detailed=detailed)
        if resource_lines:
            lines.append("")
            lines.extend(resource_lines)
        lines.append("________________________________________________________________________")

        lines.extend([
            "",
            "BACKEND DISPATCH COUNTS",
            "________________________________________________________________________",
            ""
        ])
        if backend_counts:
            for backend, count in backend_counts.items():
                lines.append(f"  {backend:<16} {count:>10} selections")
        else:
            lines.append("  No Comfy Kitchen backend selections were observed.")
        lines.append("________________________________________________________________________")

        lines.extend([
            "",
            "BACKEND STATUS",
            "________________________________________________________________________",
            ""
        ])
        backend_status = runtime.get("comfy_kitchen_backends", {})
        if backend_status:
            for name, status in sorted(backend_status.items()):
                lines.append(_backend_status_line(name, status))
        else:
            lines.append("  Backend status unavailable.")
        lines.append("________________________________________________________________________")

        lines.extend([
            "",
            "SELECTED OPERATIONS",
            "________________________________________________________________________",
            "  OPERATION                                    | CALLS",
            "  ---------------------------------------------|-----------"
        ])
        if operations:
            for backend, op_counts in sorted(operations.items()):
                for function_name, count in op_counts.items():
                    full_op_string = f"[{backend}] {function_name}"
                    lines.append(f"  {full_op_string:<44} | {count:>9}")
        else:
            lines.append("  No operations were recorded.")
        lines.append("________________________________________________________________________")

        if failures:
            lines.extend([
                "",
                "DISPATCH FAILURES",
                "________________________________________________________________________",
                ""
            ])
            for item in failures:
                lines.append(f"  {item['backend']}.{item['operation']} x{item['count']}: {item['reason']}")
            lines.append("________________________________________________________________________")

        if explicit_failures:
            lines.extend([
                "",
                "EXPLICIT BACKEND FAILURES",
                "________________________________________________________________________",
                ""
            ])
            for item in explicit_failures:
                lines.append(f"  {item['backend']}.{item['operation']} x{item['count']}: {item['reason']}")
            lines.append("________________________________________________________________________")

        lines.extend([
            "",
            "========================================================================",
        ])

    return structured, "\n".join(lines)


_ANSI_REDDISH = "\x1b[38;5;166m"
_ANSI_RESET = "\x1b[0m"


def _print_reddish_tag(tag: str, title: str) -> None:
    print(f"{_ANSI_REDDISH}{tag}{_ANSI_RESET} {title}")


def _print_terminal_report(state: TelemetryState, level: str = "detailed", executor_elapsed: float | None = None) -> None:

    _, report = _build_report(
        state,
        level=level,
        elapsed_seconds=executor_elapsed,
        measurement="prompt_executor",
    )
    
    _print_reddish_tag("[HIP+] CK-TELEMETRY [˻˺]", "")
    print("\n" + report + "\n")


def _print_compact_terminal_report(state: TelemetryState, executor_elapsed: float | None = None) -> None:
    # Cleanly sorts and processes incoming raw operation logs
    operations = {
        backend: dict(sorted(counter.items(), key=lambda item: (-item[1], item[0])))
        for backend, counter in state.operation_counts.items()
    }
    total = sum(sum(counts.values()) for counts in operations.values())

    # Prints Summary Headers
    _print_reddish_tag("[HIP+] CK-TELEMETRY [˻˺]", "COMFY-KITCHEN DISPATCH SUMMARY")
    print("============================================================")
    runtime = _runtime_info()
    print(f"  GPU: {runtime.get('gpu_name', 'unknown')} | {runtime.get('gfx_arch', 'unknown')}")
    if executor_elapsed is not None:
        print(f"  PROMPT EXECUTOR ELAPSED: {executor_elapsed:.2f} s")
    print(f"  TOTAL DISPATCH SELECTIONS: {total}")
    print("  OPERATION                                    | CALLS")
    print("  ---------------------------------------------|-----------")
    
    # Prints Aligned Operation Table Row Logs
    if not operations:
        print("  No Comfy Kitchen backend selections observed.")
    else:
        for backend, counts in sorted(operations.items()):
            for function_name, count in counts.items():
                # Formats the full operational string inside a unified layout window before padding
                full_op_string = f"[{backend}] {function_name}"
                print(f"  {full_op_string:<44} | {count:>9}")
                
    if state.dispatch_failures:
        print(f"  DISPATCH FAILURES: {sum(state.dispatch_failures.values())}")
    print("============================================================\n")

def _write_node_metadata(
    extra_pnginfo: dict[str, Any] | None,
    structured: dict[str, Any],
    report: str,
    unique_id: str | None,
) -> bool:
    if extra_pnginfo is None or not isinstance(extra_pnginfo, dict):
        return False

    # These top-level metadata entries are useful to external metadata readers.
    extra_pnginfo[REPORT_METADATA_KEY] = structured
    extra_pnginfo[REPORT_TEXT_METADATA_KEY] = report


    # Generated files later dropped into ComfyUI not only restore their workflow,
    # the frontend extension also copies this value back into the telemetry node's multiline report field.
    workflow = extra_pnginfo.get("workflow")
    if isinstance(workflow, dict):
        workflow_extra = workflow.setdefault("extra", {})
        if isinstance(workflow_extra, dict):
            workflow_extra[REPORT_METADATA_KEY] = structured
            workflow_extra[REPORT_TEXT_METADATA_KEY] = report

        if unique_id is not None:
            for node in workflow.get("nodes", []):
                if str(node.get("id")) == str(unique_id):
                    properties = node.setdefault("properties", {})
                    if isinstance(properties, dict):
                        properties[REPORT_PROPERTY] = report
                        properties[REPORT_STRUCTURED_PROPERTY] = structured
                    break
    return True


REPORT_PROPERTY = "comfy_kitchen_dispatch_report"
REPORT_STRUCTURED_PROPERTY = "comfy_kitchen_dispatch"


def _configure_state_for_node(
    state,
    show_terminal_report,
    terminal_report_level,
    node_report_level,
    embed_metadata,
):
    state.settings.active = True
    
    # 1. Capture what the global batch file environment variable originally requested
    env_had_terminal_on = state.settings.show_terminal_report
    env_terminal_level = getattr(state.settings, "terminal_report_level", None)
    
    # 2. Node Absolute Authority Assignment: The canvas node controls its own state registers cleanly
    node_wants_terminal = bool(show_terminal_report)
    state.settings.show_terminal_report = node_wants_terminal
    state.settings.embed_metadata = state.settings.embed_metadata or bool(embed_metadata)
    
    t_level = str(terminal_report_level).lower().strip()
    n_level = str(node_report_level).lower().strip()
    
    state.settings.terminal_report_level = t_level
    state.settings.node_report_level = n_level
    
    # 3. Proactive Logging Reminder Alerts
    _ANSI_REDDISH = "\x1b[38;5;166m"
    _ANSI_YELLA =  "\x1b[38;5;228m"
    _ANSI_RESET = "\x1b[0m"
    
    if env_had_terminal_on and not node_wants_terminal:
        print(f"{_ANSI_REDDISH}[HIP+] CK-TELEMETRY [˻˺]{_ANSI_YELLA} Node override:{_ANSI_RESET} show_terminal_report of node *suppressed* COMFY_KITCHEN_TELEMETRY\n initially set for {env_terminal_level.upper()} report level.\n")
        
    elif env_terminal_level and env_terminal_level != t_level and node_wants_terminal:
        print(f"{_ANSI_REDDISH}[HIP+] CK-TELEMETRY [˻˺]{_ANSI_YELLA} Node override:{_ANSI_RESET} Terminal report level set by COMFY_KITCHEN_TELEMETRY shifted\nfrom {env_terminal_level.upper()} to {t_level.upper()} via terminal_report_level selected from node.\n")


# -----------------------------------------------------------------------------
# Comfy Kitchen registry instrumentation
# -----------------------------------------------------------------------------


def _install_registry_hooks() -> tuple[Any, Any] | None:
    try:
        import comfy_kitchen
        registry = comfy_kitchen.registry
    except Exception:
        return None

    original_get_capable = registry.get_capable_backend
    original_get_implementation = registry.get_implementation

    def tracked_get_capable(func_name, kwargs=None):
        try:
            backend = original_get_capable(func_name, kwargs)
        except Exception as exc:
            state = _get_state()
            if state is not None:
                failures = getattr(exc, "failures", None)
                state.record_failure(func_name, failures)
            raise
        state = _get_state()
        if state is not None:
            state.record_selection(str(backend), str(func_name))
        return backend

    def tracked_get_implementation(func_name, backend=None, kwargs=None):
        try:
            implementation = original_get_implementation(func_name, backend, kwargs)
        except Exception as exc:
            state = _get_state()
            if state is not None and backend is not None:
                state.record_explicit_failure(str(backend), str(func_name), str(exc))
            elif state is not None and hasattr(exc, "failures"):
                state.record_failure(func_name, getattr(exc, "failures", None))
            raise

        if backend is not None:
            state = _get_state()
            if state is not None:
                state.record_selection(str(backend), str(func_name))
        return implementation

    registry.get_capable_backend = tracked_get_capable
    registry.get_implementation = tracked_get_implementation
    return original_get_capable, original_get_implementation


def _restore_registry_hooks(originals: tuple[Any, Any] | None) -> None:
    if originals is None:
        return
    try:
        import comfy_kitchen
        registry = comfy_kitchen.registry
        registry.get_capable_backend = originals[0]
        registry.get_implementation = originals[1]
    except Exception:
        pass


def _prepare_execution_state(prompt: Any, prompt_id: Any) -> TelemetryState | None:
    settings = _discover_prompt_settings(prompt)
    if not settings.active:
        return None
    state = TelemetryState(prompt_id=str(prompt_id), settings=settings)
    runtime = _runtime_info()
    state.resource_sampler = ResourceSampler(gpu_name=str(runtime.get("gpu_name", "unknown")))
    state.resource_sampler.start()
    _set_state(state)
    return state


# -----------------------------------------------------------------------------
# Prompt lifecycle hook
# -----------------------------------------------------------------------------

try:
    from execution import PromptExecutor
except Exception:
    PromptExecutor = None


if PromptExecutor is not None and not getattr(PromptExecutor, "_ck_dispatch_telemetry_patched", False):
    _ORIGINAL_EXECUTE = PromptExecutor.execute

    def _wrapped_execute(self, prompt, prompt_id, extra_data=None, execute_outputs=None):
        state = _prepare_execution_state(prompt, prompt_id)
        registry_originals = _install_registry_hooks() if state is not None else None
        executor_started = time.perf_counter()
        try:
            if extra_data is None:
                extra_data = {}
            if execute_outputs is None:
                execute_outputs = []
            return _ORIGINAL_EXECUTE(self, prompt, prompt_id, extra_data, execute_outputs)
        finally:
            executor_elapsed = max(0.0, time.perf_counter() - executor_started)
            if state is not None and state.resource_sampler is not None and state.resource_telemetry is None:
                state.resource_telemetry = state.resource_sampler.stop()
                state.resource_sampler = None

            if state is not None and state.settings.show_terminal_report:
                if state.settings.terminal_report_level == "detailed":
                    _print_terminal_report(state, level="detailed", executor_elapsed=executor_elapsed)
                else:
                    _print_compact_terminal_report(state, executor_elapsed=executor_elapsed)

            _restore_registry_hooks(registry_originals)
            _set_state(None)

    PromptExecutor.execute = _wrapped_execute
    PromptExecutor._ck_dispatch_telemetry_patched = True


# -----------------------------------------------------------------------------
# Pass-through nodes
# -----------------------------------------------------------------------------


class _TelemetryNodeMixin:
    @classmethod
    def fingerprint_inputs(cls, **kwargs):
        return time.time_ns()

    @classmethod
    def _inputs_schema(cls):
        return [
            io.Boolean.Input(
                "embed_metadata",
                default=True,
                advanced=True,
                tooltip="Embed the report in EXTRA_PNGINFO for a downstream native saver. Leave on for persistence; turn off when using this node only for temporary per session terminal telemetry.",
            ),
            io.Combo.Input(
                "terminal_report_level",
                options=["summary", "detailed"],
                default="summary",
                tooltip="Terminal report level when show_terminal_report is enabled.",
            ),
            io.Boolean.Input(
                "show_terminal_report",
                default=True,
                tooltip="Print a compact or detailed Comfy Kitchen dispatch report in the ComfyUI terminal after prompt.",
            ),
            io.Combo.Input(
                "node_report_level",
                options=["summary", "detailed"],
                default="detailed",
                tooltip="Report detail level of reports used for the node field, optional report output, and embedded metadata.",
            ),
            io.Boolean.Input(
                "show_node_report",
                default=True,
                tooltip="Show the populated report field below. The field is read-only and restored when a generated file's workflow is dropped back onto ComfyUI's canvas.",
            ),
            io.String.Input(
                "report_text",
                multiline=True,
                default="",
                socketless=True,
                tooltip="The read-only diagnostic report output; it is refreshed after each execution.",
            ),
        ]

    @classmethod
    def _execute_common(
        cls,
        embed_metadata: bool,
        terminal_report_level: str,
        show_terminal_report: bool,
        node_report_level: str,
        extra_pnginfo: dict[str, Any] | None,
        unique_id: str | None,
    ) -> str:
        state = _get_state()
        if state is None:
            state = TelemetryState(prompt_id="unknown", settings=TelemetrySettings(active=True))
            runtime = _runtime_info()
            state.resource_sampler = ResourceSampler(gpu_name=str(runtime.get("gpu_name", "unknown")))
            state.resource_sampler.start()
            _set_state(state)

        _configure_state_for_node(
            state,
            show_terminal_report=show_terminal_report,
            terminal_report_level=terminal_report_level,
            node_report_level=node_report_level,
            embed_metadata=embed_metadata,
        )

        elapsed_to_report = max(0.0, time.perf_counter() - state.started_perf)
        if state.resource_sampler is not None:
            state.resource_telemetry = state.resource_sampler.stop()
            state.resource_sampler = None
        structured, report = _build_report(
            state,
            level=state.settings.node_report_level,
            elapsed_seconds=elapsed_to_report,
            measurement="to_report_node",
        )

        if embed_metadata:
            metadata_written = _write_node_metadata(extra_pnginfo, structured, report, unique_id)
            if not metadata_written and extra_pnginfo is None:
                logging.warning("[HIP+] CK-TELEMETRY [˻˺] EXTRA_PNGINFO unavailable; file metadata was not attached.")
            elif not metadata_written:
                logging.warning("[HIP+] CK-TELEMETRY [˻˺] EXTRA_PNGINFO was not a writable dict; file metadata was not attached.")

        return report


class ComfyKitchenTelemetryReport(_TelemetryNodeMixin, io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id=PRIMARY_NODE_ID,
            display_name="CK Telemetry (ROCm/HIP)",
            category="diagnostics/comfy kitchen",
            search_aliases=[
                "comfyui",
                "comfy kitchen",
                "ck telemetry",
                "telemetry",
                "performance",
                "diagnostics",
                "benchmarking",
                "quantization",
                "amd gpu",
                "amd",
                "rocm",
                "hip",
                "wddm",
            ],
            description=(
               "Reports either detailed or summarized Comfy Kitchen kernel dispatch info, and other insightful aspects, "
               "such as hardware/stack performance." "The custom node has the ability to persistently store the reports, "
               "just by simply being wired anywhere within functional ComfyUI workflows (which includes placement "
               "even after any 'Save' Image/Video/Audio/Latent/etc node that has an output to hook into). " 
               "Project also features flag/env var option to display summary or detailed reports via "
               "terminal, without involvement of the custom node." "Intentionally very lightweight, " 
               "utilizing just psutil, WDDM, and ultimately optimized and intended for AMD GPU users "
               "on Windows."
            ),
            inputs=[
                io.AnyType.Input(
                    "input",
                    optional=True,
                    tooltip="Optional passthrough value. Connect the image, video, audio, latent, or other value you want to carry through unchanged.",
                ),
                *cls._inputs_schema(),
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
            outputs=[
                io.AnyType.Output(display_name="output", tooltip="Unchanged passthrough value."),
                io.String.Output(display_name="report", tooltip="Optional text report. Connect to Preview as Text or another STRING node if you prefer a separate report display."),
            ],
            is_output_node=True,
        )

    @classmethod
    def execute(cls, input=None, embed_metadata=True, terminal_report_level="summary", show_terminal_report=False, node_report_level="summary", show_node_report=False, report_text=""):
        extra_pnginfo = getattr(cls.hidden, "extra_pnginfo", None)
        unique_id = getattr(cls.hidden, "unique_id", None)
        report = cls._execute_common(
            embed_metadata,
            terminal_report_level,
            show_terminal_report,
            node_report_level,
            extra_pnginfo,
            unique_id,
        )
        return io.NodeOutput(input, report, ui=ui.PreviewText(report))

class LegacyCKDispatchReport(_TelemetryNodeMixin, io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="CKDispatchReport",
            display_name="CK Telemetry Report (Legacy Wrapper)",
            category="diagnostics/comfy kitchen",
            search_aliases=["ckdispatchreport", "legacy ck telemetry"],
            description="Legacy node structure retained so early development workflows continue to load natively.",
            inputs=[
                io.AnyType.Input("input", optional=True, tooltip="Legacy fallback passthrough."),
                *cls._inputs_schema()
            ],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
            outputs=[
                io.AnyType.Output(display_name="output"),
                io.String.Output(display_name="report"),
            ],
            is_deprecated=True,
            is_output_node=True,
        )

    @classmethod
    def execute(cls, input=None, embed_metadata=True, terminal_report_level="summary", show_terminal_report=True, node_report_level="detailed", show_node_report=True, report_text=""):
        extra_pnginfo = getattr(cls.hidden, "extra_pnginfo", None)
        unique_id = getattr(cls.hidden, "unique_id", None)
        report = cls._execute_common(
            embed_metadata,
            terminal_report_level,
            show_terminal_report,
            node_report_level,
            extra_pnginfo,
            unique_id,
        )
        return io.NodeOutput(input, report, ui=ui.PreviewText(report))    


class ComfyKitchenTelemetryImage(_TelemetryNodeMixin, io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyKitchenTelemetryImage",
            display_name="Comfy Kitchen Dispatch Report (Legacy Image)",
            category="diagnostics/comfy kitchen",
            search_aliases=["comfy_kitchen_dispatch_telemetry", "legacy comfy kitchen telemetry image"],
            description="Legacy image-only form retained so older workflows continue to load.",
            inputs=[io.Image.Input("image", tooltip="Legacy image passthrough."), *cls._inputs_schema()],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
            outputs=[
                io.Image.Output(display_name="image"),
                io.String.Output(display_name="report", tooltip="Optional text report; connect to Preview as Text or another STRING node."),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, image, embed_metadata, terminal_report_level, show_terminal_report, node_report_level, show_node_report, report_text):
        extra_pnginfo = getattr(cls.hidden, "extra_pnginfo", None)
        unique_id = getattr(cls.hidden, "unique_id", None)
        report = cls._execute_common(
            embed_metadata,
            terminal_report_level,
            show_terminal_report,
            node_report_level,
            extra_pnginfo,
            unique_id,
        )
        return io.NodeOutput(image, report, ui=ui.PreviewText(report))


class ComfyKitchenTelemetryVideo(_TelemetryNodeMixin, io.ComfyNode):
    @classmethod
    def define_schema(cls):
        return io.Schema(
            node_id="ComfyKitchenTelemetryVideo",
            display_name="Comfy Kitchen Dispatch Report (Legacy Video)",
            category="diagnostics/comfy kitchen",
            search_aliases=["comfy_kitchen_dispatch_telemetry video", "legacy comfy kitchen telemetry video"],
            description="Legacy video-only form retained so older workflows continue to load.",
            inputs=[io.Video.Input("video", tooltip="Legacy video passthrough."), *cls._inputs_schema()],
            hidden=[io.Hidden.prompt, io.Hidden.extra_pnginfo, io.Hidden.unique_id],
            outputs=[
                io.Video.Output(display_name="video"),
                io.String.Output(display_name="report", tooltip="Optional text report; connect to Preview as Text or another STRING node."),
            ],
            is_deprecated=True,
        )

    @classmethod
    def execute(cls, video, embed_metadata, terminal_report_level, show_terminal_report, node_report_level, show_node_report, report_text):
        extra_pnginfo = getattr(cls.hidden, "extra_pnginfo", None)
        unique_id = getattr(cls.hidden, "unique_id", None)
        report = cls._execute_common(
            embed_metadata,
            terminal_report_level,
            show_terminal_report,
            node_report_level,
            extra_pnginfo,
            unique_id,
        )
        return io.NodeOutput(video, report, ui=ui.PreviewText(report))


class ComfyKitchenTelemetryExtension(ComfyExtension):
    @override
    async def get_node_list(self) -> list[type[io.ComfyNode]]:
        return [
            ComfyKitchenTelemetryReport, 
            LegacyCKDispatchReport, 
            ComfyKitchenTelemetryImage, 
            ComfyKitchenTelemetryVideo
        ]


async def comfy_entrypoint() -> ComfyKitchenTelemetryExtension:
    env_numeric_level = _parse_env_level()
    
    if env_numeric_level > 0:
        level_string = "DETAILED" if env_numeric_level == 2 else "SUMMARY"
        
        _ANSI_REDDISH = "\x1b[38;5;166m"
        _ANSI_RESET = "\x1b[0m"
        print(f"{_ANSI_REDDISH}[HIP+] CK-TELEMETRY [˻˺]{_ANSI_RESET} - {level_string} Terminal report level enabled via COMFY_KITCHEN_TELEMETRY.")
        
    return ComfyKitchenTelemetryExtension()


def gpu_util_avg_string(gpu_data: dict[str, Any]) -> str:
    avg = gpu_data.get("average")
    if avg is not None and float(avg) > 0.0:
        return f"{avg:.0f}% Avg"
    return "-"