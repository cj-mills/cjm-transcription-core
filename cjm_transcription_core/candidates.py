"""Candidate (capability, MODEL)-instance enumeration for the comparison screen.

The comparison pair for a run is any combination of (capability, MODEL) instances,
not a capability pair (DEC db200725): the whisper capability hosts the full Whisper
family, voxtral hosts mini (fits the GPU) and small (system-memory only), and future
transcription capabilities follow the same multi-model shape. This module derives
the candidate space from the capability MANIFESTS alone — the same files the
substrate discovers from — so the TUI offers choices without loading anything: a
capability qualifies by its structural surface (a `transcribe` method, mirroring
the transcription adapter's protocol match), and its model axis is any enum config
property whose name contains `model` (the key is NOT uniform across capabilities —
whisper: `model`, voxtral-hf: `model_id` — hence data-driven, no hard-coded key).
"""

import json
from pathlib import Path
from typing import Any, Dict, List, Optional


def transcription_manifests(
    manifests_dir: str,  # Capability manifests directory (the core CLI's --manifests-dir)
) -> Dict[str, Dict[str, Any]]:  # capability name -> its manifest `code` section
    """Enumerate installed transcription capabilities from their manifest files.

    The `transcribe` specialization of manifests_with_method (kept as the
    candidate-space entry point; see that symbol for the surface-match rationale).
    """
    return manifests_with_method(manifests_dir, "transcribe")


def model_axis(
    code: Dict[str, Any],  # One capability's manifest `code` section
) -> Optional[Dict[str, Any]]:  # {"key", "options", "default"} or None (no model choice)
    """Find a capability's MODEL config axis in its config_schema.

    The key is not uniform (whisper: `model`; voxtral-hf: `model_id`), so the
    axis derives from the data: the first enum property whose name contains
    `model`. None = the capability offers no model choice; it still shows as a
    single candidate (its default instance).
    """
    props = ((code.get("config_schema") or {}).get("properties") or {})
    for key, schema in props.items():
        if "model" in key and isinstance(schema, dict) and schema.get("enum"):
            return {"key": key, "options": list(schema["enum"]),
                    "default": schema.get("default")}
    return None


def instance_id_for(
    capability: str,  # Capability name (cjm-capability-*)
    model: str,       # Model option value (may carry '/', '.', etc.)
) -> str:  # CR-10-safe instance id: [A-Za-z0-9_-]{1,64}
    """Derive an addressable instance id for a non-default (capability, MODEL) pick.

    `{short-name}--{sanitized model tail}` — readable in manifests/journals and
    valid against the manager's instance-id pattern (HF ids keep only their repo
    tail; every non-[A-Za-z0-9_-] character becomes '-').
    """
    short = capability.removeprefix("cjm-capability-")
    tail = "".join(c if (c.isalnum() or c in "_-") else "-" for c in model.split("/")[-1])
    return f"{short}--{tail}"[:64]


def candidate_directives(
    manifests_dir: str,  # Capability manifests directory
) -> List[Dict[str, Any]]:  # One row per (capability, MODEL): load directive + display fields
    """Expand every installed transcription capability into its candidate space.

    Each row is a parse_transcriber_spec-shaped load directive ({"capability",
    "instance_id", "config"}) plus display fields ("model", "default"). The
    DEFAULT model keeps the bare capability name as its instance id (stage-5
    behavior — cache-continuous with prior runs); every other model gets its own
    addressable instance id + a config override on the capability's model axis.
    """
    rows: List[Dict[str, Any]] = []
    for name, code in transcription_manifests(manifests_dir).items():
        axis = model_axis(code)
        if axis is None:
            rows.append({"capability": name, "instance_id": name, "config": {},
                         "model": None, "default": True})
            continue
        for option in axis["options"]:
            default = (option == axis["default"])
            rows.append({
                "capability": name,
                "instance_id": name if default else instance_id_for(name, option),
                "config": ({} if default else {axis["key"]: option}),
                "model": option,
                "default": default,
            })
    return rows


def spec_string(
    directive: Dict[str, Any],  # A load directive: {"capability", "instance_id", "config"}
) -> str:  # The equivalent --transcriber spec (parse_transcriber_spec inverse)
    """Render a load directive back to the core CLI's --transcriber grammar.

    The confirmed run is handed off as a plain `cjm-transcription-core run`
    invocation (printed for reproducibility), so every TUI choice must round-trip
    through NAME[@INSTANCE_ID][:key=value,...]. Bools render lowercase to match
    the parser's true/false coercion.
    """
    name = directive["capability"]
    iid = directive["instance_id"]
    config = directive.get("config") or {}
    if iid == name and not config:
        return name
    spec = f"{name}@{iid}"
    if config:
        def render(v: Any) -> str:
            if isinstance(v, bool):
                return "true" if v else "false"
            return str(v)
        spec += ":" + ",".join(f"{k}={render(v)}" for k, v in config.items())
    return spec


def manifests_with_method(
    manifests_dir: str,  # Capability manifests directory (the core CLI's --manifests-dir)
    method: str,         # Structural-surface method that identifies the role
) -> Dict[str, Dict[str, Any]]:  # capability name -> its manifest `code` section
    """Enumerate installed capabilities whose structural surface lists `method`.

    Capabilities qualify by SURFACE, not by name — the same signal the
    substrate's adapter auto-binding matches against a task protocol, read
    cheaply off the manifest json (no worker spawn). Role keys in use:
    `transcribe` (transcription tools), `add_nodes` (graph storage),
    `get_system_status` (system monitor). Adapter unit manifests carry no
    `code` section and are skipped; unreadable files are skipped rather than
    failing enumeration.
    """
    out: Dict[str, Dict[str, Any]] = {}
    for f in sorted(Path(manifests_dir).glob("*.json")):
        try:
            manifest = json.loads(f.read_text())
        except (OSError, ValueError):
            continue
        code = manifest.get("code") if isinstance(manifest, dict) else None
        if not isinstance(code, dict):
            continue
        methods = ((code.get("structural_surface") or {}).get("methods") or [])
        if any(m.get("name") == method for m in methods):
            out[code.get("name") or f.stem] = code
    return out


def discover_capability(
    manifests_dir: str,  # Capability manifests directory
    method: str,         # Surface method that identifies the role
) -> Optional[str]:  # First matching capability name (sorted), or None
    """Pick a DEFAULT capability for a role by surface match.

    Journaling-by-default (drive-1 finding): when the runtime has a graph
    storage (`add_nodes`) or a monitor (`get_system_status`) installed, the TUI
    should use it without being told — forgetting must take an explicit opt-out,
    not a forgotten flag. Sorted-first keeps the pick deterministic when several
    qualify; the operator's persisted choice (state.py) wins over discovery.
    """
    names = sorted(manifests_with_method(manifests_dir, method))
    return names[0] if names else None
