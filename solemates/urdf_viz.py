"""
solemates/urdf_viz.py

Logs the full articulated BracketBot URDF into Rerun so the robot arm
visually animates in the 3D world panel as joint angles change.

Mesh path resolution (chopped_urdf_v2 specific)
------------------------------------------------
The URDF references meshes as:
  package://chopped_urdf_v2/meshes/Base_Extrusion__Base_Extrusion.stl

But the actual files on disk are GLB (Draco-compressed) at:
  chopped_urdf_v2/draco/Base_Extrusion__Base_Extrusion.glb

Resolution strategy:
  1. Build a disk index (stem → path) covering draco/, meshes/, and all
     subdirs of the package root, including cross-format matching (.stl
     reference → .glb on disk).
  2. Look up by full stem first ("Base_Extrusion__Base_Extrusion"),
     then by the first __ component ("base_extrusion") as a fallback.
  3. Standard path candidates (relative, package-relative) tried first.
"""

from __future__ import annotations

import math
import xml.etree.ElementTree as ET
from pathlib import Path
from typing import Optional

import numpy as np

# ---------------------------------------------------------------------------
# Module state
# ---------------------------------------------------------------------------
_meshes_logged   = False
_joint_tree: list[dict] = []
_link_meshes: dict[str, list[Path]] = {}
_disk_index: dict[str, Path] = {}   # lower-case stem → resolved Path

_SUPPORTED = {".obj", ".stl", ".glb", ".gltf"}


# ---------------------------------------------------------------------------
# Public API
# ---------------------------------------------------------------------------

def init_urdf(urdf_path: Path, viz_module) -> bool:
    """Parse the URDF and log static mesh geometry. Call once after viz.init()."""
    global _joint_tree, _link_meshes, _meshes_logged, _disk_index

    if not viz_module._ready():
        return False
    if not urdf_path.is_file():
        print(f"[urdf_viz] URDF not found: {urdf_path}")
        return False

    package_root = urdf_path.parent.parent   # .../chopped_urdf_v2/
    _disk_index  = _build_disk_index(package_root)
    print(f"[urdf_viz] Disk index: {len(_disk_index)} stems under {package_root.name}/")

    _joint_tree, _link_meshes = _parse_urdf(urdf_path)

    found = sum(len(v) for v in _link_meshes.values())
    print(f"[urdf_viz] {len(_joint_tree)} joints, {found} meshes resolved")

    _log_static_meshes(viz_module)
    _meshes_logged = True
    return True


def log_robot_state(
    joint_positions: dict[str, float],
    step: int,
    viz_module,
) -> None:
    """Log Transform3D for every link at the current joint configuration."""
    if not viz_module._ready() or not _joint_tree:
        return

    rr = viz_module._rr
    dt = viz_module._dt
    rr.set_time("step", sequence=step)

    for joint in _joint_tree:
        T = _joint_transform(joint, joint_positions.get(joint["name"], 0.0))
        rr.log(
            f"world/robot/{joint['child']}",
            rr.Transform3D(
                translation=T[:3, 3].tolist(),
                quaternion=dt.Quaternion(xyzw=_mat_to_quat(T[:3, :3])),
                relation=rr.TransformRelation.ChildFromParent,
            ),
        )


# ---------------------------------------------------------------------------
# Disk index
# ---------------------------------------------------------------------------

def _build_disk_index(package_root: Path) -> dict[str, Path]:
    """
    Walk the package tree and index every supported mesh file by stem.

    Two keys per file:
      full stem  (lower)  : "base_extrusion__base_extrusion"
      short stem (lower)  : "base_extrusion"  (part before first __)

    Full-stem key wins when both match, so mirrored variants resolve correctly:
      Bicep__Bicep          → Bicep__Bicep.glb
      Bicep__Bicep_Mirrored → Bicep__Bicep_Mirrored.glb
    """
    index: dict[str, Path] = {}

    # Prioritise draco/ (GLB), then everything else
    search_dirs = []
    draco = package_root / "draco"
    if draco.is_dir():
        search_dirs.append(draco)
    search_dirs.append(package_root)   # rglob catches all subdirs anyway

    for d in search_dirs:
        for suffix in _SUPPORTED:
            for f in d.rglob(f"*{suffix}"):
                full_stem  = f.stem.lower()
                short_stem = full_stem.split("__")[0]
                # Register short stem only if not already set by a full-stem match
                index.setdefault(short_stem, f)
                # Full stem always wins (may overwrite short-stem value)
                index[full_stem] = f

    return index


# ---------------------------------------------------------------------------
# Mesh path resolution
# ---------------------------------------------------------------------------

def _resolve_mesh(filename: str, urdf_path: Path) -> Optional[Path]:
    """
    Resolve a URDF mesh filename to an absolute path.

    Priority:
      1. Standard path candidates (relative to urdf dir / package root)
      2. Disk-index lookup by full stem (cross-format: .stl ref → .glb on disk)
      3. Disk-index lookup by short stem (first __ component)
    """
    urdf_dir     = urdf_path.parent
    package_root = urdf_dir.parent

    # Strip package:// prefix variants
    stripped = filename
    for prefix in (
        f"package://{package_root.name}/",
        f"package://{package_root.parent.name}/{package_root.name}/",
        "package://",
    ):
        if stripped.startswith(prefix):
            stripped = stripped[len(prefix):]
            break

    rel       = Path(stripped)
    name_only = rel.name
    stem      = rel.stem.lower()

    # 1. Standard candidates
    for candidate in (
        urdf_dir     / rel,
        package_root / rel,
        package_root / "meshes"  / name_only,
        package_root / "draco"   / name_only,
        package_root.parent / rel,
    ):
        try:
            r = candidate.resolve()
            if r.is_file() and r.suffix.lower() in _SUPPORTED:
                return r
        except (OSError, ValueError):
            continue

    # 2. Full-stem cross-format lookup
    if stem in _disk_index:
        return _disk_index[stem]

    # 3. Short-stem fallback
    short = stem.split("__")[0]
    if short in _disk_index:
        return _disk_index[short]

    return None


# ---------------------------------------------------------------------------
# URDF parsing
# ---------------------------------------------------------------------------

def _parse_urdf(urdf_path: Path):
    root = ET.parse(urdf_path).getroot()

    raw_joints = []
    for elem in root.findall("joint"):
        origin    = elem.find("origin")
        axis_elem = elem.find("axis")
        raw_joints.append({
            "name":   elem.get("name", ""),
            "kind":   elem.get("type", "fixed"),
            "parent": elem.find("parent").get("link"),
            "child":  elem.find("child").get("link"),
            "xyz":    _vec3(origin.get("xyz") if origin is not None else None),
            "rpy":    _vec3(origin.get("rpy") if origin is not None else None),
            "axis":   _vec3(
                axis_elem.get("xyz") if axis_elem is not None else None,
                default=(1.0, 0.0, 0.0),
            ),
        })

    link_meshes: dict[str, list[Path]] = {}
    for link_elem in root.findall("link"):
        link_name = link_elem.get("name", "")
        paths = []
        for visual in link_elem.findall("visual"):
            geom = visual.find("geometry")
            if geom is None:
                continue
            mesh_elem = geom.find("mesh")
            if mesh_elem is None:
                continue
            resolved = _resolve_mesh(mesh_elem.get("filename", ""), urdf_path)
            if resolved:
                paths.append(resolved)
        if paths:
            link_meshes[link_name] = paths

    return _topo_sort(raw_joints), link_meshes


def _topo_sort(joints: list[dict]) -> list[dict]:
    children_of: dict[str, list[dict]] = {}
    all_children = {j["child"] for j in joints}
    for j in joints:
        children_of.setdefault(j["parent"], []).append(j)
    roots = [p for p in children_of if p not in all_children]
    if not roots:
        return joints
    ordered, queue = [], list(roots)
    while queue:
        node = queue.pop(0)
        for cj in children_of.get(node, []):
            ordered.append(cj)
            queue.append(cj["child"])
    return ordered


# ---------------------------------------------------------------------------
# Transform math
# ---------------------------------------------------------------------------

def _rpy_to_mat(rpy: np.ndarray) -> np.ndarray:
    roll, pitch, yaw = rpy
    cr, sr = math.cos(roll),  math.sin(roll)
    cp, sp = math.cos(pitch), math.sin(pitch)
    cy, sy = math.cos(yaw),   math.sin(yaw)
    return np.array([
        [cy*cp,  cy*sp*sr - sy*cr,  cy*sp*cr + sy*sr],
        [sy*cp,  sy*sp*sr + cy*cr,  sy*sp*cr - cy*sr],
        [-sp,    cp*sr,             cp*cr            ],
    ], dtype=float)


def _axis_angle_mat(axis: np.ndarray, angle: float) -> np.ndarray:
    n = np.linalg.norm(axis)
    if n < 1e-12:
        return np.eye(3)
    axis = axis / n
    x, y, z = axis
    c, s, one = math.cos(angle), math.sin(angle), 1.0 - math.cos(angle)
    return np.array([
        [c + x*x*one,    x*y*one - z*s,  x*z*one + y*s],
        [y*x*one + z*s,  c + y*y*one,    y*z*one - x*s],
        [z*x*one - y*s,  z*y*one + x*s,  c + z*z*one  ],
    ], dtype=float)


def _joint_transform(joint: dict, value: float) -> np.ndarray:
    T = np.eye(4)
    R = _rpy_to_mat(joint["rpy"])
    T[:3, :3] = R
    T[:3,  3] = joint["xyz"]
    if joint["kind"] in ("revolute", "continuous"):
        T[:3, :3] = R @ _axis_angle_mat(np.array(joint["axis"]), value)
    elif joint["kind"] == "prismatic":
        T[:3, 3] = joint["xyz"] + R @ (np.array(joint["axis"]) * value)
    return T


def _mat_to_quat(R: np.ndarray) -> list[float]:
    trace = R[0,0] + R[1,1] + R[2,2]
    if trace > 0:
        s = 0.5 / math.sqrt(trace + 1.0)
        return [(R[2,1]-R[1,2])*s, (R[0,2]-R[2,0])*s,
                (R[1,0]-R[0,1])*s, 0.25/s]
    elif R[0,0] > R[1,1] and R[0,0] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[0,0] - R[1,1] - R[2,2])
        return [0.25*s, (R[0,1]+R[1,0])/s,
                (R[0,2]+R[2,0])/s, (R[2,1]-R[1,2])/s]
    elif R[1,1] > R[2,2]:
        s = 2.0 * math.sqrt(1.0 + R[1,1] - R[0,0] - R[2,2])
        return [(R[0,1]+R[1,0])/s, 0.25*s,
                (R[1,2]+R[2,1])/s, (R[0,2]-R[2,0])/s]
    else:
        s = 2.0 * math.sqrt(1.0 + R[2,2] - R[0,0] - R[1,1])
        return [(R[0,2]+R[2,0])/s, (R[1,2]+R[2,1])/s,
                0.25*s, (R[1,0]-R[0,1])/s]


def _vec3(text: Optional[str], default=(0.0, 0.0, 0.0)) -> np.ndarray:
    if text:
        return np.array([float(v) for v in text.split()], dtype=float)
    return np.array(default, dtype=float)


# ---------------------------------------------------------------------------
# Static mesh logging
# ---------------------------------------------------------------------------

def _log_static_meshes(viz_module) -> None:
    """Log resolved GLB/OBJ meshes once as static entities. No dot placeholders."""
    rr     = viz_module._rr
    logged = 0
    for link_name, mesh_paths in _link_meshes.items():
        for i, mesh_path in enumerate(mesh_paths):
            try:
                rr.log(
                    f"world/robot/{link_name}/mesh_{i}",
                    rr.Asset3D(path=str(mesh_path)),
                    static=True,
                )
                logged += 1
            except Exception as exc:
                print(f"[urdf_viz] skipped {mesh_path.name}: {exc}")
    if logged:
        print(f"[urdf_viz] {logged} mesh(es) logged as static geometry in world/robot/")
    else:
        print("[urdf_viz] No meshes logged — run inspect_urdf.py to diagnose")