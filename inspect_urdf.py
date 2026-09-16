"""
inspect_urdf.py  —  run this from Battle-Bots root to show
exactly what mesh filenames the URDF contains and whether they resolve.

"""
import xml.etree.ElementTree as ET
from pathlib import Path

URDF = Path("chopped_urdf_v2/chopped_urdf_v2/urdf/chopped_urdf_v2.urdf")

if not URDF.is_file():
    print(f"ERROR: URDF not found at {URDF.resolve()}")
    raise SystemExit(1)

root = ET.parse(URDF).getroot()
urdf_dir    = URDF.parent           # .../urdf/
mesh_parent = urdf_dir.parent       # .../chopped_urdf_v2/
grandparent = mesh_parent.parent    # .../

print(f"URDF:      {URDF.resolve()}")
print(f"urdf_dir:  {urdf_dir.resolve()}")
print(f"parent:    {mesh_parent.resolve()}")
print()

# List every STL/OBJ/GLB in the package tree
print("=== Mesh files found on disk ===")
found_on_disk = list(grandparent.rglob("*.stl")) + \
                list(grandparent.rglob("*.obj")) + \
                list(grandparent.rglob("*.glb"))
for f in sorted(found_on_disk)[:30]:
    print(f"  {f.relative_to(grandparent)}")
if not found_on_disk:
    print("  (none found)")

print()
print("=== URDF mesh filename references ===")
seen = set()
for link in root.findall("link"):
    for vis in link.findall("visual"):
        geom = vis.find("geometry")
        if geom is None:
            continue
        mesh = geom.find("mesh")
        if mesh is None:
            continue
        fname = mesh.get("filename", "")
        link_name = link.get("name", "")
        if fname not in seen:
            seen.add(fname)
            # Try resolving it
            stripped = fname
            for prefix in ("package://chopped_urdf_v2/", "package://"):
                if stripped.startswith(prefix):
                    stripped = stripped[len(prefix):]
                    break
            name_only = Path(stripped).name
            candidates = [
                urdf_dir    / Path(stripped),
                mesh_parent / Path(stripped),
                mesh_parent / "meshes" / name_only,
                grandparent / Path(stripped),
            ]
            resolved = next((c.resolve() for c in candidates if c.is_file()), None)
            status = f"OK  -> {resolved.relative_to(grandparent)}" if resolved else "MISSING"
            print(f"  {link_name:30s}  {fname:55s}  [{status}]")

print()
print("=== First 8 joints ===")
for j in root.findall("joint")[:8]:
    print(f"  {j.get('name'):35s}  type={j.get('type')}")