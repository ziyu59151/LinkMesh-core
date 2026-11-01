# LinkMesh usage and M3D-C1 mesh workflow

Repository-specific environment instructions are in
[`docs/ENVIRONMENT.md`](docs/ENVIRONMENT.md). The virtual environment is the
default deployment path; Conda is optional.

## Installation and dependencies

LinkMesh generates triangular meshes in the R-Z plane, with coordinates and mesh sizes in metres. It supports mesh-only plots, TRIMEG HDF5, and M3D-C1 model TXT plus PUMI SMB output.

Clone the source with the command below:

```bash
git clone https://github.com/ziyu59151/LinkMesh-core.git
cd LinkMesh-core
python3 -m venv .venv
source .venv/bin/activate
python -m pip install --upgrade pip
python -m pip install -e '.[all]'
linkmesh-check core
```

`pyproject.toml` declares `openfusiontoolkit==26.6`; pip installs OFT and its compiled libraries. An OFT source checkout is not required. LinkMesh uses OFT's Python meshing interface, including version-specific boundary handling, so retain the 26.6 version pin. If no compatible wheel exists for your platform, build and install OFT 26.6 separately. Do not add its raw source directory to PYTHONPATH.

Python code needs no compilation. M3D-C1 SMB output additionally requires the compiled `linkmesh-scorec-helper`, which links SCOREC/PUMI 2.2.9 and MPI. It does not require M3DC1 source or `m3dc1_scorec`.

```bash
cmake -S native/linkmesh_scorec_helper -B build/helper \
  -DCMAKE_BUILD_TYPE=Release \
  -DCMAKE_PREFIX_PATH=/path/to/pumi \
  -DCMAKE_INSTALL_PREFIX="$VIRTUAL_ENV" \
  -DCMAKE_INSTALL_RPATH=/path/to/pumi/lib
cmake --build build/helper -j 4
cmake --install build/helper
export LINKMESH_SCOREC_HELPER="$VIRTUAL_ENV/bin/linkmesh-scorec-helper"
linkmesh-check all
```

Use compatible MPI builds for PUMI, the helper, partitioner and solver. The checker validates imports and native library resolution; it does not test M3D-C1 initialization or time stepping.

## Typical installation layout

- Source: `$LINKMESH_ROOT`
- Python environment: `$VIRTUAL_ENV`
- Commands and native helper: `$VIRTUAL_ENV/bin`
- Native build directory: `$LINKMESH_ROOT/build/helper`
- PUMI prefix: `/path/to/pumi`

```bash
export PATH="$VIRTUAL_ENV/bin:$PATH"
linkmesh-check all
linkmesh --help
```

The helper was built with system OpenMPI and an install RPATH pointing to the PUMI lib directory. The original binary-release helper required Intel MPI and could not run in this environment; use the rebuilt helper.

When moving the installation to another host, rebuild the native SMB helper
against that host's PUMI and MPI installation and set the install RPATH
accordingly.

## Boundary file format

Use a count followed by that many R/Z pairs:

```text
4
1.0 -0.2
1.4 -0.2
1.4  0.2
1.0  0.2
```

The contour closes cyclically. Do not append an identical copy of the first point. Duplicate or nearly duplicate closure points can cause meshing failure. Inspect the geometry before changing any nonidentical points.

## Extracting the LCFS from GEQDSK

The tested explicit-boundary workflow uses `readg.py` from the local M3DC1 Python package. Copy the equilibrium into a new case directory and run the following there:

```bash
cp /path/to/geqdsk ./geqdsk
python - <<'PY'
import importlib.util
from pathlib import Path
import numpy as np
reader = Path('/path/to/M3DC1/unstructured/idl/python/m3dc1/readg.py')
spec = importlib.util.spec_from_file_location('geqdsk_reader', reader)
module = importlib.util.module_from_spec(spec)
spec.loader.exec_module(module)
g = module.readg('geqdsk')
points = np.column_stack((g['rbbbs'], g['zbbbs']))
if len(points) > 1 and np.array_equal(points[0], points[-1]):
    points = points[:-1]
if len(points) < 3 or not np.isfinite(points).all():
    raise ValueError('Invalid LCFS boundary')
with open('boundary.txt', 'w') as f:
    f.write(f'{len(points)}\n')
    np.savetxt(f, points, fmt='%.16e')
print(f'Wrote {len(points)} boundary points')
PY
```

The tested production GEQDSK contained 127 points, with an exactly repeated closing point; the boundary file retained 126 distinct entries.

```bash
linkmesh --no-config --target m3dc1 \
  --boundary-source array_file --array-file boundary.txt \
  --gfile geqdsk --plot-lcfs --mesh-dx 0.06 \
  --model diiid.txt --smb diiid.smb \
  --plot mesh.png --quality-plot quality.png --verbose \
  > generate.log 2>&1
```

`--gfile geqdsk --plot-lcfs` overlays the reference LCFS on the plot; it does not replace the boundary supplied by `--array-file`. With the serial helper, `--smb diiid.smb` generated the actual file `diiid0.smb`.

The CLI also supports `--boundary-source lcfs --gfile geqdsk`. The explicit extraction above was tested successfully after a direct LCFS attempt rejected the duplicated closure point.

## Translating legacy meshgen input files

LinkMesh now reads the old `input`/`input2` keyword format directly:

```bash
linkmesh --input /path/to/input --plot mesh.png
linkmesh --input /path/to/input2 --plot mesh.png
linkmesh --input /path/to/modelType1/input --mesh-dx 0.02 --plot mesh.png
```

`--legacy-input` is an alias for `--input`. This mode disables TOML discovery and defaults to M3D-C1 output. Explicit CLI mesh size, target and output filenames override the input values. Resolve `inFile` relative to the input file; write outputs into the working directory. Quoted paths and `!` comments are supported; duplicate or unknown active keywords are rejected. `--input` cannot be combined with `--config`, `--no-config`, `--boundary-source` or `--array-file`.

The separate TOML configuration format requires top-level `schema_version = 1`; consult `linkmesh --help` for its sections and options.

For the tested legacy settings:

| Legacy setting | LinkMesh interpretation |
| --- | --- |
| `modelType 1` | Read the counted R-Z boundary named by `inFile`; use `array_file` |
| `meshSize` | `--mesh-dx`, in metres |
| `outFile diiid` | `--model diiid.txt --smb diiid.smb` |
| `modelType 0` | Generate an analytic contour from the five `vacuumParams` |

For compatibility cases with `modelType=0`, `vacuumParams` define the analytic contour even when `useVacuumParams=0`; the `inFile` value is ignored. For parameters `(a,b,c,d,e)`, generate:

```python
theta = np.linspace(0, 2*np.pi, 400, endpoint=False)
R = a + b*np.cos(theta + c*np.sin(theta))
Z = d + e*np.sin(theta)
```

Write those coordinates as a counted boundary file. The 400-point contour approximates the analytic shape; the original legacy generator used a periodic spline through its sampled points, so the geometries and entity tags are not identical. The built-in reader requires `adjustVacuumParams=0` and supports single-region modelType=0/1 only. It samples the analytic contour with at least 400 vertices, or numVacuumPts if larger. modelType=2/3 and unsupported active settings fail explicitly.

## Mesh inspection and refinement

Inspect `generate.log`, `mesh.png`, and `quality.png`. Check boundary location, minimum angle, shape quality, invalid triangles, and mesh resolution in the region of interest. A coarse boundary can deviate from the original LCFS even when its input points were extracted correctly. A reference-LCFS containment warning should be assessed geometrically rather than ignored.

Reduce `--mesh-dx` for global refinement. For local refinement, the CLI accepts repeated `--local-refinement R,Z,DX,W` options, all in metres. DX must be smaller than the baseline size and W positive. Consult the CLI help before selecting values; this option was not exercised in the recorded cases.

The tested case scripts capture `mesh_arrays.npz` and `quality_metrics.json` through the Python mesher interface. These are script-generated diagnostics, not default CLI outputs. Boundary validation projects each original LCFS point onto the mesh boundary segments and records the shortest distance.

| Tested case | Size (m) | Nodes | Triangles | Minimum angle |
| --- | ---: | ---: | ---: | ---: |
| GEQDSK LCFS | 0.01 | 4742 | 9288 | 24.56 degrees |
| Parent `input`, analytic contour | 0.06 | 923 | 1756 | 25.16 degrees |
| `bigmesh/input2`, analytic ellipse | 0.07 | 4260 | 8314 | 25.50 degrees |
| `modelType=1`, GEQDSK boundary | 0.06 | 144 | 254 | 29.56 degrees |

No degenerate triangles, duplicate triangles, or nonmanifold edges were found in these cases. LCFS-point-to-mesh-boundary maximum distances were approximately 0.24 mm for the 0.01 m LCFS mesh and 1.80 mm for the 0.06 m mesh. These metrics do not establish solver convergence.

Cases and their reproducible scripts can be stored in a separate case
directory outside the source tree.

## Partitioning and using the mesh in M3D-C1

Keep the generated model TXT paired with its SMB files. Do not substitute an unrelated legacy model with different entity tags. A 64-rank simulation needs 64 mesh partitions, not just the serial mesh.

Use a compatible SMB partitioner for the target rank count:

```bash
<mesh-partitioner> diiid.smb diiid-part.smb 64
```

For a prefix-based splitter, this means `diiid0.smb` as input and `diiid-part0.smb` through `diiid-part63.smb` as outputs. Compatibility with the generated LinkMesh model and mesh must be checked.

Set the corresponding C1input filenames:

```text
mesh_model = 'diiid.txt'
mesh_filename = 'diiid-part.smb'
```

Verify all 64 partitions are nonempty, then test mesh/model loading and equilibrium initialization in a separate run directory. Check debug-out and solver output before running the production simulation. Follow `M3DC1-RUN.md` for equilibrium transfer, C1input tuning, Slurm launch and first-step checks. LinkMesh creates geometry and meshes; it does not generate a new equilibrium.

## Reading m3dc1_mfmgen multi-region input

The `--mfmgen-input` mode translates the multi-region keyword format directly. Relative `bdryFile` paths resolve from the current working directory, so copy the boundary file there before running:

```bash
cp /path/to/first_wall.pts .
linkmesh --mfmgen-input /path/to/input
```

Supported keywords are `numBdry`, repeated `bdryFile FILE LOOP_ID SIZE`, `thickWall ENABLE SOURCE_ID NEW_ID WIDTH`, `useVacuum` modes 0 and 2, `vacuumParams`, optional `numVacuumPts`, `numFace`, repeated `faceBdry N LOOP_ID... SIZE`, and `outFile`. Boundary files may have raw R Z rows or a count header. Faces must exactly match nested loop boundaries. A positive `thickWall` width offsets the source polygon outward using a miter join. Loop and vacuum boundaries are sampled to their requested spacing; the vacuum follows R=R0+a cos(theta+delta sin(theta)), Z=Z0+kappa*a sin(theta).

The translator converts face loop sets to LinkMesh region IDs after OFT inspection, applies each `faceBdry` mesh size to its matching Region, and relabels the output regions in `faceBdry` order so the model IDs follow the input face order. It then writes `<outFile>.txt`, `<outFile>0.smb`, `<outFile>_mesh.png`, and `<outFile>_quality.png` in the current directory. The tutorial example generated 5,042 nodes and 9,921 triangles in three regions, with no degenerate, duplicate, or nonmanifold triangles. Its smallest angle was 20.57 degrees; inspect the quality plot for your geometry before using the mesh.

`layeredMesh` and `meshGradationRate` are rejected because their mfmgen behaviors have no direct equivalent in this translator. Only nested, nonintersecting loops are accepted by the current LinkMesh region workflow. This is a geometry and mesh conversion; compare model IDs and equilibrium setup before a production M3D-C1 run.

## Other outputs

```bash
linkmesh --no-config --target mesh_only \
  --boundary-source array_file --array-file boundary.txt \
  --mesh-dx 0.04 --plot mesh.png

linkmesh --no-config --target trimeg \
  --boundary-source array_file --array-file boundary.txt \
  --mesh-dx 0.04 --hdf5 mesh.h5
```

For multiple counted contours, consult `linkmesh regions --help` and its inspect/generate commands. The single-LCFS examples above do not add vacuum or wall regions automatically.

Direct-input verification reproduced 923 nodes / 1756 triangles for a parent input and 144 / 254 for modelType=1. The input2 case produced 4252 / 8298; small numerical differences in polygon serialization can change the triangulation. Model and serial SMB generation succeeded for all three cases.
