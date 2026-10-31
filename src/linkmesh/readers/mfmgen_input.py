"""Translate M3D-C1 m3dc1_mfmgen input files to LinkMesh region generation."""
from __future__ import annotations

import math
from pathlib import Path
import shlex
import sys
from tempfile import TemporaryDirectory

import numpy as np

from linkmesh.config import ConfigError


def _read_points(path: Path) -> np.ndarray:
    rows = []
    for number, raw in enumerate(path.read_text().splitlines(), 1):
        line = raw.split('!', 1)[0].split('#', 1)[0].strip()
        if not line:
            continue
        fields = line.split()
        if len(fields) == 1 and not rows:
            try:
                rows.append((int(fields[0]),))
            except ValueError:
                raise ConfigError(f'{path}:{number}: expected an R Z pair or point-count header')
            continue
        if len(fields) != 2:
            raise ConfigError(f'{path}:{number}: expected an R Z coordinate pair')
        try:
            rows.append((float(fields[0]), float(fields[1])))
        except ValueError as exc:
            raise ConfigError(f'{path}:{number}: invalid R Z coordinate') from exc
    if rows and len(rows[0]) == 1:
        try:
            declared = int(rows[0][0])
        except (TypeError, ValueError):
            declared = -1
        if declared >= 3 and declared == len(rows) - 1:
            rows = rows[1:]
        else:
            raise ConfigError(f'{path}: one-column first line must declare the following point count')
    points = np.asarray(rows, dtype=np.float64)
    if points.ndim != 2 or points.shape[0] < 3 or not np.isfinite(points).all():
        raise ConfigError(f'{path}: boundary requires at least 3 finite R Z points')
    if np.array_equal(points[0], points[-1]):
        points = points[:-1]
    if len(points) < 3:
        raise ConfigError(f'{path}: boundary has fewer than 3 distinct points')
    return points


def _densify(points: np.ndarray, max_spacing: float) -> np.ndarray:
    edges = np.roll(points, -1, axis=0) - points
    lengths = np.linalg.norm(edges, axis=1)
    pieces = np.maximum(1, np.ceil(lengths / max_spacing).astype(int))
    return np.concatenate([
        points[i] + (np.arange(n, dtype=np.float64) / n)[:, None] * edges[i]
        for i, n in enumerate(pieces)
    ], axis=0)


def _offset(points: np.ndarray, distance: float) -> np.ndarray:
    """Miter offset of a simple polygon; positive distance expands its interior."""
    if not math.isfinite(distance) or distance <= 0:
        raise ConfigError('thickWall thickness must be finite and positive')
    following = np.roll(points, -1, axis=0)
    edges = following - points
    lengths = np.linalg.norm(edges, axis=1)
    if np.any(lengths == 0):
        raise ConfigError('thickWall source boundary has a zero-length edge')
    area2 = float(np.sum(points[:, 0]*following[:, 1] - following[:, 0]*points[:, 1]))
    if abs(area2) < 1e-12:
        raise ConfigError('thickWall source boundary has near-zero area')
    sign = 1.0 if area2 > 0 else -1.0
    # Outward unit normal for each directed edge.
    normals = sign * np.column_stack((edges[:, 1], -edges[:, 0])) / lengths[:, None]
    previous = np.roll(normals, 1, axis=0)
    summed = previous + normals
    denom = np.sum(summed * normals, axis=1)
    if np.any(np.abs(denom) < 1e-8):
        raise ConfigError('thickWall offset is undefined at a near-180-degree corner')
    result = points + distance * summed / denom[:, None]
    if not np.isfinite(result).all():
        raise ConfigError('thickWall offset produced invalid coordinates')
    return result


def parse_mfmgen_input(filename: str | Path) -> dict:
    path = Path(filename).expanduser().resolve()
    entries: dict[str, list[list[str]]] = {}
    for line_no, raw in enumerate(path.read_text().splitlines(), 1):
        lexer = shlex.shlex(raw, posix=True)
        lexer.whitespace_split = True
        lexer.commenters = '!#'
        try:
            fields = list(lexer)
        except ValueError as exc:
            raise ConfigError(f'{path}:{line_no}: {exc}') from exc
        if fields:
            entries.setdefault(fields[0], []).append(fields[1:])
    known = {'numBdry','bdryFile','thickWall','useVacuum','vacuumParams','numVacuumPts','numFace','faceBdry','outFile','meshGradationRate','layeredMesh'}
    unknown = set(entries)-known
    if unknown:
        raise ConfigError(f'{path}: unsupported mfmgen keyword(s): {sorted(unknown)}')
    def one(key, default=None):
        values=entries.get(key, [])
        if len(values)>1: raise ConfigError(f'{path}: {key} may appear only once')
        if not values: return default
        return values[0]
    def count(key, default):
        value=one(key, [str(default)])[0]
        try: result=int(value)
        except ValueError as exc: raise ConfigError(f'{path}: {key} must be integer') from exc
        if result<0: raise ConfigError(f'{path}: {key} cannot be negative')
        return result
    nbdry=count('numBdry',0)
    specs=entries.get('bdryFile',[])
    if len(specs)!=nbdry: raise ConfigError(f'{path}: numBdry={nbdry} but found {len(specs)} bdryFile entries')
    boundaries={}; sizes={}
    for fields in specs:
        if len(fields)!=3: raise ConfigError(f'{path}: bdryFile syntax is FILE LOOP_ID MESH_SIZE')
        name, loop, dx=fields
        try: loop=int(loop); dx=float(dx)
        except ValueError as exc: raise ConfigError(f'{path}: invalid bdryFile loop ID or mesh size') from exc
        if loop<=0 or loop in boundaries or not math.isfinite(dx) or dx<=0: raise ConfigError(f'{path}: invalid or duplicate bdryFile loop ID/size')
        src = Path(name).expanduser()
        if not src.is_absolute():
            src = (Path.cwd() / src).resolve()
        boundaries[loop] = _read_points(src)
        sizes[loop] = dx
    vacuum=one('useVacuum',['0','0','0.1'])
    if len(vacuum)!=3: raise ConfigError(f'{path}: useVacuum syntax is MODE LOOP_ID MESH_SIZE')
    try: mode, vid, vdx=int(vacuum[0]),int(vacuum[1]),float(vacuum[2])
    except ValueError as exc: raise ConfigError(f'{path}: invalid useVacuum values') from exc
    if mode not in (0,2): raise ConfigError('Only useVacuum mode 0 or parameterized mode 2 is supported')
    if mode==2:
        params=one('vacuumParams')
        if params is None or len(params)!=5: raise ConfigError('useVacuum=2 requires five vacuumParams values')
        try: r0,a,delta,z0,kappa=map(float,params)
        except ValueError as exc: raise ConfigError('vacuumParams must be numeric') from exc
        if not np.isfinite([r0,a,delta,z0,kappa]).all() or a<=0 or kappa<=0 or vid<=0 or vdx<=0: raise ConfigError('vacuumParams and useVacuum loop/size values must be finite and positive where required')
        npts=count('numVacuumPts',20)
        if npts<16: raise ConfigError('numVacuumPts must be at least 16')
        theta=np.linspace(0,2*np.pi,npts,endpoint=False)
        circumference_estimate=2*np.pi*a*np.sqrt((1+kappa*kappa)/2)
        npts=max(npts,int(math.ceil(circumference_estimate/vdx)))
        theta=np.linspace(0,2*np.pi,npts,endpoint=False)
        boundaries[vid]=np.column_stack((r0+a*np.cos(theta+delta*np.sin(theta)),z0+kappa*a*np.sin(theta)))
        sizes[vid]=vdx
    elif vid in boundaries:
        raise ConfigError('useVacuum mode 0 loop ID must not collide with a boundary loop')
    wall=one('thickWall',['0','0','0','0'])
    if len(wall)!=4: raise ConfigError(f'{path}: thickWall syntax is ENABLE SOURCE_LOOP_ID NEW_LOOP_ID THICKNESS')
    try: active, source, new, width=int(wall[0]),int(wall[1]),int(wall[2]),float(wall[3])
    except ValueError as exc: raise ConfigError(f'{path}: invalid thickWall values') from exc
    if active not in (0,1): raise ConfigError('thickWall enable must be 0 or 1')
    if active:
        if source not in boundaries or new<=0 or new in boundaries: raise ConfigError('thickWall references a missing or conflicting loop ID')
        boundaries[new]=_offset(boundaries[source],width)
        sizes[new]=sizes[source]
    boundaries={loop:_densify(points,sizes[loop]) for loop,points in boundaries.items()}
    faces=count('numFace',1)
    face_specs=entries.get('faceBdry',[])
    if len(face_specs)!=faces or faces<1: raise ConfigError(f'{path}: numFace={faces} but found {len(face_specs)} faceBdry entries')
    face_list=[]
    for fields in face_specs:
        if len(fields)<3: raise ConfigError(f'{path}: faceBdry syntax is NLOOPS LOOP_ID... MESH_SIZE')
        try:
            n=int(fields[0]); loops=list(map(int,fields[1:-1])); dx=float(fields[-1])
        except ValueError as exc: raise ConfigError(f'{path}: invalid faceBdry values') from exc
        if n<1 or len(loops)!=n or len(set(loops))!=n or any(i not in boundaries for i in loops) or not math.isfinite(dx) or dx<=0: raise ConfigError(f'{path}: malformed faceBdry loop IDs/count/size')
        face_list.append((set(loops),dx))
    for key in ('layeredMesh','meshGradationRate'):
        value=one(key)
        if value is not None:
            raise ConfigError(f'{key} is not yet translated by LinkMesh; remove it or use another mesh workflow')
    out=one('outFile',['mfmgen_mesh'])
    if len(out)!=1: raise ConfigError(f'{path}: outFile expects one path prefix')
    return {'path':path,'boundaries':boundaries,'sizes':sizes,'faces':face_list,'out':out[0]}


def run_mfmgen_input(filename: str | Path, verbose: bool = False) -> None:
    parsed = parse_mfmgen_input(filename)
    # mfmgen loops need a single containment tree. The tutorial form uses an
    # original wall loop, its outward offset, and an enclosing analytic vacuum.
    from linkmesh.regions import read_regions_map
    from linkmesh import cli
    old_argv=sys.argv[:]
    try:
      with TemporaryDirectory(prefix='linkmesh-mfmgen-') as tmp:
        work=Path(tmp); ordered=sorted(parsed['boundaries'])
        multi=work/'boundaries.txt'
        with multi.open('w') as stream:
          for loop in ordered:
            pts=parsed['boundaries'][loop]
            stream.write(f'{len(pts)}\n')
            np.savetxt(stream,pts,fmt='%.16e')
        mapfile=work/'regions.toml'
        inspect=['linkmesh','regions','inspect','--multi-array-file',str(multi),'--regions-map',str(mapfile),'--plot',str(work/'inspection.png')]
        if verbose: inspect += ['--verbose']
        sys.argv=inspect; cli.main()
        records=read_regions_map(mapfile)
        loop_by_curve={i+1:loop for i,loop in enumerate(ordered)}
        loop_by_region={rec.region_id:loop_by_curve[rec.curve_id] for rec in records}
        assigned={}
        face_region_order=[]
        for loops,dx in parsed['faces']:
          matches=[]
          for rec in records:
            represented={loop_by_region[rec.region_id]} | {loop_by_region[x] for x in rec.child_region_ids}
            if loops == represented:
              matches.append(rec.region_id)
          if len(matches)!=1: raise ConfigError(f'faceBdry does not match one nested boundary set: {sorted(loops)}')
          face_region_order.append(matches[0])
          assigned[matches[0]]=dx
        if len(assigned)!=len(records): raise ConfigError('faceBdry definitions do not cover each generated nested region exactly once')
        generate=['linkmesh','regions','generate','--multi-array-file',str(multi),'--regions-map',str(mapfile),'--target','m3dc1','--model',parsed['out']+'.txt','--smb',parsed['out']+'.smb','--plot',parsed['out']+'_mesh.png','--quality-plot',parsed['out']+'_quality.png','--verbose']
        for rid,dx in sorted(assigned.items()): generate += ['--region-dx',f'{rid}={dx}']
        generate += ['--region-order', *map(str, face_region_order)]
        sys.argv=generate; cli.main()
    finally:
      sys.argv=old_argv
