from __future__ import annotations

import builtins
import sys
from pathlib import Path
from types import ModuleType, SimpleNamespace

import numpy as np
import pytest

from linkmesh.mesh import Mesh2D
from linkmesh.readers.boundary_list_reader import (
    BoundaryListError,
    read_boundary_list,
)
from linkmesh.readers.boundary_reader import (
    BoundaryFormatError,
    read_boundary_file,
    read_multi_boundary_file,
)
from linkmesh.regions import (
    RegionMapEntry,
    RegionsError,
    build_region_plan,
    resolve_region_dx,
    validate_plan_against_map,
    validate_region_map,
)
from linkmesh.readers.gfile_reader import read_gfile


def _clockwise_square() -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        dtype=np.float64,
    )


def _mesh(boundary=None) -> Mesh2D:
    return Mesh2D(
        nodes=_clockwise_square(),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        region_ids=np.array([1, 1], dtype=np.int32),
        outer_boundary=_clockwise_square() if boundary is None else boundary,
    )


def _signed_area(points: np.ndarray) -> float:
    following = np.roll(points, -1, axis=0)
    return 0.5 * float(
        np.sum(
            points[:, 0] * following[:, 1]
            - following[:, 0] * points[:, 1]
        )
    )


def test_mesh2d_preserves_core_arrays_and_owns_outer_boundary():
    nodes = _clockwise_square()
    triangles = np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32)
    regions = np.array([7, 7], dtype=np.int32)
    boundary = _clockwise_square()

    mesh = Mesh2D(nodes, triangles, regions, outer_boundary=boundary)
    snapshot = mesh.outer_boundary.copy()
    boundary[0, 0] = 99.0

    assert mesh.nodes is nodes
    assert mesh.triangles is triangles
    assert mesh.region_ids is regions
    assert (mesh.n_nodes, mesh.n_triangles, mesh.n_boundary_segments) == (4, 2, 0)
    assert mesh.outer_boundary.dtype == np.float64
    assert not np.shares_memory(mesh.outer_boundary, boundary)
    np.testing.assert_array_equal(mesh.outer_boundary, snapshot)


@pytest.mark.parametrize(
    "boundary, message",
    [
        (
            np.array([0.0, 1.0, 2.0]),
            "二维",
        ),
        (
            np.array([[0.0, 0.0], [1.0, 0.0]]),
            "3",
        ),
        (
            np.array(
                [[0.0, 0.0], [0.0, np.nan], [1.0, 1.0], [1.0, 0.0]]
            ),
            "有限",
        ),
        (
            np.array(
                [[0.0, 0.0], [0.0, 1.0], [0.0, 1.0], [1.0, 0.0]]
            ),
            "相邻重复",
        ),
        (
            np.array(
                [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [0.0, 0.0]]
            ),
            "首点与末点",
        ),
        (
            np.array(
                [[0.0, 0.0], [1.0, 0.0], [1.0, 1.0], [0.0, 1.0]]
            ),
            "顺时针",
        ),
        (
            np.array(
                [[0.0, 0.0], [2.0, 3.0], [0.0, 2.0], [3.0, 0.0]]
            ),
            "自交",
        ),
    ],
)
def test_outer_boundary_independent_failure_modes(boundary, message):
    with pytest.raises(ValueError, match=message):
        _mesh(boundary)


def test_boundary_reader_reads_real_text_and_normalizes_direction(tmp_path):
    path = tmp_path / "EAST"
    path.write_text(
        "4\n0 0\n1 0\n1 1\n0 1\n",
        encoding="utf-8",
    )

    boundary = read_boundary_file(path)

    assert boundary.shape == (4, 2)
    assert boundary.dtype == np.float64
    np.testing.assert_array_equal(boundary[0], [0.0, 0.0])
    assert _signed_area(boundary) < 0.0
    assert not np.array_equal(boundary[0], boundary[-1])


@pytest.mark.parametrize(
    "content, exception",
    [
        ("header\n0 0\n0 1\n1 0\n", BoundaryFormatError),
        ("4\n0 0\n0 1\n1 1\n", BoundaryFormatError),
        ("3\n0 0\nnan 1\n1 0\n", BoundaryFormatError),
        ("3\n0 0\n0 1\n1 0 2\n", BoundaryFormatError),
        ("4\n0 0\n2 3\n0 2\n3 0\n", ValueError),
    ],
)
def test_boundary_reader_rejects_distinct_format_and_geometry_errors(
    tmp_path,
    content,
    exception,
):
    path = tmp_path / "boundary"
    path.write_text(content, encoding="utf-8")

    with pytest.raises(exception):
        read_boundary_file(path)


def test_multi_boundary_reader_ignores_empty_lines_and_preserves_blocks(
    tmp_path,
):
    path = tmp_path / "multi"
    path.write_text(
        "\n4\n0 0\n\n2 0\n2 2\n0 2\n\n"
        "4\n0.5 0.5\n1.0 0.5\n\n1.0 1.0\n0.5 1.0\n\n",
        encoding="utf-8",
    )

    blocks = read_multi_boundary_file(path)

    assert len(blocks) == 2
    assert all(block.dtype == np.float64 for block in blocks)
    np.testing.assert_array_equal(blocks[0][0], [0.0, 0.0])
    np.testing.assert_array_equal(blocks[1][0], [0.5, 0.5])


def test_boundary_list_reader_reads_comments_paths_and_curve_ids(tmp_path):
    boundary_dir = tmp_path / "boundaries"
    boundary_dir.mkdir()
    (boundary_dir / "outer.pts").write_text(
        "# outer boundary\n"
        "4  # point count\n"
        "0 0  # R Z\n"
        "0 3\n"
        "3 3\n"
        "3 0\n",
        encoding="utf-8",
    )
    (boundary_dir / "inner.pts").write_text(
        "4\n"
        "1 1\n"
        "1 2\n"
        "2 2\n"
        "2 1\n",
        encoding="utf-8",
    )
    list_path = boundary_dir / "boundaries.toml"
    list_path.write_text(
        "schema_version = 1\n\n"
        "[[boundaries]]\n"
        'file = "outer.pts"\n'
        "curve_id = 10\n\n"
        "[[boundaries]]\n"
        'file = "inner.pts"\n',
        encoding="utf-8",
    )

    records = read_boundary_list(list_path)

    assert [record.curve_id for record in records] == [10, 1]
    assert [record.source_path for record in records] == [
        (boundary_dir / "outer.pts").resolve(),
        (boundary_dir / "inner.pts").resolve(),
    ]
    assert all(record.points.dtype == np.float64 for record in records)
    assert all(_signed_area(record.points) < 0.0 for record in records)


def test_boundary_list_reader_rejects_duplicate_references_and_bad_ids(
    tmp_path,
):
    point_file = tmp_path / "curve.pts"
    point_file.write_text(
        "4\n0 0\n0 1\n1 1\n1 0\n",
        encoding="utf-8",
    )
    duplicate_path = tmp_path / "duplicate.toml"
    duplicate_path.write_text(
        "schema_version = 1\n"
        "[[boundaries]]\nfile = 'curve.pts'\ncurve_id = 1\n"
        "[[boundaries]]\nfile = './curve.pts'\ncurve_id = 2\n",
        encoding="utf-8",
    )
    with pytest.raises(BoundaryListError, match="重复引用"):
        read_boundary_list(duplicate_path)

    bad_id_path = tmp_path / "bad-id.toml"
    bad_id_path.write_text(
        "schema_version = 1\n"
        "[[boundaries]]\nfile = 'curve.pts'\ncurve_id = true\n"
        "[[boundaries]]\nfile = 'curve.pts.2'\n",
        encoding="utf-8",
    )
    with pytest.raises(BoundaryListError, match="curve_id"):
        read_boundary_list(bad_id_path)


def test_multi_boundary_reader_rejects_single_and_truncated_blocks(tmp_path):
    single = tmp_path / "single"
    single.write_text("3\n0 0\n1 0\n0 1\n", encoding="utf-8")
    with pytest.raises(BoundaryFormatError, match="array_file"):
        read_multi_boundary_file(single)

    truncated = tmp_path / "truncated"
    truncated.write_text(
        "3\n0 0\n1 0\n0 1\n4\n0 0\n1 0\n",
        encoding="utf-8",
    )
    with pytest.raises(BoundaryFormatError, match="EOF"):
        read_multi_boundary_file(truncated)


def test_boundary_list_records_reuse_region_plan_and_map_semantics():
    from linkmesh.readers.boundary_list_reader import BoundaryRecord

    root = np.array([[-1, -1], [-1, 3], [5, 3], [5, -1]], dtype=float)
    child = np.array([[0, 0], [0, 2], [2, 2], [2, 0]], dtype=float)

    plan = build_region_plan(
        (
            BoundaryRecord(10, root, Path("outer.pts")),
            BoundaryRecord(20, child, Path("inner.pts")),
        )
    )

    assert plan.root_curve_id == 10
    assert plan.by_curve_id()[10].child_curve_ids == (20,)
    assert plan.by_curve_id()[20].parent_curve_id == 10

    from linkmesh.regions import assign_final_region_ids, region_map_entries

    numbered = assign_final_region_ids(plan, {10: 2, 20: 1})
    records = region_map_entries(numbered)
    assert {record.curve_id for record in records} == {10, 20}
    assert len(records) == 2


def test_region_plan_supports_nested_siblings_and_synthetic_tie_break():
    root = np.array([[-1, -1], [-1, 3], [5, 3], [5, -1]], dtype=float)
    left = np.array([[0, 0], [0, 2], [2, 2], [2, 0]], dtype=float)
    right = np.array([[3, 0], [3, 2], [4, 2], [4, 0]], dtype=float)
    core = np.array([[0.5, 0.5], [0.5, 1.5], [1.5, 1.5], [1.5, 0.5]])

    plan = build_region_plan((root, left, right, core))
    entries = plan.by_curve_id()

    assert plan.root_curve_id == 1
    assert entries[1].child_curve_ids == (2, 3)
    assert entries[2].child_curve_ids == (4,)
    assert entries[4].depth == 2
    assert entries[4].technical_role == "plasma"
    assert entries[3].technical_role == "vacuum"


def test_region_plan_rejects_multiple_roots():
    first = np.array([[0, 0], [0, 1], [1, 1], [1, 0]], dtype=float)
    second = np.array([[2, 0], [2, 1], [3, 1], [3, 0]], dtype=float)

    with pytest.raises(RegionsError, match="唯一 root"):
        build_region_plan((first, second))


def test_region_plan_keeps_boundary_ambiguity_for_larger_parent_candidate():
    root = np.array([[0, 0], [0, 4], [4, 4], [4, 0]], dtype=float)
    crossing = np.array([[3, -1], [3, 1], [5, 1], [5, -1]], dtype=float)

    with pytest.raises(RegionsError, match="包含关系有歧义"):
        build_region_plan((root, crossing))


def test_region_map_rejects_cycle_disconnected_from_unique_root():
    records = (
        RegionMapEntry(1, 1, 0, (), "#0072B2"),
        RegionMapEntry(2, 2, 3, (3,), "#D55E00"),
        RegionMapEntry(3, 3, 2, (2,), "#009E73"),
    )

    with pytest.raises(RegionsError, match="cycle"):
        validate_region_map(records)


def test_region_map_accepts_valid_nested_and_sibling_tree():
    records = (
        RegionMapEntry(1, 4, 2, (), "#0072B2"),
        RegionMapEntry(2, 2, 4, (1,), "#D55E00"),
        RegionMapEntry(3, 3, 4, (), "#009E73"),
        RegionMapEntry(4, 1, 0, (2, 3), "#CC79A7"),
    )

    validate_region_map(records)


def test_numbered_plan_matches_map_while_allowing_user_metadata_edits():
    root = np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=float)
    child = np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=float)
    from linkmesh.regions import assign_final_region_ids

    plan = assign_final_region_ids(
        build_region_plan((root, child)),
        {1: 2, 2: 1},
    )
    records = (
        RegionMapEntry(1, 2, 2, (), "#FFFFFF", "user label"),
        RegionMapEntry(2, 1, 0, (1,), "#000000", ""),
    )

    validate_plan_against_map(plan, records)


@pytest.mark.parametrize(
    "records, message",
    [
        (
            (
                RegionMapEntry(1, 2, 0, (), "#FFFFFF"),
                RegionMapEntry(2, 1, 1, (1,), "#000000"),
            ),
            "parent_region_id",
        ),
        (
            (
                RegionMapEntry(1, 2, 2, (), "#FFFFFF"),
                RegionMapEntry(2, 1, 0, (), "#000000"),
            ),
            "child_region_ids",
        ),
    ],
)
def test_numbered_plan_rejects_map_topology_conflicts(records, message):
    root = np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=float)
    child = np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=float)
    from linkmesh.regions import assign_final_region_ids

    plan = assign_final_region_ids(
        build_region_plan((root, child)),
        {1: 2, 2: 1},
    )

    with pytest.raises(RegionsError, match=message):
        validate_plan_against_map(plan, records)


def test_numbered_plan_rejects_map_identity_and_topology_conflicts():
    root = np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=float)
    child = np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=float)
    from linkmesh.regions import assign_final_region_ids

    plan = assign_final_region_ids(
        build_region_plan((root, child)),
        {1: 2, 2: 1},
    )
    conflicting = (
        RegionMapEntry(1, 1, 0, (2,), "#0072B2"),
        RegionMapEntry(2, 2, 1, (), "#D55E00"),
    )

    with pytest.raises(RegionsError, match="Region ID 已变化"):
        validate_plan_against_map(plan, conflicting)


def test_resolve_region_dx_supports_default_overrides_and_full_coverage():
    records = (
        RegionMapEntry(1, 2, 2, (), "#0072B2"),
        RegionMapEntry(2, 1, 0, (1,), "#D55E00"),
    )

    assert resolve_region_dx(records, 0.08, {1: 0.03}) == {
        1: 0.03,
        2: 0.08,
    }
    assert resolve_region_dx(records, None, {1: 0.03, 2: 0.05}) == {
        1: 0.03,
        2: 0.05,
    }

    with pytest.raises(RegionsError, match="完整覆盖"):
        resolve_region_dx(records, None, {1: 0.03})
    with pytest.raises(RegionsError, match="未知 Region ID 3"):
        resolve_region_dx(records, 0.08, {3: 0.03})


def _install_fake_oft(monkeypatch, equilibrium) -> None:
    root = ModuleType("OpenFUSIONToolkit")
    tokamaker = ModuleType("OpenFUSIONToolkit.TokaMaker")
    eqdsk = ModuleType("OpenFUSIONToolkit.TokaMaker.eqdsk")
    eqdsk.GEQDSKEquilibrium = lambda _path: equilibrium
    root.TokaMaker = tokamaker
    tokamaker.eqdsk = eqdsk

    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit", root)
    monkeypatch.setitem(sys.modules, "OpenFUSIONToolkit.TokaMaker", tokamaker)
    monkeypatch.setitem(
        sys.modules,
        "OpenFUSIONToolkit.TokaMaker.eqdsk",
        eqdsk,
    )


def test_gfile_reader_maps_fields_and_normalizes_lcfs(monkeypatch, tmp_path):
    equilibrium = SimpleNamespace(
        boundary_R=np.array([0.0, 0.0, 1.0, 1.0]),
        boundary_Z=np.array([0.0, 1.0, 1.0, 0.0]),
        R_mag=0.5,
        Z_mag=0.25,
        limiter_R=np.array([0.0, 2.0, 2.0]),
        limiter_Z=np.array([0.0, 0.0, 2.0]),
        R_center=1.7,
        B_center=2.1,
        Ip=3.2e6,
    )
    _install_fake_oft(monkeypatch, equilibrium)

    result = read_gfile(str(tmp_path / "eq.gfile"))
    lcfs = np.column_stack((result.lcfs_r, result.lcfs_z))

    assert result.rmaxis == 0.5
    assert result.zmaxis == 0.25
    assert result.rcentr == 1.7
    assert result.bcentr == 2.1
    assert result.current == 3.2e6
    assert result.wall_r is not None
    assert result.wall_z is not None
    np.testing.assert_array_equal(lcfs[0], lcfs[-1])
    assert _signed_area(lcfs[:-1]) > 0.0


def test_gfile_reader_represents_missing_limiter_as_none(monkeypatch):
    equilibrium = SimpleNamespace(
        boundary_R=np.array([0.0, 1.0, 0.0]),
        boundary_Z=np.array([0.0, 0.0, 1.0]),
        R_mag=0.5,
        Z_mag=0.25,
        limiter_R=np.array([]),
        limiter_Z=np.array([]),
        R_center=None,
        B_center=None,
        Ip=None,
    )
    _install_fake_oft(monkeypatch, equilibrium)

    result = read_gfile("eq.gfile")

    assert result.wall_r is None
    assert result.wall_z is None


def test_gfile_reader_reports_oft_import_failure(monkeypatch):
    for name in (
        "OpenFUSIONToolkit.TokaMaker.eqdsk",
        "OpenFUSIONToolkit.TokaMaker",
        "OpenFUSIONToolkit",
    ):
        monkeypatch.delitem(sys.modules, name, raising=False)

    real_import = builtins.__import__

    def failing_import(name, *args, **kwargs):
        if name.startswith("OpenFUSIONToolkit"):
            raise OSError("native OFT unavailable")
        return real_import(name, *args, **kwargs)

    monkeypatch.setattr(builtins, "__import__", failing_import)

    with pytest.raises(ImportError) as captured:
        read_gfile("eq.gfile")

    assert "native OFT unavailable" in str(captured.value)
    assert isinstance(captured.value.__cause__, OSError)
