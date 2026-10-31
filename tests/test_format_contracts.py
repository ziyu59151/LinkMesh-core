from __future__ import annotations

import os
import stat
import struct
import subprocess
from pathlib import Path
from types import SimpleNamespace
from urllib.parse import quote

import numpy as np
import pytest

from linkmesh.m3dc1_model import (
    GeometricEdge,
    M3DC1ModelSpec,
    build_multi_face_model,
    build_single_face_model_spec,
)
from linkmesh.mesh import Mesh2D
from linkmesh.writers import smb_writer
from linkmesh.writers.hdf5_writer import write_hdf5
from linkmesh.writers.m3dc1_model_writer import (
    serialize_m3dc1_model,
    write_m3dc1_model,
)
from linkmesh.writers.smb_writer import (
    SMBWriterError,
    _MAGIC,
    _prepare_exchange,
    _rank0_path,
    _write_exchange,
    write_smb,
)


def _boundary() -> np.ndarray:
    return np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
        dtype=np.float64,
    )


def _mesh() -> Mesh2D:
    nodes = np.array(
        [
            [0.0, 0.0],
            [0.0, 0.5],
            [0.0, 1.0],
            [0.5, 1.0],
            [1.0, 1.0],
            [1.0, 0.5],
            [1.0, 0.0],
            [0.5, 0.0],
            [0.5, 0.5],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 8],
            [1, 2, 8],
            [2, 3, 8],
            [3, 4, 8],
            [4, 5, 8],
            [5, 6, 8],
            [6, 7, 8],
            [7, 0, 8],
        ],
        dtype=np.int32,
    )
    return Mesh2D(
        nodes=nodes,
        triangles=triangles,
        region_ids=np.ones(8, dtype=np.int32),
        outer_boundary=_boundary(),
    )


def test_trimeg_hdf5_exact_eleven_dataset_contract(tmp_path):
    h5py = pytest.importorskip("h5py")
    mesh = _mesh()
    path = tmp_path / "mesh.h5"

    write_hdf5(mesh, str(path), verbose=False)

    expected_names = {
        "nvert",
        "ntria",
        "nvert_in",
        "nvert_ex",
        "vert",
        "idxivert",
        "idxevert",
        "tria",
        "nbcRZ",
        "Rbc",
        "Zbc",
    }
    with h5py.File(path, "r") as file:
        group = file["femmesh"]
        assert set(group.keys()) == expected_names
        assert all(group[name].dtype == np.float64 for name in expected_names)
        assert group["vert"].shape == (9, 2)
        assert group["tria"].shape == (8, 3)
        assert group["idxivert"].shape == (1, 1)
        assert group["idxevert"].shape == (1, 8)
        assert group["nbcRZ"].shape == (1, 1)
        assert group["Rbc"].shape == (4, 1)
        assert group["Zbc"].shape == (4, 1)
        np.testing.assert_array_equal(group["vert"][:], mesh.nodes)
        np.testing.assert_array_equal(
            group["tria"][:],
            mesh.triangles.astype(np.float64) + 1.0,
        )
        np.testing.assert_array_equal(
            group["idxivert"][:],
            np.array([[9.0]]),
        )
        np.testing.assert_array_equal(
            group["idxevert"][:],
            np.arange(1.0, 9.0).reshape(1, 8),
        )
        np.testing.assert_array_equal(
            group["Rbc"][:].reshape(-1),
            mesh.outer_boundary[:, 0],
        )
        np.testing.assert_array_equal(
            group["Zbc"][:].reshape(-1),
            mesh.outer_boundary[:, 1],
        )


def test_hdf5_validation_precedes_destination_truncation(tmp_path):
    path = tmp_path / "mesh.h5"
    path.write_bytes(b"sentinel")
    mesh = _mesh()
    mesh.triangles = np.array([[0, 1, 99]], dtype=np.int32)

    with pytest.raises(ValueError, match="越界"):
        write_hdf5(mesh, str(path), verbose=False)

    assert path.read_bytes() == b"sentinel"


def test_m3dc1_model_and_txt_preserve_topology_and_layout(tmp_path):
    mesh = _mesh()
    spec = build_single_face_model_spec(mesh)
    text = serialize_m3dc1_model(spec)
    lines = text.splitlines()

    assert len(spec.vertices) == 4
    assert len(spec.edges) == 4
    assert spec.loops[0].edge_tags == (1, 2, 3, 4)
    assert spec.faces[0].loop_tags == (1,)
    assert lines[0] == "1 -1 1 -1 -1"
    assert lines[1] == "1 4"
    assert lines[6:10] == [
        "1 1 2 1",
        "2 2 3 1",
        "3 3 4 1",
        "4 4 1 1",
    ]
    assert lines[-2:] == ["1", "1 1 1"]

    destination = tmp_path / "model.txt"
    write_m3dc1_model(spec, destination)
    assert destination.read_text(encoding="utf-8") == text


def test_m3dc1_model_rejects_invalid_region_and_topology():
    mesh = _mesh()
    mesh.region_ids[0] = 2
    with pytest.raises(ValueError, match="single-region"):
        build_single_face_model_spec(mesh)

    with pytest.raises(ValueError, match="不连续"):
        M3DC1ModelSpec(
            vertices=build_single_face_model_spec(_mesh()).vertices,
            edges=(
                GeometricEdge(1, 1, 2),
                GeometricEdge(2, 3, 4),
                GeometricEdge(3, 4, 3),
                GeometricEdge(4, 2, 1),
            ),
            loops=build_single_face_model_spec(_mesh()).loops,
            faces=build_single_face_model_spec(_mesh()).faces,
        )


def test_exchange_schema_layout_and_classification(tmp_path):
    mesh = _mesh()
    spec = build_single_face_model_spec(mesh)
    base = tmp_path / "mesh.smb"
    prepared = _prepare_exchange(mesh, spec, base)
    path = tmp_path / "exchange.bin"

    _write_exchange(prepared, path)
    data = path.read_bytes()
    header = struct.Struct("<8sII7Q4i")
    unpacked = header.unpack_from(data)

    assert unpacked[0] == _MAGIC
    assert unpacked[1] == 1
    assert unpacked[2] == 0x01020304
    assert unpacked[3:10] == (9, 8, 16, 4, 4, 1, 1)
    assert unpacked[10:14] == (-1, 1, -1, -1)
    assert prepared.vertex_classes == (
        (0, 1),
        (1, 1),
        (0, 2),
        (1, 2),
        (0, 3),
        (1, 3),
        (0, 4),
        (1, 4),
        (2, 1),
    )
    assert prepared.triangle_classes == ((2, 1),) * 8
    assert [
        (edge[0], edge[1]) for edge in prepared.mesh_edges
    ] == sorted((edge[0], edge[1]) for edge in prepared.mesh_edges)


def _two_region_mesh() -> Mesh2D:
    """Return one outer Region enclosing one conforming inner Region."""
    nodes = np.array(
        [
            [0.0, 0.0],
            [0.0, 4.0],
            [4.0, 4.0],
            [4.0, 0.0],
            [1.0, 1.0],
            [1.0, 3.0],
            [3.0, 3.0],
            [3.0, 1.0],
            [2.0, 2.0],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 1, 5],
            [0, 5, 4],
            [1, 2, 6],
            [1, 6, 5],
            [2, 3, 7],
            [2, 7, 6],
            [3, 0, 4],
            [3, 4, 7],
            [4, 5, 8],
            [5, 6, 8],
            [6, 7, 8],
            [7, 4, 8],
        ],
        dtype=np.int32,
    )
    return Mesh2D(
        nodes=nodes,
        triangles=triangles,
        region_ids=np.array(
            [20, 20, 20, 20, 20, 20, 20, 20, 10, 10, 10, 10],
            dtype=np.int32,
        ),
        outer_boundary=np.array(
            [[0.0, 0.0], [0.0, 4.0], [4.0, 4.0], [4.0, 0.0]],
            dtype=np.float64,
        ),
    )


def _shared_outer_boundary_mesh() -> Mesh2D:
    """Return the rejected topology where two Regions touch one outer loop."""
    nodes = np.array(
        [
            [0.0, 0.0],
            [0.0, 2.0],
            [2.0, 2.0],
            [2.0, 0.0],
            [1.0, 0.5],
            [1.0, 1.5],
        ],
        dtype=np.float64,
    )
    triangles = np.array(
        [
            [0, 4, 3],
            [3, 4, 2],
            [2, 4, 5],
            [2, 5, 1],
            [1, 5, 0],
            [0, 5, 4],
        ],
        dtype=np.int32,
    )
    return Mesh2D(
        nodes=nodes,
        triangles=triangles,
        region_ids=np.array([20, 20, 10, 10, 10, 10], dtype=np.int32),
        outer_boundary=np.array(
            [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]],
            dtype=np.float64,
        ),
    )


def test_multi_face_build_result_is_topology_driven_and_schema_v2(tmp_path):
    mesh = _two_region_mesh()
    nodes_before = mesh.nodes.copy()
    triangles_before = mesh.triangles.copy()
    regions_before = mesh.region_ids.copy()

    result = build_multi_face_model(mesh)

    assert dict(result.region_id_to_face_tag) == {10: 1, 20: 2}
    assert len(result.spec.faces) == 2
    assert len(result.spec.loops) == 2
    assert len(result.spec.vertices) == 8
    assert len(result.spec.edges) == 8
    assert result.spec.separatrix_loop == -1
    assert result.spec.inner_wall_loop == -1
    assert result.spec.outer_wall_loop == -1
    assert result.spec.vacuum_loop == -1

    outer_cycle, interface_cycle = result.cycles
    assert outer_cycle.loop_tag == 1
    assert outer_cycle.node_indices == (0, 1, 2, 3)
    assert outer_cycle.adjacent_region_ids == (20,)
    assert interface_cycle.loop_tag == 2
    assert interface_cycle.node_indices == (4, 5, 6, 7)
    assert interface_cycle.adjacent_region_ids == (10, 20)

    face_by_tag = {face.tag: face for face in result.spec.faces}
    assert face_by_tag[1].loop_tags == (2,)
    assert face_by_tag[2].loop_tags == (1, 2)
    assert sum(
        interface_cycle.loop_tag in face.loop_tags
        for face in result.spec.faces
    ) == 2
    assert len(set(interface_cycle.geometric_edge_tags)) == 4

    assert [
        classification.tag
        for classification in result.triangle_classifications
    ] == [2] * 8 + [1] * 4
    assert {
        classification.dimension
        for classification in result.triangle_classifications
    } == {2}
    assert len(result.vertex_classifications) == mesh.n_nodes
    assert result.vertex_classifications[8].dimension == 2
    assert result.vertex_classifications[8].tag == 1

    assert len(result.mesh_edges) == 20
    assert len(result.mesh_edges) == len(set(result.mesh_edges))
    outer_edges = {(0, 1), (1, 2), (2, 3), (0, 3)}
    interface_edges = {(4, 5), (5, 6), (6, 7), (4, 7)}
    cycle_edges = outer_edges | interface_edges
    edge_class_by_nodes = dict(
        zip(
            result.mesh_edges,
            result.edge_classifications,
            strict=True,
        )
    )
    assert all(
        edge_class_by_nodes[edge].dimension == 1
        for edge in cycle_edges
    )
    assert all(
        classification.dimension == 2
        for edge, classification in edge_class_by_nodes.items()
        if edge not in cycle_edges
    )
    assert {
        edge_class_by_nodes[edge].tag for edge in outer_edges
    } == set(outer_cycle.geometric_edge_tags)
    assert {
        edge_class_by_nodes[edge].tag for edge in interface_edges
    } == set(interface_cycle.geometric_edge_tags)

    prepared = smb_writer._prepare_build_result_exchange(
        mesh,
        result,
        tmp_path / "multi.smb",
    )
    assert prepared.spec is result.spec
    assert len(prepared.nodes) == mesh.n_nodes
    assert len(prepared.triangles) == mesh.n_triangles
    assert len(prepared.mesh_edges) == len(result.mesh_edges)

    exchange = tmp_path / "multi.bin"
    _write_exchange(prepared, exchange)
    header = struct.Struct("<8sII7Q4i").unpack_from(exchange.read_bytes())
    assert header[1] == 2
    assert header[3:10] == (9, 12, 20, 8, 8, 2, 2)
    assert header[10:14] == (-1, -1, -1, -1)

    text = serialize_m3dc1_model(result)
    assert text.splitlines()[0] == "2 -1 -1 -1 -1"
    assert text.splitlines()[-3] == "2"
    assert text.splitlines()[-2:] == ["1 1 2", "2 2 1 2"]

    np.testing.assert_array_equal(mesh.nodes, nodes_before)
    np.testing.assert_array_equal(mesh.triangles, triangles_before)
    np.testing.assert_array_equal(mesh.region_ids, regions_before)


def _disconnected_region_mesh() -> Mesh2D:
    """Return two disjoint interior triangle components with Region ID 10."""
    columns = 6
    rows = 4
    nodes = np.array(
        [
            [float(column), float(row)]
            for row in range(rows)
            for column in range(columns)
        ],
        dtype=np.float64,
    )
    triangles: list[list[int]] = []
    for row in range(rows - 1):
        for column in range(columns - 1):
            lower_left = row * columns + column
            lower_right = lower_left + 1
            upper_left = (row + 1) * columns + column
            upper_right = upper_left + 1
            triangles.extend(
                [
                    [lower_left, upper_left, upper_right],
                    [lower_left, upper_right, lower_right],
                ]
            )

    region_ids = np.full(len(triangles), 20, dtype=np.int32)
    region_ids[12] = 10
    region_ids[16] = 10
    return Mesh2D(
        nodes=nodes,
        triangles=np.asarray(triangles, dtype=np.int32),
        region_ids=region_ids,
        outer_boundary=np.array(
            [[0.0, 0.0], [0.0, 3.0], [5.0, 3.0], [5.0, 0.0]],
            dtype=np.float64,
        ),
    )


def test_multi_face_rejects_disconnected_region():
    mesh = _disconnected_region_mesh()

    with pytest.raises(ValueError, match="不连通"):
        build_multi_face_model(mesh)


def test_multi_face_rejects_multiple_regions_on_one_outer_boundary_cycle():
    mesh = _shared_outer_boundary_mesh()

    with pytest.raises(
        ValueError,
        match="outer-boundary cycle 邻接多个 Regions",
    ):
        build_multi_face_model(mesh)


def _executable(path: Path) -> Path:
    path.write_text("helper", encoding="utf-8")
    path.chmod(path.stat().st_mode | stat.S_IXUSR)
    return path


def test_helper_success_confirms_rank0_and_removes_exchange(
    tmp_path,
    monkeypatch,
):
    mesh = _mesh()
    spec = build_single_face_model_spec(mesh)
    base = tmp_path / "mesh.smb"
    helper = _executable(tmp_path / "helper")
    monkeypatch.setattr(smb_writer.os, "access", lambda *_args: True)

    def successful_run(*_args, **_kwargs):
        rank0 = _rank0_path(base.absolute())
        rank0.write_bytes(b"native-smb")
        return subprocess.CompletedProcess(
            args=[],
            returncode=0,
            stdout=(
                "LINKMESH_M4_OK "
                f"base={quote(str(base.absolute()), safe='/:\\\\')} "
                f"rank0={quote(str(rank0), safe='/:\\\\')}\n"
            ),
            stderr="",
        )

    monkeypatch.setattr(subprocess, "run", successful_run)

    write_smb(mesh, spec, base, helper_path=helper)

    assert _rank0_path(base.absolute()).read_bytes() == b"native-smb"
    assert list(tmp_path.glob(".*.lm4xchg")) == []


def test_helper_failure_reports_native_stage_and_retains_exchange(
    tmp_path,
    monkeypatch,
):
    mesh = _mesh()
    spec = build_single_face_model_spec(mesh)
    base = tmp_path / "mesh.smb"
    helper = _executable(tmp_path / "helper")
    monkeypatch.setattr(smb_writer.os, "access", lambda *_args: True)
    monkeypatch.setattr(
        subprocess,
        "run",
        lambda *_args, **_kwargs: SimpleNamespace(
            returncode=17,
            stdout="",
            stderr=(
                "LINKMESH_M4_ERROR "
                "stage=fake_native message=failure\n"
            ),
        ),
    )

    with pytest.raises(SMBWriterError, match="stage=fake_native"):
        write_smb(mesh, spec, base, helper_path=helper)

    assert len(list(tmp_path.glob(".*.lm4xchg"))) == 1
    assert not _rank0_path(base.absolute()).exists()


def test_smb_input_validation_stops_before_helper(tmp_path):
    mesh = _mesh()
    spec = build_single_face_model_spec(mesh)
    mesh.region_ids[0] = 2

    with pytest.raises(ValueError, match="single-region"):
        write_smb(
            mesh,
            spec,
            tmp_path / "mesh.smb",
            helper_path=tmp_path / "unused",
        )
