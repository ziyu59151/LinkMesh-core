"""M3D-C1 serial native SMB writer.

The Python side validates Mesh2D and M3DC1ModelSpec, derives deterministic
geometric classification, writes the LinkMesh exchange schema, and invokes the
standalone SCOREC/PUMI helper.

This module never reads, writes, parses, or modifies SMB bytes. Native SMB
serialization is performed exclusively by SCOREC/PUMI writeNative().
"""

from __future__ import annotations

import os
import shutil
import struct
import subprocess
import tempfile
from dataclasses import dataclass
from pathlib import Path
from typing import TYPE_CHECKING
from urllib.parse import unquote

import numpy as np

from linkmesh.m3dc1_model import (
    CurveType,
    M3DC1BuildResult,
    M3DC1ModelSpec,
)
from linkmesh.mesh import Mesh2D

if TYPE_CHECKING:
    from collections.abc import Iterable


_MAGIC = b"LM4XCHG\x00"
_SCHEMA_VERSION = 1
_MULTI_FACE_SCHEMA_VERSION = 2
_ENDIAN_MARKER = 0x01020304
_HELPER_NAME = "linkmesh-scorec-helper"
_HELPER_ENV = "LINKMESH_SCOREC_HELPER"
_HELPER_TIMEOUT_SECONDS = 300

_U32 = struct.Struct("<I")
_I32 = struct.Struct("<i")
_U64 = struct.Struct("<Q")
_F64 = struct.Struct("<d")
_COUNTS = struct.Struct("<7Q")
_SPECIAL_LOOPS = struct.Struct("<4i")
_GEOMETRIC_VERTEX = struct.Struct("<i3d")
_GEOMETRIC_EDGE = struct.Struct("<4i")
_CLASSIFICATION = struct.Struct("<2i")
_MESH_EDGE = struct.Struct("<2Q2i")


class SMBWriterError(RuntimeError):
    """Raised when the Python/helper SMB production boundary fails."""


@dataclass(frozen=True)
class _PreparedExchange:
    schema_version: int
    nodes: np.ndarray
    triangles: np.ndarray
    region_ids: np.ndarray
    mesh_edges: tuple[tuple[int, int, int, int], ...]
    vertex_classes: tuple[tuple[int, int], ...]
    triangle_classes: tuple[tuple[int, int], ...]
    spec: M3DC1ModelSpec
    output_base: Path


def _as_nodes(nodes_in) -> np.ndarray:
    try:
        nodes = np.asarray(nodes_in, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Mesh2D.nodes 无法转换为 float64 数组") from exc

    if nodes.ndim != 2 or nodes.shape[1] != 2:
        raise ValueError(
            f"Mesh2D.nodes 必须为 N×2，实际 shape={nodes.shape}"
        )
    if nodes.shape[0] < 3:
        raise ValueError(
            f"Mesh2D.nodes 至少需要 3 个节点，实际 N={nodes.shape[0]}"
        )
    if not np.all(np.isfinite(nodes)):
        raise ValueError("Mesh2D.nodes 含非有限值（NaN/Inf）")
    return np.array(nodes, dtype=np.float64, copy=True, order="C")


def _as_triangles(triangles_in, n_nodes: int) -> np.ndarray:
    try:
        values = np.asarray(triangles_in, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Mesh2D.triangles 无法转换为数值数组") from exc

    if values.ndim != 2 or values.shape[1] != 3:
        raise ValueError(
            f"Mesh2D.triangles 必须为 T×3，实际 shape={values.shape}"
        )
    if values.shape[0] == 0:
        raise ValueError("Mesh2D.triangles 为空")
    if not np.all(np.isfinite(values)):
        raise ValueError("Mesh2D.triangles 含非有限值（NaN/Inf）")
    if not np.all(values == np.round(values)):
        raise ValueError("Mesh2D.triangles 必须具有整数语义")

    triangles = values.astype(np.uint64)
    if np.any(values < 0.0):
        raise ValueError("Mesh2D.triangles 必须使用非负 0-based 节点索引")
    if int(triangles.max()) >= n_nodes:
        raise ValueError(
            "Mesh2D.triangles 节点索引越界："
            f"最大索引={int(triangles.max())}，节点数={n_nodes}"
        )

    for index, triangle in enumerate(triangles):
        if len(set(int(value) for value in triangle)) != 3:
            raise ValueError(f"triangle {index} 含重复节点索引")

    return np.array(triangles, dtype=np.uint64, copy=True, order="C")


def _as_single_region_ids(region_ids_in, n_triangles: int) -> np.ndarray:
    try:
        values = np.asarray(region_ids_in, dtype=np.float64)
    except (TypeError, ValueError, OverflowError) as exc:
        raise ValueError("Mesh2D.region_ids 无法转换为数值数组") from exc

    if values.ndim != 1:
        raise ValueError(
            f"Mesh2D.region_ids 必须为一维数组，实际 ndim={values.ndim}"
        )
    if values.shape[0] != n_triangles:
        raise ValueError(
            "Mesh2D.region_ids 数量必须等于 triangle 数："
            f"region_ids={values.shape[0]}，triangles={n_triangles}"
        )
    if not np.all(np.isfinite(values)):
        raise ValueError("Mesh2D.region_ids 含非有限值（NaN/Inf）")
    if not np.all(values == np.round(values)):
        raise ValueError("Mesh2D.region_ids 必须具有整数语义")
    if np.any(values <= 0.0):
        raise ValueError("Mesh2D.region_ids 必须是正整数")

    region_ids = values.astype(np.int64)
    unique = np.unique(region_ids)
    if unique.size != 1:
        raise ValueError(
            "当前 writer 只支持 single-region Mesh2D，"
            f"实际 region IDs={unique.tolist()}"
        )
    return np.array(region_ids, dtype=np.int64, copy=True, order="C")


def _validate_triangle_geometry(
    nodes: np.ndarray,
    triangles: np.ndarray,
) -> None:
    for index, triangle in enumerate(triangles):
        p0 = nodes[int(triangle[0])]
        p1 = nodes[int(triangle[1])]
        p2 = nodes[int(triangle[2])]
        cross = float(
            (p1[0] - p0[0]) * (p2[1] - p0[1])
            - (p1[1] - p0[1]) * (p2[0] - p0[0])
        )
        if not np.isfinite(cross) or abs(cross) <= 2.0e-12:
            raise ValueError(
                f"triangle {index} 面积退化或不大于 1e-12："
                f"signed_double_area={cross}"
            )


def _build_edge_counts(
    triangles: np.ndarray,
) -> dict[tuple[int, int], int]:
    edge_counts: dict[tuple[int, int], int] = {}
    for triangle in triangles:
        v0, v1, v2 = (int(value) for value in triangle)
        for first, second in ((v0, v1), (v1, v2), (v2, v0)):
            edge = (
                (first, second)
                if first < second
                else (second, first)
            )
            edge_counts[edge] = edge_counts.get(edge, 0) + 1

    for edge, count in edge_counts.items():
        if count not in (1, 2):
            raise ValueError(
                f"检测到非流形 mesh edge {edge}：共享 triangle 数={count}"
            )
    return edge_counts


def _build_boundary_adjacency(
    edge_counts: dict[tuple[int, int], int],
) -> dict[int, tuple[int, int]]:
    mutable: dict[int, list[int]] = {}
    boundary_edges = [
        edge for edge, count in edge_counts.items() if count == 1
    ]
    if not boundary_edges:
        raise ValueError("Mesh2D 不含拓扑外边界 edge")

    for first, second in boundary_edges:
        mutable.setdefault(first, []).append(second)
        mutable.setdefault(second, []).append(first)

    adjacency: dict[int, tuple[int, int]] = {}
    for node, neighbors in mutable.items():
        if len(neighbors) != 2:
            raise ValueError(
                "拓扑外边界必须形成唯一简单环："
                f"boundary node {node} 的 degree={len(neighbors)}"
            )
        adjacency[node] = (neighbors[0], neighbors[1])
    return adjacency


def _walk_boundary_cycle(
    adjacency: dict[int, tuple[int, int]],
    start: int,
    first_neighbor: int,
) -> tuple[int, ...]:
    cycle = [start]
    previous = start
    current = first_neighbor

    while current != start:
        if current in cycle:
            raise ValueError(
                "拓扑外边界遍历在返回起点前重复访问节点，边界不是简单环"
            )
        cycle.append(current)
        neighbors = adjacency.get(current)
        if neighbors is None or len(neighbors) != 2:
            raise ValueError("拓扑外边界邻接信息不完整")

        next_node = (
            neighbors[1] if neighbors[0] == previous else neighbors[0]
        )
        previous, current = current, next_node

        if len(cycle) > len(adjacency):
            raise ValueError("拓扑外边界遍历超过 boundary node 数")

    if len(cycle) != len(adjacency):
        raise ValueError(
            "拓扑外边界不是唯一闭环："
            f"遍历节点数={len(cycle)}，boundary node 数={len(adjacency)}"
        )
    return tuple(cycle)


def _loop_entities(
    spec: M3DC1ModelSpec,
):
    spec.validate()

    if len(spec.loops) != 1 or len(spec.faces) != 1:
        raise ValueError(
            "当前 writer 只支持 single-loop、single-face M3DC1ModelSpec"
        )

    loop = spec.loops[0]
    face = spec.faces[0]
    edge_by_tag = {edge.tag: edge for edge in spec.edges}
    vertex_by_tag = {vertex.tag: vertex for vertex in spec.vertices}

    directed_edges = []
    for edge_tag in loop.edge_tags:
        edge = edge_by_tag.get(edge_tag)
        if edge is None:
            raise ValueError(
                f"geometric loop {loop.tag} 引用不存在的 edge {edge_tag}"
            )
        if edge.curve.serialized_curve_type is not CurveType.LIN:
            raise ValueError(
                "native SMB exchange schema v1 只支持 LIN geometric edges，"
                f"edge {edge.tag} curve_type="
                f"{int(edge.curve.serialized_curve_type)}"
            )
        directed_edges.append(edge)

    directed_vertex_tags = tuple(
        edge.begin_vertex_tag for edge in directed_edges
    )
    if len(set(directed_vertex_tags)) != len(directed_vertex_tags):
        raise ValueError("geometric loop 的 directed corner tag 存在重复")

    for tag in directed_vertex_tags:
        if tag not in vertex_by_tag:
            raise ValueError(
                f"geometric edge 引用不存在的 geometric vertex {tag}"
            )

    return (
        loop,
        face,
        tuple(directed_edges),
        directed_vertex_tags,
        vertex_by_tag,
    )


def _match_exact_corner_nodes(
    nodes: np.ndarray,
    adjacency: dict[int, tuple[int, int]],
    directed_vertex_tags: tuple[int, ...],
    vertex_by_tag,
) -> tuple[int, ...]:
    boundary_nodes = set(adjacency)
    matched: list[int] = []

    for vertex_tag in directed_vertex_tags:
        vertex = vertex_by_tag[vertex_tag]
        coordinate = np.array([vertex.r, vertex.z], dtype=np.float64)
        candidates = np.flatnonzero(np.all(nodes == coordinate, axis=1))

        if candidates.size != 1:
            raise ValueError(
                "geometric corner 必须与恰好一个 Mesh2D node 逐分量精确相等："
                f"vertex tag={vertex_tag}，coordinate={coordinate.tolist()}，"
                f"候选数={int(candidates.size)}"
            )

        node_index = int(candidates[0])
        if node_index not in boundary_nodes:
            raise ValueError(
                f"geometric vertex {vertex_tag} 对应 mesh node "
                f"{node_index} 不是拓扑外边界节点"
            )
        matched.append(node_index)

    if len(set(matched)) != len(matched):
        raise ValueError("多个 geometric corners 映射到同一个 mesh node")
    return tuple(matched)


def _corner_positions(
    cycle: tuple[int, ...],
    corner_nodes: tuple[int, ...],
) -> tuple[int, ...] | None:
    positions_by_node = {
        node: index for index, node in enumerate(cycle)
    }
    try:
        positions = tuple(
            positions_by_node[node] for node in corner_nodes
        )
    except KeyError:
        return None

    if positions[0] != 0:
        return None
    if any(
        positions[index] >= positions[index + 1]
        for index in range(len(positions) - 1)
    ):
        return None
    return positions


def _select_oriented_boundary_cycle(
    adjacency: dict[int, tuple[int, int]],
    corner_nodes: tuple[int, ...],
) -> tuple[tuple[int, ...], tuple[int, ...]]:
    start = corner_nodes[0]
    neighbors = adjacency[start]
    valid: list[tuple[tuple[int, ...], tuple[int, ...]]] = []

    for neighbor in neighbors:
        cycle = _walk_boundary_cycle(adjacency, start, neighbor)
        positions = _corner_positions(cycle, corner_nodes)
        if positions is not None:
            valid.append((cycle, positions))

    if len(valid) != 1:
        raise ValueError(
            "无法唯一确定拓扑外边界方向与 geometric loop 方向的对应关系："
            f"有效方向数={len(valid)}"
        )
    return valid[0]


def _derive_classification(
    nodes: np.ndarray,
    triangles: np.ndarray,
    spec: M3DC1ModelSpec,
):
    edge_counts = _build_edge_counts(triangles)
    adjacency = _build_boundary_adjacency(edge_counts)

    (
        _loop,
        face,
        directed_edges,
        directed_vertex_tags,
        vertex_by_tag,
    ) = _loop_entities(spec)

    corner_nodes = _match_exact_corner_nodes(
        nodes,
        adjacency,
        directed_vertex_tags,
        vertex_by_tag,
    )
    cycle, positions = _select_oriented_boundary_cycle(
        adjacency,
        corner_nodes,
    )

    face_class = (2, face.tag)
    vertex_classes = [face_class for _ in range(nodes.shape[0])]
    edge_classes: dict[tuple[int, int], tuple[int, int]] = {
        edge: face_class for edge in edge_counts
    }

    for vertex_tag, node_index in zip(
        directed_vertex_tags,
        corner_nodes,
        strict=True,
    ):
        vertex_classes[node_index] = (0, vertex_tag)

    cycle_size = len(cycle)
    for index, geometric_edge in enumerate(directed_edges):
        start_position = positions[index]
        end_position = (
            positions[index + 1]
            if index + 1 < len(positions)
            else cycle_size
        )
        if end_position <= start_position:
            raise ValueError(
                f"geometric edge {geometric_edge.tag} 对应空 boundary chain"
            )

        for position in range(start_position, end_position):
            first = cycle[position % cycle_size]
            second = cycle[(position + 1) % cycle_size]
            mesh_edge = (
                (first, second)
                if first < second
                else (second, first)
            )
            if edge_counts.get(mesh_edge) != 1:
                raise ValueError(
                    f"geometric edge {geometric_edge.tag} 对应的 "
                    f"mesh edge {mesh_edge} 不是拓扑外边界 edge"
                )
            edge_classes[mesh_edge] = (1, geometric_edge.tag)

        for position in range(start_position + 1, end_position):
            node_index = cycle[position]
            if node_index in corner_nodes:
                raise ValueError(
                    "boundary chain 内出现非预期 geometric corner："
                    f"mesh node={node_index}"
                )
            vertex_classes[node_index] = (1, geometric_edge.tag)

    classified_edges = tuple(
        (
            first,
            second,
            edge_classes[(first, second)][0],
            edge_classes[(first, second)][1],
        )
        for first, second in sorted(edge_counts)
    )
    triangle_classes = tuple(
        face_class for _ in range(triangles.shape[0])
    )

    return (
        tuple(vertex_classes),
        classified_edges,
        triangle_classes,
    )


def _validate_output_base(filepath: str | os.PathLike[str]) -> Path:
    output = Path(filepath)
    if output.suffix != ".smb":
        raise ValueError(
            f"SMB 基础输出路径必须以 .smb 结尾，实际={output}"
        )
    if output.name == ".smb":
        raise ValueError("SMB 基础输出文件名不能为空")
    if output.stem.endswith("0"):
        raise ValueError(
            "SMB 输出参数必须是未插入 rank 的基础 .smb 路径；"
            f"疑似传入 rank-0 物理路径：{output}"
        )

    parent = output.parent
    if not parent.exists():
        raise ValueError(f"SMB 输出父目录不存在：{parent}")
    if not parent.is_dir():
        raise ValueError(f"SMB 输出父路径不是目录：{parent}")
    if output.exists():
        raise ValueError(f"SMB 基础输出路径已存在：{output}")

    rank0 = _rank0_path(output)
    if rank0.exists():
        raise ValueError(f"SMB rank-0 输出路径已存在：{rank0}")

    return output.absolute()


def _rank0_path(output_base: Path) -> Path:
    return output_base.with_name(
        f"{output_base.stem}0{output_base.suffix}"
    )


def _prepare_exchange(
    mesh: Mesh2D,
    spec: M3DC1ModelSpec,
    filepath: str | os.PathLike[str],
) -> _PreparedExchange:
    if not isinstance(mesh, Mesh2D):
        raise TypeError(
            f"mesh 必须是 Mesh2D，实际={type(mesh).__name__}"
        )
    if not isinstance(spec, M3DC1ModelSpec):
        raise TypeError(
            f"spec 必须是 M3DC1ModelSpec，实际={type(spec).__name__}"
        )

    spec.validate()
    nodes = _as_nodes(mesh.nodes)
    triangles = _as_triangles(mesh.triangles, nodes.shape[0])
    region_ids = _as_single_region_ids(
        mesh.region_ids,
        triangles.shape[0],
    )
    _validate_triangle_geometry(nodes, triangles)

    vertex_classes, mesh_edges, triangle_classes = (
        _derive_classification(nodes, triangles, spec)
    )
    output_base = _validate_output_base(filepath)

    return _PreparedExchange(
        schema_version=_SCHEMA_VERSION,
        nodes=nodes,
        triangles=triangles,
        region_ids=region_ids,
        mesh_edges=mesh_edges,
        vertex_classes=vertex_classes,
        triangle_classes=triangle_classes,
        spec=spec,
        output_base=output_base,
    )


def _prepare_build_result_exchange(
    mesh: Mesh2D,
    result: M3DC1BuildResult,
    filepath: str | os.PathLike[str],
) -> _PreparedExchange:
    if not isinstance(mesh, Mesh2D):
        raise TypeError(f"mesh 必须是 Mesh2D，实际={type(mesh).__name__}")
    if not isinstance(result, M3DC1BuildResult):
        raise TypeError(
            f"result 必须是 M3DC1BuildResult，实际={type(result).__name__}"
        )

    nodes = _as_nodes(mesh.nodes)
    triangles = _as_triangles(mesh.triangles, nodes.shape[0])
    region_values = np.asarray(mesh.region_ids, dtype=np.float64)
    if (
        region_values.ndim != 1
        or region_values.shape[0] != triangles.shape[0]
        or not np.all(np.isfinite(region_values))
        or not np.all(region_values == np.round(region_values))
        or np.any(region_values <= 0)
    ):
        raise ValueError("Mesh2D.region_ids 不满足 multi-face exchange 契约")
    region_ids = region_values.astype(np.int64)
    if set(int(value) for value in np.unique(region_ids)) != set(
        result.region_id_to_face_tag
    ):
        raise ValueError(
            "M3DC1BuildResult 的 Region ID mapping 与 Mesh2D 不一致"
        )
    _validate_triangle_geometry(nodes, triangles)

    edge_counts = _build_edge_counts(triangles)
    if tuple(sorted(edge_counts)) != result.mesh_edges:
        raise ValueError(
            "M3DC1BuildResult mesh edge table 与 Mesh2D topology 不一致"
        )
    if len(result.vertex_classifications) != nodes.shape[0]:
        raise ValueError("vertex classifications 数量与 Mesh2D.nodes 不一致")
    if len(result.triangle_classifications) != triangles.shape[0]:
        raise ValueError("triangle classifications 数量与 triangles 不一致")

    mesh_edges = tuple(
        (
            first,
            second,
            classification.dimension,
            classification.tag,
        )
        for (first, second), classification in zip(
            result.mesh_edges,
            result.edge_classifications,
            strict=True,
        )
    )
    vertex_classes = tuple(
        (value.dimension, value.tag)
        for value in result.vertex_classifications
    )
    triangle_classes = tuple(
        (value.dimension, value.tag)
        for value in result.triangle_classifications
    )
    return _PreparedExchange(
        schema_version=_MULTI_FACE_SCHEMA_VERSION,
        nodes=nodes,
        triangles=triangles,
        region_ids=np.array(region_ids, copy=True),
        mesh_edges=mesh_edges,
        vertex_classes=vertex_classes,
        triangle_classes=triangle_classes,
        spec=result.spec,
        output_base=_validate_output_base(filepath),
    )


def _write_i32(file_object, value: int) -> None:
    file_object.write(_I32.pack(int(value)))


def _write_u64(file_object, value: int) -> None:
    if value < 0:
        raise ValueError(f"uint64 值不得为负数：{value}")
    file_object.write(_U64.pack(int(value)))


def _write_exchange(
    prepared: _PreparedExchange,
    exchange_path: Path,
) -> None:
    spec = prepared.spec
    output_bytes = os.fsencode(prepared.output_base)

    with exchange_path.open("wb") as stream:
        stream.write(_MAGIC)
        stream.write(_U32.pack(prepared.schema_version))
        stream.write(_U32.pack(_ENDIAN_MARKER))
        stream.write(
            _COUNTS.pack(
                prepared.nodes.shape[0],
                prepared.triangles.shape[0],
                len(prepared.mesh_edges),
                len(spec.vertices),
                len(spec.edges),
                len(spec.loops),
                len(spec.faces),
            )
        )
        stream.write(
            _SPECIAL_LOOPS.pack(
                spec.separatrix_loop,
                spec.inner_wall_loop,
                spec.outer_wall_loop,
                spec.vacuum_loop,
            )
        )
        _write_u64(stream, len(output_bytes))
        stream.write(output_bytes)

        stream.write(
            np.asarray(
                prepared.nodes,
                dtype="<f8",
                order="C",
            ).tobytes(order="C")
        )
        stream.write(
            np.asarray(
                prepared.triangles,
                dtype="<u8",
                order="C",
            ).tobytes(order="C")
        )

        for vertex in spec.vertices:
            stream.write(
                _GEOMETRIC_VERTEX.pack(
                    vertex.tag,
                    vertex.r,
                    vertex.z,
                    vertex.y,
                )
            )

        for edge in spec.edges:
            stream.write(
                _GEOMETRIC_EDGE.pack(
                    edge.tag,
                    edge.begin_vertex_tag,
                    edge.end_vertex_tag,
                    int(edge.curve.serialized_curve_type),
                )
            )

        for loop in spec.loops:
            _write_i32(stream, loop.tag)
            _write_u64(stream, len(loop.edge_tags))
            for edge_tag in loop.edge_tags:
                _write_i32(stream, edge_tag)

        for face in spec.faces:
            _write_i32(stream, face.tag)
            _write_u64(stream, len(face.loop_tags))
            for loop_tag in face.loop_tags:
                _write_i32(stream, loop_tag)

        for dimension, tag in prepared.vertex_classes:
            stream.write(_CLASSIFICATION.pack(dimension, tag))

        for node0, node1, dimension, tag in prepared.mesh_edges:
            stream.write(
                _MESH_EDGE.pack(
                    node0,
                    node1,
                    dimension,
                    tag,
                )
            )

        for dimension, tag in prepared.triangle_classes:
            stream.write(_CLASSIFICATION.pack(dimension, tag))

        stream.flush()
        os.fsync(stream.fileno())


def _resolve_helper(
    helper_path: str | os.PathLike[str] | None,
) -> Path:
    if helper_path is not None:
        candidate = Path(helper_path)
        source = "helper_path"
    else:
        environment_value = os.environ.get(_HELPER_ENV)
        if environment_value:
            candidate = Path(environment_value)
            source = _HELPER_ENV
        else:
            located = shutil.which(_HELPER_NAME)
            if located is None:
                raise SMBWriterError(
                    "helper 定位失败：未提供 helper_path，"
                    f"未设置 {_HELPER_ENV}，且 PATH 中不存在 {_HELPER_NAME}"
                )
            candidate = Path(located)
            source = "PATH"

    if not candidate.exists():
        raise SMBWriterError(
            f"helper 定位失败：{source} 指向的文件不存在：{candidate}"
        )
    if not candidate.is_file():
        raise SMBWriterError(
            f"helper 定位失败：{source} 指向的路径不是普通文件：{candidate}"
        )
    if not os.access(candidate, os.X_OK):
        raise SMBWriterError(
            f"helper 定位失败：{source} 指向的文件不可执行：{candidate}"
        )
    return candidate.absolute()


def _extract_native_error(stderr: str) -> str:
    structured = [
        line.strip()
        for line in stderr.splitlines()
        if line.startswith("LINKMESH_M4_ERROR ")
    ]
    if structured:
        return structured[-1]
    stripped = stderr.strip()
    return stripped[-2000:] if stripped else "helper 未提供错误摘要"


def _parse_success_marker(stdout: str) -> tuple[Path, Path]:
    markers = [
        line.strip()
        for line in stdout.splitlines()
        if line.startswith("LINKMESH_M4_OK ")
    ]
    if len(markers) != 1:
        raise SMBWriterError(
            "helper 协议错误：成功退出但未返回唯一 "
            "LINKMESH_M4_OK 标记"
        )

    fields: dict[str, str] = {}
    for token in markers[0].split()[1:]:
        key, separator, value = token.partition("=")
        if not separator or key in fields:
            raise SMBWriterError(
                f"helper 协议错误：非法成功字段 {token!r}"
            )
        fields[key] = unquote(value)

    if set(fields) != {"base", "rank0"}:
        raise SMBWriterError(
            "helper 协议错误：LINKMESH_M4_OK 必须且只能包含 "
            "base 与 rank0 字段"
        )
    return Path(fields["base"]), Path(fields["rank0"])


def write_smb(
    mesh: Mesh2D,
    spec: M3DC1ModelSpec | M3DC1BuildResult,
    filepath: str | os.PathLike[str],
    *,
    helper_path: str | os.PathLike[str] | None = None,
) -> None:
    """Write a serial rank-0 native PUMI SMB through the standalone helper.

    Helper lookup priority is:

    1. Explicit ``helper_path``.
    2. ``LINKMESH_SCOREC_HELPER``.
    3. ``linkmesh-scorec-helper`` on ``PATH``.

    The destination is the base ``.smb`` path accepted by PUMI
    ``writeNative``. For SCOREC/PUMI 2.2.9 the expected physical rank-0 file
    is ``<base>0.smb``.

    The private exchange file is removed only after helper success and
    successful rank-0 output verification. It is retained on every failure
    after creation, and its path is included in the raised exception.
    """
    prepared = (
        _prepare_build_result_exchange(mesh, spec, filepath)
        if isinstance(spec, M3DC1BuildResult)
        else _prepare_exchange(mesh, spec, filepath)
    )
    helper = _resolve_helper(helper_path)

    temporary_file = tempfile.NamedTemporaryFile(
        mode="wb",
        dir=prepared.output_base.parent,
        prefix=f".{prepared.output_base.stem}.",
        suffix=".lm4xchg",
        delete=False,
    )
    exchange_path = Path(temporary_file.name)
    temporary_file.close()

    try:
        _write_exchange(prepared, exchange_path)

        try:
            completed = subprocess.run(
                [str(helper), "--input", str(exchange_path)],
                check=False,
                capture_output=True,
                text=True,
                timeout=_HELPER_TIMEOUT_SECONDS,
            )
        except subprocess.TimeoutExpired as exc:
            raise SMBWriterError(
                "helper 调用超时："
                f"stage=helper_timeout，timeout={_HELPER_TIMEOUT_SECONDS}s，"
                f"exchange={exchange_path}"
            ) from exc
        except OSError as exc:
            raise SMBWriterError(
                "helper 启动失败："
                f"stage=helper_start，helper={helper}，"
                f"exchange={exchange_path}，error={exc}"
            ) from exc

        if completed.returncode != 0:
            summary = _extract_native_error(completed.stderr)
            raise SMBWriterError(
                "helper 执行失败："
                f"stage=helper_native，exit_code={completed.returncode}，"
                f"native_error={summary}，exchange={exchange_path}"
            )

        reported_base, reported_rank0 = _parse_success_marker(
            completed.stdout
        )
        expected_base = prepared.output_base
        expected_rank0 = _rank0_path(expected_base)

        if reported_base.absolute() != expected_base:
            raise SMBWriterError(
                "helper 协议错误：报告的 base 与请求不一致："
                f"reported={reported_base}，expected={expected_base}，"
                f"exchange={exchange_path}"
            )
        if reported_rank0.absolute() != expected_rank0:
            raise SMBWriterError(
                "helper 协议错误：报告的 rank0 路径与目标版本规则不一致："
                f"reported={reported_rank0}，expected={expected_rank0}，"
                f"exchange={exchange_path}"
            )
        if not expected_rank0.exists():
            raise SMBWriterError(
                "helper 成功但 rank-0 SMB 不存在："
                f"path={expected_rank0}，exchange={exchange_path}"
            )
        if not expected_rank0.is_file():
            raise SMBWriterError(
                "helper 成功但 rank-0 SMB 不是普通文件："
                f"path={expected_rank0}，exchange={exchange_path}"
            )
        if expected_rank0.stat().st_size <= 0:
            raise SMBWriterError(
                "helper 成功但 rank-0 SMB 为空："
                f"path={expected_rank0}，exchange={exchange_path}"
            )
    except Exception:
        raise
    else:
        exchange_path.unlink()
