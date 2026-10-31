"""
M3D-C1 single-face analytic geometric model specification.

This module implements an in-memory target model and the adapter from
Mesh2D.outer_boundary. It does not depend on the native SMB backend.

The TXT representation follows the M3D-C1 model interchange structure.

The initial model supports exactly one region, one geometric face, and one
geometric loop. The loop is represented by ordered linear geometric edges.
"""

from __future__ import annotations

from dataclasses import dataclass, field
from enum import IntEnum
from numbers import Integral
from types import MappingProxyType
from typing import TYPE_CHECKING, Any, Mapping

import numpy as np

from linkmesh.mesh import (
    Mesh2D,
    _validate_outer_boundary as _mesh_validate_outer_boundary,
)

if TYPE_CHECKING:
    from collections.abc import Iterable


class CurveType(IntEnum):
    """M3D-C1 curve type numeric codes."""

    LIN = 1
    POLY = 2
    BSPLINE = 3


def _require_positive_tag(value: Any, name: str) -> int:
    """Validate and return a positive integer geometric tag."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须是正整数，实际={value!r}")
    tag = int(value)
    if tag <= 0:
        raise ValueError(f"{name} 必须是正整数，实际={tag}")
    return tag


def _require_special_loop_tag(value: Any, name: str) -> int:
    """Validate a special loop tag, where -1 means absent."""
    if isinstance(value, bool) or not isinstance(value, Integral):
        raise ValueError(f"{name} 必须是 -1 或正整数，实际={value!r}")
    tag = int(value)
    if tag == -1 or tag > 0:
        return tag
    raise ValueError(f"{name} 必须是 -1 或正整数，实际={tag}")


def _require_contiguous_tags(tags: list[int], entity_name: str) -> None:
    """Require deterministic contiguous 1-based tags for the P0 model."""
    expected = list(range(1, len(tags) + 1))
    if tags != expected:
        raise ValueError(
            f"{entity_name} tag 必须按确定性 1-based 连续编号；"
            f"期望={expected}，实际={tags}"
        )


def _normalize_finite_pair(value: Any, name: str) -> tuple[float, float]:
    """Normalize a coordinate pair to an immutable finite float tuple."""
    array = np.asarray(value, dtype=np.float64)
    if array.shape != (2,):
        raise ValueError(f"{name} 必须为二维坐标对，实际 shape={array.shape}")
    if not np.all(np.isfinite(array)):
        raise ValueError(f"{name} 含非有限值（NaN/Inf）")
    return float(array[0]), float(array[1])


@dataclass(frozen=True)
class M3DC1Curve:
    """Curve representation serialized after an edge header.

    LIN has no additional fields.

    POLY and BSPLINE carry the same serialized spline payload: order, number
    of control points, order+num_points knots, followed by num_points pairs of
    R/Z control coordinates. POLY is written with the BSPLINE numeric code.
    """

    curve_type: CurveType = CurveType.LIN
    order: int | None = None
    knots: tuple[float, ...] = ()
    control_points: tuple[tuple[float, float], ...] = ()

    def __post_init__(self) -> None:
        try:
            curve_type = CurveType(self.curve_type)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"curve_type 必须是 LIN(1)、POLY(2) 或 BSPLINE(3)，"
                f"实际={self.curve_type!r}"
            ) from exc

        object.__setattr__(self, "curve_type", curve_type)

        knots = tuple(float(value) for value in self.knots)
        control_points = tuple(
            _normalize_finite_pair(point, f"control_points[{index}]")
            for index, point in enumerate(self.control_points)
        )
        object.__setattr__(self, "knots", knots)
        object.__setattr__(self, "control_points", control_points)

        self.validate()

    @property
    def serialized_curve_type(self) -> CurveType:
        """Return the curve code required by the model format."""
        if self.curve_type is CurveType.POLY:
            return CurveType.BSPLINE
        return self.curve_type

    def validate(self) -> None:
        """Validate curve type and required serialized payload."""
        if self.curve_type is CurveType.LIN:
            if self.order is not None or self.knots or self.control_points:
                raise ValueError(
                    "LIN 曲线不得包含 order、knots 或 control_points 附加字段"
                )
            return

        if isinstance(self.order, bool) or not isinstance(self.order, Integral):
            raise ValueError(
                f"{self.curve_type.name} 曲线的 order 必须是正整数，"
                f"实际={self.order!r}"
            )

        order = int(self.order)
        if order <= 0:
            raise ValueError(
                f"{self.curve_type.name} 曲线的 order 必须大于 0，实际={order}"
            )

        num_points = len(self.control_points)
        if num_points < 2:
            raise ValueError(
                f"{self.curve_type.name} 曲线至少需要 2 个控制点，"
                f"实际={num_points}"
            )
        if num_points < order:
            raise ValueError(
                f"{self.curve_type.name} 曲线控制点数不得小于 order："
                f"num_points={num_points}，order={order}"
            )

        expected_knots = order + num_points
        if len(self.knots) != expected_knots:
            raise ValueError(
                f"{self.curve_type.name} 曲线 knot 数量错误："
                f"期望 order+num_points={expected_knots}，"
                f"实际={len(self.knots)}"
            )

        knot_array = np.asarray(self.knots, dtype=np.float64)
        if not np.all(np.isfinite(knot_array)):
            raise ValueError(
                f"{self.curve_type.name} 曲线 knots 含非有限值（NaN/Inf）"
            )
        if np.any(np.diff(knot_array) < 0.0):
            raise ValueError(
                f"{self.curve_type.name} 曲线 knots 必须单调不减"
            )


@dataclass(frozen=True)
class GeometricVertex:
    """M3D-C1 geometric vertex."""

    tag: int
    r: float
    z: float
    y: float = 0.0

    def __post_init__(self) -> None:
        tag = _require_positive_tag(self.tag, "geometric vertex tag")
        coordinates = np.asarray([self.r, self.z, self.y], dtype=np.float64)
        if not np.all(np.isfinite(coordinates)):
            raise ValueError(
                f"geometric vertex {tag} 坐标含非有限值（NaN/Inf）"
            )
        if float(self.y) != 0.0:
            raise ValueError(
                f"geometric vertex {tag} 的第三坐标必须为 0，"
                f"实际={self.y}"
            )

        object.__setattr__(self, "tag", tag)
        object.__setattr__(self, "r", float(self.r))
        object.__setattr__(self, "z", float(self.z))
        object.__setattr__(self, "y", 0.0)


@dataclass(frozen=True)
class GeometricEdge:
    """Directed M3D-C1 geometric edge."""

    tag: int
    begin_vertex_tag: int
    end_vertex_tag: int
    curve: M3DC1Curve = field(default_factory=M3DC1Curve)

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tag",
            _require_positive_tag(self.tag, "geometric edge tag"),
        )
        object.__setattr__(
            self,
            "begin_vertex_tag",
            _require_positive_tag(
                self.begin_vertex_tag,
                f"geometric edge {self.tag} begin vertex tag",
            ),
        )
        object.__setattr__(
            self,
            "end_vertex_tag",
            _require_positive_tag(
                self.end_vertex_tag,
                f"geometric edge {self.tag} end vertex tag",
            ),
        )
        if self.begin_vertex_tag == self.end_vertex_tag:
            raise ValueError(
                f"geometric edge {self.tag} 起点和终点相同："
                f"vertex={self.begin_vertex_tag}，形成零长度拓扑边"
            )
        if not isinstance(self.curve, M3DC1Curve):
            raise ValueError(
                f"geometric edge {self.tag} 的 curve 必须是 M3DC1Curve，"
                f"实际={type(self.curve).__name__}"
            )
        self.curve.validate()


@dataclass(frozen=True)
class GeometricLoop:
    """Ordered edge references forming one closed geometric loop."""

    tag: int
    edge_tags: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tag",
            _require_positive_tag(self.tag, "geometric loop tag"),
        )
        edge_tags = tuple(
            _require_positive_tag(value, f"geometric loop {self.tag} edge tag")
            for value in self.edge_tags
        )
        object.__setattr__(self, "edge_tags", edge_tags)

        if len(edge_tags) < 3:
            raise ValueError(
                f"geometric loop {self.tag} 至少需要 3 条边，"
                f"实际={len(edge_tags)}"
            )
        if len(set(edge_tags)) != len(edge_tags):
            raise ValueError(
                f"geometric loop {self.tag} 含重复 edge tag：{edge_tags}"
            )


@dataclass(frozen=True)
class GeometricFace:
    """M3D-C1 geometric face and its bounding loop references."""

    tag: int
    loop_tags: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "tag",
            _require_positive_tag(self.tag, "geometric face tag"),
        )
        loop_tags = tuple(
            _require_positive_tag(value, f"geometric face {self.tag} loop tag")
            for value in self.loop_tags
        )
        object.__setattr__(self, "loop_tags", loop_tags)

        if not loop_tags:
            raise ValueError(
                f"geometric face {self.tag} 至少必须引用 1 个 loop"
            )
        if len(set(loop_tags)) != len(loop_tags):
            raise ValueError(
                f"geometric face {self.tag} 含重复 loop tag：{loop_tags}"
            )


@dataclass(frozen=True)
class M3DC1ModelSpec:
    """Validated single-region, single-face M3D-C1 analytic model."""

    vertices: tuple[GeometricVertex, ...]
    edges: tuple[GeometricEdge, ...]
    loops: tuple[GeometricLoop, ...]
    faces: tuple[GeometricFace, ...]
    separatrix_loop: int = -1
    inner_wall_loop: int = 1
    outer_wall_loop: int = -1
    vacuum_loop: int = -1

    def __post_init__(self) -> None:
        object.__setattr__(self, "vertices", tuple(self.vertices))
        object.__setattr__(self, "edges", tuple(self.edges))
        object.__setattr__(self, "loops", tuple(self.loops))
        object.__setattr__(self, "faces", tuple(self.faces))
        object.__setattr__(
            self,
            "separatrix_loop",
            _require_special_loop_tag(
                self.separatrix_loop,
                "separatrix_loop",
            ),
        )
        object.__setattr__(
            self,
            "inner_wall_loop",
            _require_special_loop_tag(
                self.inner_wall_loop,
                "inner_wall_loop",
            ),
        )
        object.__setattr__(
            self,
            "outer_wall_loop",
            _require_special_loop_tag(
                self.outer_wall_loop,
                "outer_wall_loop",
            ),
        )
        object.__setattr__(
            self,
            "vacuum_loop",
            _require_special_loop_tag(
                self.vacuum_loop,
                "vacuum_loop",
            ),
        )
        self.validate()

    @property
    def num_loops(self) -> int:
        """Number written as the first integer in the model header."""
        return len(self.loops)

    @property
    def num_faces(self) -> int:
        """Number written before the face records."""
        return len(self.faces)

    def validate(self) -> None:
        """Validate a deterministic single-face or multi-face model."""
        if not self.loops:
            raise ValueError("M3D-C1 model 至少需要 1 个 geometric loop")
        if not self.faces:
            raise ValueError("M3D-C1 model 至少需要 1 个 geometric face")
        if len(self.vertices) < 3:
            raise ValueError(
                "M3D-C1 model 至少需要 3 个 geometric vertices，"
                f"实际={len(self.vertices)}"
            )
        if not self.edges:
            raise ValueError("M3D-C1 model 至少需要 1 条 geometric edge")

        for index, vertex in enumerate(self.vertices):
            if not isinstance(vertex, GeometricVertex):
                raise ValueError(
                    f"vertices[{index}] 必须是 GeometricVertex，"
                    f"实际={type(vertex).__name__}"
                )
        for index, edge in enumerate(self.edges):
            if not isinstance(edge, GeometricEdge):
                raise ValueError(
                    f"edges[{index}] 必须是 GeometricEdge，"
                    f"实际={type(edge).__name__}"
                )
        for index, loop in enumerate(self.loops):
            if not isinstance(loop, GeometricLoop):
                raise ValueError(
                    f"loops[{index}] 必须是 GeometricLoop，"
                    f"实际={type(loop).__name__}"
                )
        for index, face in enumerate(self.faces):
            if not isinstance(face, GeometricFace):
                raise ValueError(
                    f"faces[{index}] 必须是 GeometricFace，"
                    f"实际={type(face).__name__}"
                )

        vertex_tags = [vertex.tag for vertex in self.vertices]
        edge_tags = [edge.tag for edge in self.edges]
        loop_tags = [loop.tag for loop in self.loops]
        face_tags = [face.tag for face in self.faces]

        for tags, entity_name in (
            (vertex_tags, "geometric vertex"),
            (edge_tags, "geometric edge"),
            (loop_tags, "geometric loop"),
            (face_tags, "geometric face"),
        ):
            if len(set(tags)) != len(tags):
                raise ValueError(f"{entity_name} tag 存在重复：{tags}")
            _require_contiguous_tags(tags, entity_name)

        vertex_tag_set = set(vertex_tags)
        edge_by_tag = {edge.tag: edge for edge in self.edges}
        loop_tag_set = set(loop_tags)

        for edge in self.edges:
            if edge.begin_vertex_tag not in vertex_tag_set:
                raise ValueError(
                    f"geometric edge {edge.tag} 起点引用不存在的 vertex "
                    f"{edge.begin_vertex_tag}"
                )
            if edge.end_vertex_tag not in vertex_tag_set:
                raise ValueError(
                    f"geometric edge {edge.tag} 终点引用不存在的 vertex "
                    f"{edge.end_vertex_tag}"
                )
            edge.curve.validate()

        edge_use_count = {tag: 0 for tag in edge_tags}
        for loop in self.loops:
            for index, edge_tag in enumerate(loop.edge_tags):
                if edge_tag not in edge_by_tag:
                    raise ValueError(
                        f"geometric loop {loop.tag} 引用不存在的 edge {edge_tag}"
                    )
                edge_use_count[edge_tag] += 1
                current_edge = edge_by_tag[edge_tag]
                next_edge = edge_by_tag[
                    loop.edge_tags[(index + 1) % len(loop.edge_tags)]
                ]
                if current_edge.end_vertex_tag != next_edge.begin_vertex_tag:
                    raise ValueError(
                        f"geometric loop {loop.tag} 不连续：edge "
                        f"{current_edge.tag} 结束于 vertex "
                        f"{current_edge.end_vertex_tag}，下一条 edge "
                        f"{next_edge.tag} 起始于 vertex "
                        f"{next_edge.begin_vertex_tag}"
                    )

        if any(count != 1 for count in edge_use_count.values()):
            raise ValueError(
                "每条 geometric edge 必须且只能属于一个 geometric loop"
            )

        loop_use_count = {tag: 0 for tag in loop_tags}
        for face in self.faces:
            for loop_tag in face.loop_tags:
                if loop_tag not in loop_tag_set:
                    raise ValueError(
                        f"geometric face {face.tag} 引用不存在的 loop {loop_tag}"
                    )
                loop_use_count[loop_tag] += 1

        invalid_loop_uses = {
            tag: count
            for tag, count in loop_use_count.items()
            if count not in (1, 2)
        }
        if invalid_loop_uses:
            raise ValueError(
                "每个 geometric loop 必须被一个或两个 faces 引用，"
                f"实际={invalid_loop_uses}"
            )

        special_tags = (
            self.separatrix_loop,
            self.inner_wall_loop,
            self.outer_wall_loop,
            self.vacuum_loop,
        )
        unknown_special = {
            tag for tag in special_tags if tag != -1 and tag not in loop_tag_set
        }
        if unknown_special:
            raise ValueError(
                f"special loop tags 引用了不存在的 loops：{sorted(unknown_special)}"
            )

        if len(self.loops) == 1 and len(self.faces) == 1:
            loop = self.loops[0]
            face = self.faces[0]
            if set(loop.edge_tags) != set(edge_tags):
                raise ValueError(
                    f"single-face loop {loop.tag} 必须引用全部 geometric edges"
                )
            if face.loop_tags != (loop.tag,):
                raise ValueError(
                    f"single-face 模型必须引用唯一 loop {loop.tag}"
                )
            if special_tags != (-1, loop.tag, -1, -1):
                raise ValueError(
                    "single-face 模型必须保持 special loop tags "
                    f"(-1, {loop.tag}, -1, -1)"
                )
        elif any(tag != -1 for tag in special_tags):
            raise ValueError(
                "multi-face 首版的四个 special loop tags 必须全部为 -1"
            )


@dataclass(frozen=True, order=True)
class ModelClassification:
    """One immutable geometric classification pair."""

    dimension: int
    tag: int

    def __post_init__(self) -> None:
        if (
            isinstance(self.dimension, bool)
            or not isinstance(self.dimension, Integral)
            or int(self.dimension) not in (0, 1, 2)
        ):
            raise ValueError(
                f"classification dimension 必须为 0、1 或 2，实际={self.dimension!r}"
            )
        object.__setattr__(self, "dimension", int(self.dimension))
        object.__setattr__(
            self,
            "tag",
            _require_positive_tag(self.tag, "classification tag"),
        )


@dataclass(frozen=True)
class TopologyCycle:
    """Canonical outer-boundary or Region-interface cycle."""

    loop_tag: int
    node_indices: tuple[int, ...]
    adjacent_region_ids: tuple[int, ...]
    geometric_edge_tags: tuple[int, ...]

    def __post_init__(self) -> None:
        object.__setattr__(
            self,
            "loop_tag",
            _require_positive_tag(self.loop_tag, "topology cycle loop tag"),
        )
        nodes = tuple(int(value) for value in self.node_indices)
        regions = tuple(int(value) for value in self.adjacent_region_ids)
        edge_tags = tuple(int(value) for value in self.geometric_edge_tags)
        if len(nodes) < 3 or len(set(nodes)) != len(nodes):
            raise ValueError("topology cycle 必须包含至少 3 个互异节点")
        if len(regions) not in (1, 2) or any(value <= 0 for value in regions):
            raise ValueError("topology cycle 必须邻接一个或两个正 Region IDs")
        if tuple(sorted(set(regions))) != regions:
            raise ValueError("topology cycle adjacent_region_ids 必须唯一且升序")
        if len(edge_tags) != len(nodes):
            raise ValueError("topology cycle edge tags 数量必须等于节点数")
        object.__setattr__(self, "node_indices", nodes)
        object.__setattr__(self, "adjacent_region_ids", regions)
        object.__setattr__(self, "geometric_edge_tags", edge_tags)


@dataclass(frozen=True)
class M3DC1BuildResult:
    """Immutable topology-driven M3D-C1 target build result."""

    spec: M3DC1ModelSpec
    region_id_to_face_tag: Mapping[int, int]
    cycles: tuple[TopologyCycle, ...]
    mesh_edges: tuple[tuple[int, int], ...]
    vertex_classifications: tuple[ModelClassification, ...]
    edge_classifications: tuple[ModelClassification, ...]
    triangle_classifications: tuple[ModelClassification, ...]

    def __post_init__(self) -> None:
        if not isinstance(self.spec, M3DC1ModelSpec):
            raise TypeError("M3DC1BuildResult.spec 必须是 M3DC1ModelSpec")
        self.spec.validate()
        mapping = {
            int(region_id): int(face_tag)
            for region_id, face_tag in self.region_id_to_face_tag.items()
        }
        if (
            any(region_id <= 0 for region_id in mapping)
            or sorted(mapping.values()) != list(range(1, len(mapping) + 1))
        ):
            raise ValueError(
                "region_id_to_face_tag 必须将正 Region IDs 映射到连续 face tags"
            )
        object.__setattr__(
            self,
            "region_id_to_face_tag",
            MappingProxyType(mapping),
        )
        object.__setattr__(self, "cycles", tuple(self.cycles))
        object.__setattr__(self, "mesh_edges", tuple(self.mesh_edges))
        object.__setattr__(
            self,
            "vertex_classifications",
            tuple(self.vertex_classifications),
        )
        object.__setattr__(
            self,
            "edge_classifications",
            tuple(self.edge_classifications),
        )
        object.__setattr__(
            self,
            "triangle_classifications",
            tuple(self.triangle_classifications),
        )
        if len(self.mesh_edges) != len(self.edge_classifications):
            raise ValueError("mesh edge table 与 edge classifications 数量不一致")


def _validated_multi_mesh_arrays(
    mesh: Mesh2D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    triangles_float = np.asarray(mesh.triangles, dtype=np.float64)
    region_float = np.asarray(mesh.region_ids, dtype=np.float64)

    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] < 3:
        raise ValueError(f"Mesh2D.nodes 必须为 N×2，实际 shape={nodes.shape}")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("Mesh2D.nodes 含 NaN/Inf")
    if (
        triangles_float.ndim != 2
        or triangles_float.shape[1] != 3
        or triangles_float.shape[0] == 0
    ):
        raise ValueError(
            f"Mesh2D.triangles 必须为非空 T×3，实际 shape={triangles_float.shape}"
        )
    if (
        not np.all(np.isfinite(triangles_float))
        or not np.all(triangles_float == np.round(triangles_float))
    ):
        raise ValueError("Mesh2D.triangles 必须具有有限整数语义")
    triangles = triangles_float.astype(np.int64)
    if np.any(triangles < 0) or int(triangles.max()) >= nodes.shape[0]:
        raise ValueError("Mesh2D.triangles 必须使用合法全局 0-based 节点索引")
    if (
        region_float.ndim != 1
        or region_float.shape[0] != triangles.shape[0]
        or not np.all(np.isfinite(region_float))
        or not np.all(region_float == np.round(region_float))
        or np.any(region_float <= 0)
    ):
        raise ValueError(
            "Mesh2D.region_ids 必须是与 triangles 等长的正整数一维数组"
        )
    region_ids = region_float.astype(np.int64)
    if np.unique(region_ids).size < 2:
        raise ValueError("multi-face builder 至少需要两个实际 Region")

    canonical_triangles: set[tuple[int, int, int]] = set()
    for index, triangle in enumerate(triangles):
        values = tuple(int(value) for value in triangle)
        if len(set(values)) != 3:
            raise ValueError(f"triangle {index} 含重复 node")
        key = tuple(sorted(values))
        if key in canonical_triangles:
            raise ValueError(f"检测到重复 triangle：{key}")
        canonical_triangles.add(key)
        p0, p1, p2 = nodes[triangle]
        edge1 = p1 - p0
        edge2 = p2 - p0
        area2 = float(
            edge1[0] * edge2[1] - edge1[1] * edge2[0]
        )
        if not np.isfinite(area2) or abs(area2) <= 2.0e-12:
            raise ValueError(f"triangle {index} 面积退化或不大于 1e-12")

    return (
        np.array(nodes, dtype=np.float64, copy=True),
        np.array(triangles, dtype=np.int64, copy=True),
        np.array(region_ids, dtype=np.int64, copy=True),
    )


def _canonical_cycle(
    edges: set[tuple[int, int]],
    label: str,
) -> tuple[int, ...]:
    adjacency: dict[int, list[int]] = {}
    for first, second in edges:
        adjacency.setdefault(first, []).append(second)
        adjacency.setdefault(second, []).append(first)
    invalid = {
        node: len(neighbors)
        for node, neighbors in adjacency.items()
        if len(neighbors) != 2
    }
    if invalid:
        raise ValueError(
            f"{label} 不是闭合 2-regular cycle，node degrees={invalid}"
        )

    start = min(adjacency)

    def walk(first_neighbor: int) -> tuple[int, ...]:
        result = [start]
        previous = start
        current = first_neighbor
        while current != start:
            if current in result:
                raise ValueError(f"{label} 在闭合前重复访问 node {current}")
            result.append(current)
            neighbors = adjacency[current]
            next_node = (
                neighbors[1] if neighbors[0] == previous else neighbors[0]
            )
            previous, current = current, next_node
            if len(result) > len(adjacency):
                raise ValueError(f"{label} cycle 遍历越界")
        if len(result) != len(adjacency):
            raise ValueError(f"{label} 含多个或歧义闭合 components")
        return tuple(result)

    directions = tuple(walk(neighbor) for neighbor in sorted(adjacency[start]))
    return min(directions)


def _edge_components(
    edges: set[tuple[int, int]],
) -> tuple[set[tuple[int, int]], ...]:
    remaining = set(edges)
    components: list[set[tuple[int, int]]] = []
    while remaining:
        first_edge = min(remaining)
        nodes = set(first_edge)
        component: set[tuple[int, int]] = set()
        changed = True
        while changed:
            changed = False
            for edge in tuple(remaining):
                if nodes.intersection(edge):
                    remaining.remove(edge)
                    component.add(edge)
                    nodes.update(edge)
                    changed = True
        components.append(component)
    return tuple(components)


def build_multi_face_model(
    mesh: Mesh2D,
) -> M3DC1BuildResult:
    """Build deterministic multi-face topology and classifications."""
    if not isinstance(mesh, Mesh2D):
        raise TypeError(f"mesh 必须是 Mesh2D，实际={type(mesh).__name__}")

    nodes, triangles, region_ids = _validated_multi_mesh_arrays(mesh)
    sorted_region_ids = tuple(
        int(value) for value in sorted(np.unique(region_ids).tolist())
    )
    region_to_face = {
        region_id: index + 1
        for index, region_id in enumerate(sorted_region_ids)
    }

    incidence: dict[tuple[int, int], list[int]] = {}
    triangle_neighbors: list[set[int]] = [
        set() for _ in range(triangles.shape[0])
    ]
    vertex_regions: list[set[int]] = [
        set() for _ in range(nodes.shape[0])
    ]
    for triangle_index, triangle in enumerate(triangles):
        region_id = int(region_ids[triangle_index])
        for node_index in triangle:
            vertex_regions[int(node_index)].add(region_id)
        for first, second in (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        ):
            key = (first, second) if first < second else (second, first)
            incidence.setdefault(key, []).append(triangle_index)

    for edge, adjacent in incidence.items():
        if len(adjacent) not in (1, 2):
            raise ValueError(
                f"检测到 non-manifold mesh edge {edge}，incidence={len(adjacent)}"
            )
        if len(adjacent) == 2:
            first, second = adjacent
            triangle_neighbors[first].add(second)
            triangle_neighbors[second].add(first)

    for region_id in sorted_region_ids:
        members = {
            index
            for index, value in enumerate(region_ids)
            if int(value) == region_id
        }
        reached = {min(members)}
        pending = list(reached)
        while pending:
            current = pending.pop()
            for neighbor in triangle_neighbors[current]:
                if neighbor in members and neighbor not in reached:
                    reached.add(neighbor)
                    pending.append(neighbor)
        if reached != members:
            raise ValueError(
                f"Region ID {region_id} 的 triangles 包含多个不连通 components"
            )

    cycle_groups: list[tuple[tuple[int, ...], tuple[int, ...]]] = []
    outer_edges = {
        edge for edge, adjacent in incidence.items() if len(adjacent) == 1
    }
    for component in _edge_components(outer_edges):
        cycle = _canonical_cycle(component, "outer boundary")
        adjacent_regions = {
            int(region_ids[incidence[edge][0]]) for edge in component
        }
        if len(adjacent_regions) != 1:
            raise ValueError("一个 outer-boundary cycle 邻接多个 Regions")
        cycle_groups.append((cycle, tuple(sorted(adjacent_regions))))

    interface_by_pair: dict[
        tuple[int, int], set[tuple[int, int]]
    ] = {}
    for edge, adjacent in incidence.items():
        if len(adjacent) != 2:
            continue
        pair = tuple(
            sorted(
                {
                    int(region_ids[adjacent[0]]),
                    int(region_ids[adjacent[1]]),
                }
            )
        )
        if len(pair) == 2:
            interface_by_pair.setdefault(pair, set()).add(edge)

    for pair, edges in sorted(interface_by_pair.items()):
        for component in _edge_components(edges):
            cycle_groups.append(
                (_canonical_cycle(component, f"interface {pair}"), pair)
            )

    cycle_groups.sort(key=lambda item: item[0])
    used_cycle_nodes: set[int] = set()
    for cycle, _regions in cycle_groups:
        overlap = used_cycle_nodes.intersection(cycle)
        if overlap:
            raise ValueError(
                "global mesh node 同时属于多个 topology cycles，"
                f"无法无歧义表达：{sorted(overlap)}"
            )
        used_cycle_nodes.update(cycle)

    geometric_vertex_tag = {
        node_index: tag
        for tag, node_index in enumerate(sorted(used_cycle_nodes), start=1)
    }
    geometric_vertices = tuple(
        GeometricVertex(
            tag=tag,
            r=float(nodes[node_index, 0]),
            z=float(nodes[node_index, 1]),
        )
        for node_index, tag in sorted(
            geometric_vertex_tag.items(),
            key=lambda item: item[1],
        )
    )

    geometric_edges: list[GeometricEdge] = []
    loops: list[GeometricLoop] = []
    cycles: list[TopologyCycle] = []
    mesh_edge_to_geometric_tag: dict[tuple[int, int], int] = {}
    face_loop_tags: dict[int, list[int]] = {
        region_id: [] for region_id in sorted_region_ids
    }

    for loop_tag, (cycle, adjacent_regions) in enumerate(
        cycle_groups,
        start=1,
    ):
        edge_tags: list[int] = []
        for index, first in enumerate(cycle):
            second = cycle[(index + 1) % len(cycle)]
            mesh_edge = (
                (first, second) if first < second else (second, first)
            )
            if mesh_edge in mesh_edge_to_geometric_tag:
                raise ValueError(
                    f"cycle mesh edge 被重复使用：{mesh_edge}"
                )
            edge_tag = len(geometric_edges) + 1
            mesh_edge_to_geometric_tag[mesh_edge] = edge_tag
            edge_tags.append(edge_tag)
            geometric_edges.append(
                GeometricEdge(
                    tag=edge_tag,
                    begin_vertex_tag=geometric_vertex_tag[first],
                    end_vertex_tag=geometric_vertex_tag[second],
                )
            )
        loops.append(GeometricLoop(loop_tag, tuple(edge_tags)))
        cycles.append(
            TopologyCycle(
                loop_tag=loop_tag,
                node_indices=cycle,
                adjacent_region_ids=adjacent_regions,
                geometric_edge_tags=tuple(edge_tags),
            )
        )
        for region_id in adjacent_regions:
            face_loop_tags[region_id].append(loop_tag)

    faces = tuple(
        GeometricFace(
            tag=region_to_face[region_id],
            loop_tags=tuple(sorted(face_loop_tags[region_id])),
        )
        for region_id in sorted_region_ids
    )
    if any(not face.loop_tags for face in faces):
        raise ValueError("至少一个 Region face 没有邻接 topology cycle")

    spec = M3DC1ModelSpec(
        vertices=geometric_vertices,
        edges=tuple(geometric_edges),
        loops=tuple(loops),
        faces=faces,
        separatrix_loop=-1,
        inner_wall_loop=-1,
        outer_wall_loop=-1,
        vacuum_loop=-1,
    )

    mesh_edges = tuple(sorted(incidence))
    edge_classes: list[ModelClassification] = []
    for edge in mesh_edges:
        geometric_tag = mesh_edge_to_geometric_tag.get(edge)
        if geometric_tag is not None:
            edge_classes.append(ModelClassification(1, geometric_tag))
            continue
        adjacent_regions = {
            int(region_ids[index]) for index in incidence[edge]
        }
        if len(adjacent_regions) != 1:
            raise ValueError(
                f"interface edge {edge} 未被 topology cycle 分类"
            )
        region_id = next(iter(adjacent_regions))
        edge_classes.append(
            ModelClassification(2, region_to_face[region_id])
        )

    vertex_classes: list[ModelClassification] = []
    for node_index, regions in enumerate(vertex_regions):
        if node_index in geometric_vertex_tag:
            vertex_classes.append(
                ModelClassification(
                    0,
                    geometric_vertex_tag[node_index],
                )
            )
        elif len(regions) == 1:
            region_id = next(iter(regions))
            vertex_classes.append(
                ModelClassification(2, region_to_face[region_id])
            )
        elif regions:
            raise ValueError(
                f"非 cycle node {node_index} 邻接多个 Regions：{sorted(regions)}"
            )
        else:
            raise ValueError(f"mesh node {node_index} 未被任何 triangle 使用")

    triangle_classes = tuple(
        ModelClassification(2, region_to_face[int(region_id)])
        for region_id in region_ids
    )
    return M3DC1BuildResult(
        spec=spec,
        region_id_to_face_tag=region_to_face,
        cycles=tuple(cycles),
        mesh_edges=mesh_edges,
        vertex_classifications=tuple(vertex_classes),
        edge_classifications=tuple(edge_classes),
        triangle_classifications=triangle_classes,
    )


def _validate_single_region_ids(region_ids_in: Any) -> None:
    """Reject Mesh2D inputs outside the single-region P0 scope."""
    region_float = np.asarray(region_ids_in, dtype=np.float64)
    if region_float.ndim != 1:
        raise ValueError(
            f"Mesh2D.region_ids 必须是一维数组，实际 ndim={region_float.ndim}"
        )
    if region_float.size == 0:
        raise ValueError("Mesh2D.region_ids 为空，无法确认 single-region 拓扑")
    if not np.all(np.isfinite(region_float)):
        raise ValueError("Mesh2D.region_ids 含非有限值（NaN/Inf）")
    if not np.all(region_float == np.round(region_float)):
        raise ValueError("Mesh2D.region_ids 含非整数值")

    region_ids = region_float.astype(np.int64)
    if np.any(region_ids <= 0):
        raise ValueError(
            "Mesh2D.region_ids 必须是正整数区域编号，"
            f"实际最小值={int(region_ids.min())}"
        )

    unique_regions = np.unique(region_ids)
    if unique_regions.size != 1:
        raise ValueError(
            "当前 model 只支持 single-region Mesh2D，"
            f"实际 region IDs={unique_regions.tolist()}"
        )


def build_single_face_model_spec(mesh: Mesh2D) -> M3DC1ModelSpec:
    """Build a single-face M3DC1ModelSpec from Mesh2D.outer_boundary.

    Geometry is taken exclusively from the ordered, implicitly closed
    Mesh2D.outer_boundary. No geometry is inferred from triangles, topology
    boundary nodes, idxevert, boundary markers, or Steiner points.

    The input Mesh2D and all of its NumPy arrays remain unchanged.
    """
    if not isinstance(mesh, Mesh2D):
        raise TypeError(
            f"mesh 必须是 Mesh2D，实际={type(mesh).__name__}"
        )

    _validate_single_region_ids(mesh.region_ids)

    if mesh.outer_boundary is None:
        raise ValueError(
            "Mesh2D.outer_boundary 缺失（None）："
            "analytic model 必须使用有序原始外边界"
        )

    boundary = _mesh_validate_outer_boundary(mesh.outer_boundary)
    point_count = int(boundary.shape[0])

    vertices = tuple(
        GeometricVertex(
            tag=index + 1,
            r=float(boundary[index, 0]),
            z=float(boundary[index, 1]),
            y=0.0,
        )
        for index in range(point_count)
    )

    linear_curve = M3DC1Curve(CurveType.LIN)
    edges = tuple(
        GeometricEdge(
            tag=index + 1,
            begin_vertex_tag=index + 1,
            end_vertex_tag=((index + 1) % point_count) + 1,
            curve=linear_curve,
        )
        for index in range(point_count)
    )

    loop = GeometricLoop(
        tag=1,
        edge_tags=tuple(edge.tag for edge in edges),
    )
    face = GeometricFace(tag=1, loop_tags=(loop.tag,))

    return M3DC1ModelSpec(
        vertices=vertices,
        edges=edges,
        loops=(loop,),
        faces=(face,),
        separatrix_loop=-1,
        inner_wall_loop=loop.tag,
        outer_wall_loop=-1,
        vacuum_loop=-1,
    )
