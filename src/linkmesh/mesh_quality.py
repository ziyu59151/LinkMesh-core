"""Read-only numerical quality reporting for a final :class:`Mesh2D`.

The module computes deterministic, solver-independent mesh statistics from
Python-owned arrays. It does not modify meshes, call Triangle, print output,
or define acceptance thresholds for diagnostic quality values.
"""

from __future__ import annotations

import math
from dataclasses import dataclass
from typing import TYPE_CHECKING

import numpy as np

from linkmesh.mesh import Mesh2D

if TYPE_CHECKING:
    from linkmesh.mesher.triangle_refinement import RefinementDiagnostics


_TRIANGLE_AREA_MINIMUM = 1.0e-12


@dataclass(frozen=True)
class MeshQualityReport:
    """Immutable numerical quality summary for one final Mesh2D."""

    node_count: int
    triangle_count: int
    region_count: int
    minimum_angle: float
    minimum_angle_p01: float
    minimum_angle_p05: float
    maximum_angle: float
    minimum_shape_quality: float
    shape_quality_p01: float
    shape_quality_p05: float
    median_shape_quality: float
    minimum_equivalent_size: float
    median_equivalent_size: float
    maximum_equivalent_size: float
    maximum_adjacent_size_ratio: float | None
    adjacent_size_ratio_p95: float | None
    adjacent_size_ratio_p99: float | None
    degenerate_triangle_count: int
    duplicate_triangle_count: int
    non_manifold_edge_count: int
    refinement: RefinementDiagnostics | None


def _percentile(values: np.ndarray, percentile: float) -> float:
    """Return one float64 NumPy percentile using a single project definition."""
    return float(np.percentile(values, percentile, method="linear"))


def _validated_arrays(
    mesh: Mesh2D,
) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    nodes = np.asarray(mesh.nodes, dtype=np.float64)
    triangles_raw = np.asarray(mesh.triangles)
    region_ids_raw = np.asarray(mesh.region_ids)

    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] == 0:
        raise ValueError(
            f"mesh quality nodes 必须为非空 (N,2)，实际 shape={nodes.shape}"
        )
    if not np.all(np.isfinite(nodes)):
        raise ValueError("mesh quality nodes 含 NaN/Inf")

    if (
        triangles_raw.ndim != 2
        or triangles_raw.shape[1] != 3
        or triangles_raw.shape[0] == 0
    ):
        raise ValueError(
            "mesh quality triangles 必须为非空 (T,3)，"
            f"实际 shape={triangles_raw.shape}"
        )
    triangles_float = np.asarray(triangles_raw, dtype=np.float64)
    if (
        not np.all(np.isfinite(triangles_float))
        or not np.all(triangles_float == np.round(triangles_float))
    ):
        raise ValueError("mesh quality triangles 不具备有限整数索引语义")
    if np.any(triangles_float < np.iinfo(np.int64).min) or np.any(
        triangles_float > np.iinfo(np.int64).max
    ):
        raise ValueError("mesh quality triangles 超出 int64 范围")
    triangles = np.asarray(triangles_float, dtype=np.int64)
    if triangles.min() < 0 or triangles.max() >= nodes.shape[0]:
        raise ValueError("mesh quality triangles 含越界的 0-based 节点索引")

    region_ids_float = np.asarray(region_ids_raw, dtype=np.float64).reshape(-1)
    if region_ids_float.shape != (triangles.shape[0],):
        raise ValueError("mesh quality region_ids 必须与 triangles 一一对应")
    if (
        not np.all(np.isfinite(region_ids_float))
        or not np.all(region_ids_float == np.round(region_ids_float))
        or np.any(region_ids_float <= 0)
    ):
        raise ValueError("mesh quality region_ids 必须具有正整数语义")
    region_ids = np.asarray(region_ids_float, dtype=np.int64)

    return nodes, triangles, region_ids


def _shape_quality_values(
    areas: np.ndarray,
    side_squares: np.ndarray,
) -> np.ndarray:
    """Return normalized float64 Q values from the project's unique formula."""
    quality = np.asarray(
        4.0 * math.sqrt(3.0) * areas / np.sum(side_squares, axis=1),
        dtype=np.float64,
    )
    if not np.all(np.isfinite(quality)):
        raise ValueError("mesh quality Q 含 NaN/Inf")

    tolerance = 64.0 * np.finfo(np.float64).eps
    if np.any(quality < -tolerance) or np.any(quality > 1.0 + tolerance):
        raise ValueError("mesh quality Q 明显超出理论范围 [0,1]")
    return np.clip(quality, 0.0, 1.0)


def calculate_triangle_shape_quality(mesh: Mesh2D) -> np.ndarray:
    """Return independently owned per-triangle Q values in triangle order."""
    nodes, triangles, _region_ids = _validated_arrays(mesh)
    first = nodes[triangles[:, 0]]
    second = nodes[triangles[:, 1]]
    third = nodes[triangles[:, 2]]

    side_lengths = np.column_stack(
        (
            np.linalg.norm(second - third, axis=1),
            np.linalg.norm(first - third, axis=1),
            np.linalg.norm(first - second, axis=1),
        )
    )
    doubled_area = np.abs(
        (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
        - (third[:, 0] - first[:, 0]) * (second[:, 1] - first[:, 1])
    )
    areas = 0.5 * doubled_area
    if (
        not np.all(np.isfinite(side_lengths))
        or not np.all(np.isfinite(areas))
    ):
        raise ValueError("mesh quality triangle 几何含 NaN/Inf")
    if np.any(areas <= _TRIANGLE_AREA_MINIMUM) or np.any(
        side_lengths <= 0.0
    ):
        raise ValueError("mesh quality Q 无法用于退化 triangle")

    values = _shape_quality_values(
        areas,
        side_lengths * side_lengths,
    )
    if values.shape != (mesh.n_triangles,):
        raise ValueError("mesh quality Q 必须与 triangles 一一对应")
    return np.array(values, dtype=np.float64, copy=True)


def _edge_statistics(
    triangles: np.ndarray,
    equivalent_sizes: np.ndarray,
) -> tuple[np.ndarray, int]:
    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    for triangle_index, triangle in enumerate(triangles):
        for raw_first, raw_second in (
            (triangle[0], triangle[1]),
            (triangle[1], triangle[2]),
            (triangle[2], triangle[0]),
        ):
            first, second = int(raw_first), int(raw_second)
            edge = (first, second) if first < second else (second, first)
            edge_to_triangles.setdefault(edge, []).append(triangle_index)

    ratios: list[float] = []
    non_manifold_count = 0
    for incident_triangles in edge_to_triangles.values():
        if len(incident_triangles) > 2:
            non_manifold_count += 1
        if len(incident_triangles) != 2:
            continue
        first_size = float(equivalent_sizes[incident_triangles[0]])
        second_size = float(equivalent_sizes[incident_triangles[1]])
        smaller = min(first_size, second_size)
        if smaller <= 0.0:
            continue
        ratios.append(max(first_size, second_size) / smaller)

    return np.asarray(ratios, dtype=np.float64), non_manifold_count


def calculate_mesh_quality(
    mesh: Mesh2D,
    refinement: RefinementDiagnostics | None = None,
) -> MeshQualityReport:
    """Compute a deterministic immutable quality report without modifying mesh."""
    nodes, triangles, region_ids = _validated_arrays(mesh)

    first = nodes[triangles[:, 0]]
    second = nodes[triangles[:, 1]]
    third = nodes[triangles[:, 2]]

    side_opposite_first = np.linalg.norm(second - third, axis=1)
    side_opposite_second = np.linalg.norm(first - third, axis=1)
    side_opposite_third = np.linalg.norm(first - second, axis=1)
    side_lengths = np.column_stack(
        (
            side_opposite_first,
            side_opposite_second,
            side_opposite_third,
        )
    )
    if not np.all(np.isfinite(side_lengths)):
        raise ValueError("mesh quality triangle 边长含 NaN/Inf")

    doubled_area = np.abs(
        (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
        - (third[:, 0] - first[:, 0]) * (second[:, 1] - first[:, 1])
    )
    areas = 0.5 * doubled_area
    if not np.all(np.isfinite(areas)):
        raise ValueError("mesh quality triangle 面积含 NaN/Inf")

    degenerate = (
        (areas <= _TRIANGLE_AREA_MINIMUM)
        | np.any(side_lengths <= 0.0, axis=1)
    )
    valid = ~degenerate
    if not np.any(valid):
        raise ValueError("mesh quality 无可安全计算质量指标的非退化 triangle")

    valid_sides = side_lengths[valid]
    side_squares = valid_sides * valid_sides
    cosines = np.column_stack(
        (
            (
                side_squares[:, 1]
                + side_squares[:, 2]
                - side_squares[:, 0]
            )
            / (2.0 * valid_sides[:, 1] * valid_sides[:, 2]),
            (
                side_squares[:, 0]
                + side_squares[:, 2]
                - side_squares[:, 1]
            )
            / (2.0 * valid_sides[:, 0] * valid_sides[:, 2]),
            (
                side_squares[:, 0]
                + side_squares[:, 1]
                - side_squares[:, 2]
            )
            / (2.0 * valid_sides[:, 0] * valid_sides[:, 1]),
        )
    )
    angles = np.degrees(np.arccos(np.clip(cosines, -1.0, 1.0)))
    if not np.all(np.isfinite(angles)):
        raise ValueError("mesh quality triangle 角度含 NaN/Inf")
    minimum_angles = np.min(angles, axis=1)

    shape_quality = _shape_quality_values(
        areas[valid],
        side_squares,
    )
    equivalent_sizes = np.sqrt(2.0 * areas)
    valid_equivalent_sizes = equivalent_sizes[valid]
    if (
        not np.all(np.isfinite(shape_quality))
        or not np.all(np.isfinite(valid_equivalent_sizes))
    ):
        raise ValueError("mesh quality Q 或 equivalent size 含 NaN/Inf")

    canonical_triangles = np.sort(triangles, axis=1)
    duplicate_count = int(
        triangles.shape[0]
        - np.unique(canonical_triangles, axis=0).shape[0]
    )
    adjacent_ratios, non_manifold_count = _edge_statistics(
        triangles,
        equivalent_sizes,
    )
    if adjacent_ratios.size:
        maximum_adjacent_ratio: float | None = float(
            np.max(adjacent_ratios)
        )
        adjacent_p95: float | None = _percentile(adjacent_ratios, 95.0)
        adjacent_p99: float | None = _percentile(adjacent_ratios, 99.0)
    else:
        maximum_adjacent_ratio = None
        adjacent_p95 = None
        adjacent_p99 = None

    return MeshQualityReport(
        node_count=int(nodes.shape[0]),
        triangle_count=int(triangles.shape[0]),
        region_count=int(np.unique(region_ids).size),
        minimum_angle=float(np.min(minimum_angles)),
        minimum_angle_p01=_percentile(minimum_angles, 1.0),
        minimum_angle_p05=_percentile(minimum_angles, 5.0),
        maximum_angle=float(np.max(angles)),
        minimum_shape_quality=float(np.min(shape_quality)),
        shape_quality_p01=_percentile(shape_quality, 1.0),
        shape_quality_p05=_percentile(shape_quality, 5.0),
        median_shape_quality=_percentile(shape_quality, 50.0),
        minimum_equivalent_size=float(np.min(valid_equivalent_sizes)),
        median_equivalent_size=_percentile(valid_equivalent_sizes, 50.0),
        maximum_equivalent_size=float(np.max(valid_equivalent_sizes)),
        maximum_adjacent_size_ratio=maximum_adjacent_ratio,
        adjacent_size_ratio_p95=adjacent_p95,
        adjacent_size_ratio_p99=adjacent_p99,
        degenerate_triangle_count=int(np.count_nonzero(degenerate)),
        duplicate_triangle_count=duplicate_count,
        non_manifold_edge_count=non_manifold_count,
        refinement=refinement,
    )
