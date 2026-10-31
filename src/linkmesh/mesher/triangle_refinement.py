"""LinkMesh-side Triangle transition refinement for multi-region meshes.

This module is the only owner of transition topology extraction, distance
profiles, per-triangle area constraints, the OFT 26.6 ``rpqazQ`` native call,
native output cleanup, bounded iteration, and final Mesh2D validation.
"""

from __future__ import annotations

import ctypes
import importlib
import math
from dataclasses import dataclass
from typing import Any, Mapping

import numpy as np

from linkmesh.mesh import Mesh2D


_MAX_REFINEMENT_ROUNDS = 3
_AREA_EPSILON_FACTOR = 64.0
_TRIANGLE_AREA_MINIMUM = 1.0e-12
_DISTANCE_CHUNK_SIZE = 1024
_OPTION_STRING = b"rpqazQ"


@dataclass(frozen=True)
class InterfaceConstraint:
    """One effective-dx transition source grouped by its two Region IDs."""

    coarse_region_id: int
    fine_region_id: int
    fine_dx: float
    coarse_dx: float
    segments: np.ndarray


@dataclass(frozen=True)
class EdgeTopology:
    """Canonical global edges, incidences, PSLG, and transition interfaces."""

    edges: np.ndarray
    incidences: tuple[tuple[int, ...], ...]
    pslg_segments: np.ndarray
    interfaces: tuple[InterfaceConstraint, ...]


@dataclass(frozen=True)
class LocalRefinementSource:
    """One validated physical R-Z point size-field source."""

    r: float
    z: float
    dx: float
    width: float


@dataclass(frozen=True)
class LocatedLocalRefinementSource:
    """One point source with its initial-mesh Region assignment."""

    r: float
    z: float
    dx: float
    width: float
    region_id: int
    baseline_dx: float


@dataclass(frozen=True)
class LocalRefinementDiagnostics:
    """Immutable final diagnostics for one R-Z point source."""

    r: float
    z: float
    dx: float
    width: float
    region_id: int
    baseline_dx: float
    affected_triangle_count: int
    maximum_area_excess: float


@dataclass(frozen=True)
class InterfaceRefinementDiagnostics:
    """Immutable final diagnostics for one different-dx Region interface."""

    region_ids: tuple[int, int]
    fine_region_id: int
    coarse_region_id: int
    fine_dx: float
    coarse_dx: float
    transition_width: float
    affected_coarse_triangle_count: int
    maximum_area_excess: float


@dataclass(frozen=True)
class RefinementDiagnostics:
    """Immutable bounded-refinement diagnostics for verbose reporting."""

    enabled: bool
    transition_mode: str
    transition_width: float | None
    transition_gradient: float | None
    iteration_count: int
    initial_node_count: int
    initial_triangle_count: int
    final_node_count: int
    final_triangle_count: int
    transition_interface_count: int
    affected_coarse_triangle_count: int
    final_violating_triangle_count: int | None
    final_maximum_area_excess: float | None
    interfaces: tuple[InterfaceRefinementDiagnostics, ...]
    local_refinement_enabled: bool
    local_sources: tuple[LocalRefinementDiagnostics, ...]


@dataclass(frozen=True)
class RefinementResult:
    """The unique final mesh and its immutable transition diagnostics."""

    mesh: Mesh2D
    diagnostics: RefinementDiagnostics


def _owned_mesh_copy(mesh: Mesh2D) -> Mesh2D:
    """Return an equivalent Mesh2D whose arrays are all independently owned."""
    outer_boundary = (
        None
        if mesh.outer_boundary is None
        else np.array(mesh.outer_boundary, dtype=np.float64, copy=True)
    )
    return Mesh2D(
        nodes=np.array(mesh.nodes, dtype=np.float64, order="C", copy=True),
        triangles=np.array(mesh.triangles, dtype=np.int32, order="C", copy=True),
        region_ids=np.array(mesh.region_ids, dtype=np.int32, order="C", copy=True),
        boundary_segments=np.empty((0, 2), dtype=np.int32),
        boundary_markers=np.empty((0,), dtype=np.int32),
        node_markers=np.empty((0,), dtype=np.int32),
        outer_boundary=outer_boundary,
    )


def _validate_effective_dx(
    region_ids: np.ndarray,
    dx_by_region_id: Mapping[int, float],
) -> dict[int, float]:
    expected_ids = {int(value) for value in region_ids.tolist()}
    normalized: dict[int, float] = {}

    for raw_region_id, raw_dx in dx_by_region_id.items():
        if isinstance(raw_region_id, bool):
            raise ValueError("effective dx mapping 的 Region ID 必须是正整数")
        try:
            numeric_region_id = float(raw_region_id)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"effective dx mapping 含非法 Region ID：{raw_region_id!r}"
            ) from exc
        if (
            not math.isfinite(numeric_region_id)
            or numeric_region_id != round(numeric_region_id)
            or numeric_region_id <= 0
        ):
            raise ValueError(
                f"effective dx mapping 含非法 Region ID：{raw_region_id!r}"
            )
        region_id = int(numeric_region_id)

        if isinstance(raw_dx, bool):
            raise ValueError(f"Region ID {region_id} 的 dx 必须是有限正数")
        try:
            dx = float(raw_dx)
        except (TypeError, ValueError) as exc:
            raise ValueError(
                f"Region ID {region_id} 的 dx 必须是有限正数"
            ) from exc
        if not (math.isfinite(dx) and dx > 0.0):
            raise ValueError(f"Region ID {region_id} 的 dx 必须是有限正数")
        normalized[region_id] = dx

    if set(normalized) != expected_ids:
        raise ValueError(
            "effective dx mapping 必须精确覆盖 Mesh2D Region ID 集合："
            f"期望={sorted(expected_ids)}，实际={sorted(normalized)}"
        )
    return normalized


def _validate_mesh_arrays(
    mesh: Mesh2D,
    *,
    expected_region_ids: set[int] | None = None,
) -> Mesh2D:
    """Validate refinement input/output and return an independently owned mesh."""
    nodes = np.array(mesh.nodes, dtype=np.float64, order="C", copy=True)
    triangles_raw = np.asarray(mesh.triangles)
    region_ids_raw = np.asarray(mesh.region_ids)

    if nodes.ndim != 2 or nodes.shape[1] != 2 or nodes.shape[0] == 0:
        raise ValueError(
            f"refinement nodes 必须为非空 (N,2)，实际 shape={nodes.shape}"
        )
    if not np.all(np.isfinite(nodes)):
        raise ValueError("refinement nodes 含 NaN/Inf")

    if (
        triangles_raw.ndim != 2
        or triangles_raw.shape[1] != 3
        or triangles_raw.shape[0] == 0
    ):
        raise ValueError(
            "refinement triangles 必须为非空 (T,3)，"
            f"实际 shape={triangles_raw.shape}"
        )
    triangles_float = np.asarray(triangles_raw, dtype=np.float64)
    if (
        not np.all(np.isfinite(triangles_float))
        or not np.all(triangles_float == np.round(triangles_float))
    ):
        raise ValueError("refinement triangles 不具备有限整数索引语义")
    if np.any(triangles_float < np.iinfo(np.int32).min) or np.any(
        triangles_float > np.iinfo(np.int32).max
    ):
        raise ValueError("refinement triangles 超出 int32 范围")
    triangles = np.array(
        triangles_float,
        dtype=np.int32,
        order="C",
        copy=True,
    )
    if triangles.min() < 0 or triangles.max() >= nodes.shape[0]:
        raise ValueError("refinement triangles 含越界的 0-based 节点索引")
    if np.any(
        (triangles[:, 0] == triangles[:, 1])
        | (triangles[:, 1] == triangles[:, 2])
        | (triangles[:, 0] == triangles[:, 2])
    ):
        raise ValueError("refinement triangles 含重复顶点")

    canonical_triangles = np.sort(triangles, axis=1)
    if np.unique(canonical_triangles, axis=0).shape[0] != triangles.shape[0]:
        raise ValueError("refinement mesh 含重复 triangle")

    first = nodes[triangles[:, 0]]
    second = nodes[triangles[:, 1]]
    third = nodes[triangles[:, 2]]
    doubled_area = (
        (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
        - (third[:, 0] - first[:, 0]) * (second[:, 1] - first[:, 1])
    )
    areas = 0.5 * np.abs(doubled_area)
    if not np.all(np.isfinite(areas)) or np.any(
        areas <= _TRIANGLE_AREA_MINIMUM
    ):
        raise ValueError("refinement mesh 含面积退化 triangle")

    region_float = np.asarray(region_ids_raw, dtype=np.float64).reshape(-1)
    if region_float.shape != (triangles.shape[0],):
        raise ValueError(
            "refinement region_ids 必须与 triangles 一一对应，"
            f"实际 shape={region_float.shape}"
        )
    if (
        not np.all(np.isfinite(region_float))
        or not np.all(region_float == np.round(region_float))
        or np.any(region_float <= 0)
        or np.any(region_float < np.iinfo(np.int32).min)
        or np.any(region_float > np.iinfo(np.int32).max)
    ):
        raise ValueError("refinement Region attributes 不具备正 int32 整数语义")
    region_ids = np.array(
        region_float,
        dtype=np.int32,
        order="C",
        copy=True,
    )
    actual_region_ids = {int(value) for value in region_ids.tolist()}
    if expected_region_ids is not None and actual_region_ids != expected_region_ids:
        raise ValueError(
            "refinement 输出 Region ID 集合发生变化："
            f"期望={sorted(expected_region_ids)}，实际={sorted(actual_region_ids)}"
        )
    if any(not np.any(region_ids == region_id) for region_id in actual_region_ids):
        raise ValueError("refinement 输出至少一个 Region 为空")

    validated = Mesh2D(
        nodes=nodes,
        triangles=triangles,
        region_ids=region_ids,
        boundary_segments=np.empty((0, 2), dtype=np.int32),
        boundary_markers=np.empty((0,), dtype=np.int32),
        node_markers=np.empty((0,), dtype=np.int32),
        outer_boundary=(
            None
            if mesh.outer_boundary is None
            else np.array(
                mesh.outer_boundary,
                dtype=np.float64,
                order="C",
                copy=True,
            )
        ),
    )
    _extract_edge_topology(validated, None)
    return validated


def _extract_edge_topology(
    mesh: Mesh2D,
    dx_by_region_id: Mapping[int, float] | None,
) -> EdgeTopology:
    """Extract canonical edge incidences, PSLG, and effective-dx interfaces."""
    edge_to_triangles: dict[tuple[int, int], list[int]] = {}
    for triangle_index, triangle in enumerate(mesh.triangles):
        local_edges = (
            (int(triangle[0]), int(triangle[1])),
            (int(triangle[1]), int(triangle[2])),
            (int(triangle[2]), int(triangle[0])),
        )
        for first, second in local_edges:
            edge = (first, second) if first < second else (second, first)
            edge_to_triangles.setdefault(edge, []).append(triangle_index)

    ordered_edges = tuple(sorted(edge_to_triangles))
    incidences = tuple(
        tuple(edge_to_triangles[edge])
        for edge in ordered_edges
    )
    for edge, incident in zip(ordered_edges, incidences):
        if len(incident) > 2:
            raise ValueError(
                f"refinement topology non-manifold：edge={edge}, "
                f"incidence={len(incident)}"
            )
        if len(incident) == 0:
            raise ValueError(f"refinement topology edge 无 incidence：{edge}")

    pslg_edges: list[tuple[int, int]] = []
    grouped_interfaces: dict[
        tuple[int, int],
        list[tuple[int, int]],
    ] = {}

    for edge, incident in zip(ordered_edges, incidences):
        if len(incident) == 1:
            pslg_edges.append(edge)
            continue

        first_triangle, second_triangle = incident
        first_region = int(mesh.region_ids[first_triangle])
        second_region = int(mesh.region_ids[second_triangle])
        if first_region == second_region:
            continue

        pslg_edges.append(edge)
        if dx_by_region_id is None:
            continue

        first_dx = dx_by_region_id[first_region]
        second_dx = dx_by_region_id[second_region]
        if first_dx == second_dx:
            continue
        if first_dx > second_dx:
            coarse_region, fine_region = first_region, second_region
        else:
            coarse_region, fine_region = second_region, first_region
        grouped_interfaces.setdefault(
            (coarse_region, fine_region),
            [],
        ).append(edge)

    interfaces = tuple(
        InterfaceConstraint(
            coarse_region_id=coarse_region,
            fine_region_id=fine_region,
            fine_dx=float(dx_by_region_id[fine_region]),
            coarse_dx=float(dx_by_region_id[coarse_region]),
            segments=np.ascontiguousarray(
                grouped_interfaces[(coarse_region, fine_region)],
                dtype=np.int32,
            ),
        )
        for coarse_region, fine_region in sorted(grouped_interfaces)
    )

    return EdgeTopology(
        edges=np.ascontiguousarray(ordered_edges, dtype=np.int32),
        incidences=incidences,
        pslg_segments=np.ascontiguousarray(pslg_edges, dtype=np.int32),
        interfaces=interfaces,
    )


def _normalize_local_sources(
    sources: Any,
) -> tuple[LocalRefinementSource, ...]:
    normalized: list[LocalRefinementSource] = []
    for source in sources:
        try:
            r = float(source.r)
            z = float(source.z)
            dx = float(source.dx)
            width = float(source.width)
        except (AttributeError, TypeError, ValueError) as exc:
            raise ValueError(
                "local refinement source 必须提供 r、z、dx、width"
            ) from exc
        if not (math.isfinite(r) and math.isfinite(z)):
            raise ValueError("local refinement R/Z 必须有限")
        if not (math.isfinite(dx) and dx > 0.0):
            raise ValueError("local refinement DX 必须是有限正数")
        if not (math.isfinite(width) and width > 0.0):
            raise ValueError("local refinement W 必须是有限正数")
        normalized.append(
            LocalRefinementSource(r=r, z=z, dx=dx, width=width)
        )
    return tuple(normalized)


def _locate_local_sources(
    mesh: Mesh2D,
    effective_dx: Mapping[int, float],
    sources: tuple[LocalRefinementSource, ...],
) -> tuple[LocatedLocalRefinementSource, ...]:
    if not sources:
        return ()

    points = np.array(
        [[source.r, source.z] for source in sources],
        dtype=np.float64,
    )
    triangle_points = mesh.nodes[mesh.triangles]
    first = triangle_points[:, 0]
    second = triangle_points[:, 1]
    third = triangle_points[:, 2]
    denominator = (
        (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
        - (second[:, 1] - first[:, 1]) * (third[:, 0] - first[:, 0])
    )

    r_span = float(np.ptp(mesh.nodes[:, 0]))
    z_span = float(np.ptp(mesh.nodes[:, 1]))
    absolute_magnitude = float(np.max(np.abs(mesh.nodes)))
    length_scale = max(1.0, r_span, z_span, absolute_magnitude)
    orientation_tolerance = (
        64.0 * np.finfo(np.float64).eps * length_scale * length_scale
    )

    located: list[LocatedLocalRefinementSource] = []
    for source, point in zip(sources, points):
        first_cross = (
            (second[:, 0] - first[:, 0]) * (point[1] - first[:, 1])
            - (second[:, 1] - first[:, 1]) * (point[0] - first[:, 0])
        )
        second_cross = (
            (third[:, 0] - second[:, 0]) * (point[1] - second[:, 1])
            - (third[:, 1] - second[:, 1]) * (point[0] - second[:, 0])
        )
        third_cross = (
            (first[:, 0] - third[:, 0]) * (point[1] - third[:, 1])
            - (first[:, 1] - third[:, 1]) * (point[0] - third[:, 0])
        )
        positive = denominator > 0.0
        candidates = np.flatnonzero(
            np.where(
                positive,
                (first_cross >= -orientation_tolerance)
                & (second_cross >= -orientation_tolerance)
                & (third_cross >= -orientation_tolerance),
                (first_cross <= orientation_tolerance)
                & (second_cross <= orientation_tolerance)
                & (third_cross <= orientation_tolerance),
            )
        )
        if candidates.size == 0:
            raise ValueError(
                "local refinement point 位于计算域外："
                f"({source.r}, {source.z})"
            )

        candidate_regions = {
            int(region_id)
            for region_id in mesh.region_ids[candidates].tolist()
        }
        if len(candidate_regions) != 1:
            raise ValueError(
                "local refinement point 位于或无法区分于 Region interface："
                f"({source.r}, {source.z})，候选 Region={sorted(candidate_regions)}"
            )
        region_id = next(iter(candidate_regions))
        baseline_dx = float(effective_dx[region_id])
        if not (
            math.isfinite(baseline_dx)
            and baseline_dx > 0.0
            and source.dx < baseline_dx
        ):
            raise ValueError(
                "local refinement DX 必须严格小于 source Region baseline dx："
                f"DX={source.dx}, Region {region_id} baseline={baseline_dx}"
            )
        located.append(
            LocatedLocalRefinementSource(
                r=source.r,
                z=source.z,
                dx=source.dx,
                width=source.width,
                region_id=region_id,
                baseline_dx=baseline_dx,
            )
        )
    return tuple(located)


def _centroid_to_segments_distance(
    centroids: np.ndarray,
    nodes: np.ndarray,
    segments: np.ndarray,
) -> np.ndarray:
    """Compute deterministic minimum Euclidean centroid-to-segment distances."""
    centroids = np.ascontiguousarray(centroids, dtype=np.float64)
    segments = np.ascontiguousarray(segments, dtype=np.int32)
    if centroids.ndim != 2 or centroids.shape[1] != 2:
        raise ValueError("centroids 必须为 (K,2)")
    if segments.ndim != 2 or segments.shape[1] != 2 or len(segments) == 0:
        raise ValueError("interface segments 必须为非空 (S,2)")

    result = np.full(centroids.shape[0], np.inf, dtype=np.float64)
    segment_starts = nodes[segments[:, 0]]
    segment_ends = nodes[segments[:, 1]]
    vectors = segment_ends - segment_starts
    lengths_squared = np.sum(vectors * vectors, axis=1)
    if np.any(lengths_squared <= 0.0):
        raise ValueError("interface 含零长度 segment")

    for centroid_start in range(
        0,
        centroids.shape[0],
        _DISTANCE_CHUNK_SIZE,
    ):
        centroid_stop = min(
            centroid_start + _DISTANCE_CHUNK_SIZE,
            centroids.shape[0],
        )
        centroid_chunk = centroids[centroid_start:centroid_stop]
        chunk_minimum = np.full(len(centroid_chunk), np.inf, dtype=np.float64)

        for segment_start in range(
            0,
            len(segments),
            _DISTANCE_CHUNK_SIZE,
        ):
            segment_stop = min(
                segment_start + _DISTANCE_CHUNK_SIZE,
                len(segments),
            )
            starts = segment_starts[segment_start:segment_stop]
            directions = vectors[segment_start:segment_stop]
            denominators = lengths_squared[segment_start:segment_stop]

            offsets = centroid_chunk[:, None, :] - starts[None, :, :]
            parameters = np.sum(
                offsets * directions[None, :, :],
                axis=2,
            ) / denominators[None, :]
            parameters = np.clip(parameters, 0.0, 1.0)
            closest = (
                starts[None, :, :]
                + parameters[:, :, None] * directions[None, :, :]
            )
            differences = centroid_chunk[:, None, :] - closest
            squared_distances = np.sum(differences * differences, axis=2)
            chunk_minimum = np.minimum(
                chunk_minimum,
                np.min(squared_distances, axis=1),
            )

        result[centroid_start:centroid_stop] = np.sqrt(chunk_minimum)

    if not np.all(np.isfinite(result)):
        raise ValueError("centroid-to-interface distance 计算得到 NaN/Inf")
    return result


def _interface_transition_width(
    interface: InterfaceConstraint,
    transition_width: float | None,
    transition_gradient: float | None,
) -> float:
    """Return the effective width for one interface and validate the profile."""
    fine_dx = float(interface.fine_dx)
    coarse_dx = float(interface.coarse_dx)
    if not (
        math.isfinite(fine_dx)
        and math.isfinite(coarse_dx)
        and fine_dx > 0.0
        and coarse_dx > fine_dx
    ):
        raise ValueError("transition interface 必须满足有限正数 hc > hf")

    if transition_gradient is not None:
        logarithmic_ratio = math.log(coarse_dx / fine_dx)
        width = logarithmic_ratio / transition_gradient
        if not (
            math.isfinite(logarithmic_ratio)
            and logarithmic_ratio > 0.0
            and math.isfinite(width)
            and width > 0.0
        ):
            raise ValueError("gradient 模式无法得到有限、严格正的 interface Wi")
        return width

    if transition_width is None or not (
        math.isfinite(transition_width) and transition_width > 0.0
    ):
        raise ValueError("transition_width 必须是有限正数")
    return transition_width


def _triangle_area_bounds(
    mesh: Mesh2D,
    dx_by_region_id: Mapping[int, float],
    transition_width: float | None,
    topology: EdgeTopology | None = None,
    *,
    transition_gradient: float | None = None,
    local_sources: tuple[LocatedLocalRefinementSource, ...] = (),
) -> np.ndarray:
    """Construct C-contiguous positive per-triangle exponential area bounds."""
    if transition_gradient is not None and not (
        math.isfinite(transition_gradient) and transition_gradient > 0.0
    ):
        raise ValueError("transition_gradient 必须是有限正数")

    topology = (
        _extract_edge_topology(mesh, dx_by_region_id)
        if topology is None
        else topology
    )
    region_dx = np.array(
        [dx_by_region_id[int(region_id)] for region_id in mesh.region_ids],
        dtype=np.float64,
    )
    target_size = np.array(region_dx, dtype=np.float64, copy=True)
    centroids = np.mean(mesh.nodes[mesh.triangles], axis=1)
    transition_enabled = (
        transition_gradient is not None
        or (
            transition_width is not None
            and transition_width > 0.0
        )
    )
    active_interfaces = (
        topology.interfaces if transition_enabled else ()
    )

    for interface in active_interfaces:
        interface_width = _interface_transition_width(
            interface,
            transition_width,
            transition_gradient,
        )
        triangle_indices = np.flatnonzero(
            mesh.region_ids == interface.coarse_region_id
        )
        if triangle_indices.size == 0:
            raise ValueError(
                f"粗侧 Region {interface.coarse_region_id} 没有 triangle"
            )
        distances = _centroid_to_segments_distance(
            centroids[triangle_indices],
            mesh.nodes,
            interface.segments,
        )
        exponent = np.minimum(distances / interface_width, 1.0)
        candidate_size = interface.fine_dx * np.power(
            interface.coarse_dx / interface.fine_dx,
            exponent,
        )
        candidate_size = np.clip(
            candidate_size,
            interface.fine_dx,
            interface.coarse_dx,
        )
        target_size[triangle_indices] = np.minimum(
            target_size[triangle_indices],
            candidate_size,
        )

    for source in local_sources:
        triangle_indices = np.flatnonzero(
            mesh.region_ids == source.region_id
        )
        if triangle_indices.size == 0:
            raise ValueError(
                f"local refinement source Region {source.region_id} 没有 triangle"
            )
        differences = centroids[triangle_indices] - np.array(
            [source.r, source.z],
            dtype=np.float64,
        )
        distances = np.linalg.norm(differences, axis=1)
        exponent = np.minimum(distances / source.width, 1.0)
        candidate_size = source.dx * np.power(
            source.baseline_dx / source.dx,
            exponent,
        )
        candidate_size = np.clip(
            candidate_size,
            source.dx,
            source.baseline_dx,
        )
        target_size[triangle_indices] = np.minimum(
            target_size[triangle_indices],
            candidate_size,
        )

    area_bounds = np.ascontiguousarray(
        0.5 * target_size * target_size,
        dtype=np.float64,
    )
    if (
        area_bounds.shape != (mesh.n_triangles,)
        or not np.all(np.isfinite(area_bounds))
        or np.any(area_bounds <= 0.0)
    ):
        raise ValueError("trianglearealist 必须是有限、严格正的 (T,) float64")
    return area_bounds


def _triangle_areas(mesh: Mesh2D) -> np.ndarray:
    first = mesh.nodes[mesh.triangles[:, 0]]
    second = mesh.nodes[mesh.triangles[:, 1]]
    third = mesh.nodes[mesh.triangles[:, 2]]
    return 0.5 * np.abs(
        (second[:, 0] - first[:, 0]) * (third[:, 1] - first[:, 1])
        - (third[:, 0] - first[:, 0]) * (second[:, 1] - first[:, 1])
    )


def _area_violations(mesh: Mesh2D, area_bounds: np.ndarray) -> np.ndarray:
    areas = _triangle_areas(mesh)
    tolerance = _AREA_EPSILON_FACTOR * np.finfo(np.float64).eps * np.maximum(
        1.0,
        np.maximum(np.abs(areas), np.abs(area_bounds)),
    )
    return areas > area_bounds + tolerance


def _maximum_area_excess(
    areas: np.ndarray,
    area_bounds: np.ndarray,
) -> float:
    excess = np.maximum(0.0, areas / area_bounds - 1.0)
    if not np.all(np.isfinite(excess)):
        raise ValueError("refinement area excess 含 NaN/Inf")
    return float(np.max(excess, initial=0.0))


def _refinement_diagnostics(
    *,
    initial_mesh: Mesh2D,
    final_mesh: Mesh2D,
    effective_dx: Mapping[int, float],
    transition_width: float | None,
    transition_gradient: float | None,
    local_sources: tuple[LocatedLocalRefinementSource, ...],
    iteration_count: int,
    topology: EdgeTopology,
) -> RefinementDiagnostics:
    """Build final diagnostics from the same topology and area rules."""
    transition_enabled = (
        bool(topology.interfaces)
        and (
            transition_gradient is not None
            or (transition_width is not None and transition_width > 0.0)
        )
    )
    active_interfaces = (
        topology.interfaces if transition_enabled else ()
    )
    enabled = transition_enabled or bool(local_sources)
    if not enabled:
        return RefinementDiagnostics(
            enabled=False,
            transition_mode="disabled",
            transition_width=None,
            transition_gradient=None,
            iteration_count=0,
            initial_node_count=initial_mesh.n_nodes,
            initial_triangle_count=initial_mesh.n_triangles,
            final_node_count=final_mesh.n_nodes,
            final_triangle_count=final_mesh.n_triangles,
            transition_interface_count=0,
            affected_coarse_triangle_count=0,
            final_violating_triangle_count=None,
            final_maximum_area_excess=None,
            interfaces=(),
            local_refinement_enabled=False,
            local_sources=(),
        )

    centroids = np.mean(final_mesh.nodes[final_mesh.triangles], axis=1)
    areas = _triangle_areas(final_mesh)
    area_bounds = _triangle_area_bounds(
        final_mesh,
        effective_dx,
        transition_width,
        topology,
        transition_gradient=transition_gradient,
        local_sources=local_sources,
    )
    violations = _area_violations(final_mesh, area_bounds)

    affected_indices: set[int] = set()
    interface_diagnostics: list[InterfaceRefinementDiagnostics] = []
    for interface in active_interfaces:
        interface_width = _interface_transition_width(
            interface,
            transition_width,
            transition_gradient,
        )
        coarse_indices = np.flatnonzero(
            final_mesh.region_ids == interface.coarse_region_id
        )
        distances = _centroid_to_segments_distance(
            centroids[coarse_indices],
            final_mesh.nodes,
            interface.segments,
        )
        affected_mask = distances < interface_width
        affected = coarse_indices[affected_mask]
        affected_indices.update(int(index) for index in affected.tolist())

        if affected.size:
            exponent = np.minimum(
                distances[affected_mask] / interface_width,
                1.0,
            )
            candidate_size = interface.fine_dx * np.power(
                interface.coarse_dx / interface.fine_dx,
                exponent,
            )
            candidate_bounds = 0.5 * candidate_size * candidate_size
            maximum_excess = _maximum_area_excess(
                areas[affected],
                candidate_bounds,
            )
        else:
            maximum_excess = 0.0

        interface_diagnostics.append(
            InterfaceRefinementDiagnostics(
                region_ids=tuple(
                    sorted(
                        (
                            interface.fine_region_id,
                            interface.coarse_region_id,
                        )
                    )
                ),
                fine_region_id=interface.fine_region_id,
                coarse_region_id=interface.coarse_region_id,
                fine_dx=interface.fine_dx,
                coarse_dx=interface.coarse_dx,
                transition_width=interface_width,
                affected_coarse_triangle_count=int(affected.size),
                maximum_area_excess=maximum_excess,
            )
        )

    local_diagnostics: list[LocalRefinementDiagnostics] = []
    for source in local_sources:
        source_indices = np.flatnonzero(
            final_mesh.region_ids == source.region_id
        )
        differences = centroids[source_indices] - np.array(
            [source.r, source.z],
            dtype=np.float64,
        )
        distances = np.linalg.norm(differences, axis=1)
        affected_mask = distances < source.width
        affected = source_indices[affected_mask]
        if affected.size:
            exponent = np.minimum(
                distances[affected_mask] / source.width,
                1.0,
            )
            candidate_size = source.dx * np.power(
                source.baseline_dx / source.dx,
                exponent,
            )
            maximum_excess = _maximum_area_excess(
                areas[affected],
                0.5 * candidate_size * candidate_size,
            )
        else:
            maximum_excess = 0.0
        local_diagnostics.append(
            LocalRefinementDiagnostics(
                r=source.r,
                z=source.z,
                dx=source.dx,
                width=source.width,
                region_id=source.region_id,
                baseline_dx=source.baseline_dx,
                affected_triangle_count=int(affected.size),
                maximum_area_excess=maximum_excess,
            )
        )

    return RefinementDiagnostics(
        enabled=True,
        transition_mode=(
            "gradient"
            if transition_enabled and transition_gradient is not None
            else "width"
            if transition_enabled
            else "disabled"
        ),
        transition_width=transition_width,
        transition_gradient=transition_gradient,
        iteration_count=iteration_count,
        initial_node_count=initial_mesh.n_nodes,
        initial_triangle_count=initial_mesh.n_triangles,
        final_node_count=final_mesh.n_nodes,
        final_triangle_count=final_mesh.n_triangles,
        transition_interface_count=len(active_interfaces),
        affected_coarse_triangle_count=len(affected_indices),
        final_violating_triangle_count=int(np.count_nonzero(violations)),
        final_maximum_area_excess=_maximum_area_excess(areas, area_bounds),
        interfaces=tuple(interface_diagnostics),
        local_refinement_enabled=bool(local_sources),
        local_sources=tuple(local_diagnostics),
    )


def _result(
    *,
    initial_mesh: Mesh2D,
    final_mesh: Mesh2D,
    effective_dx: Mapping[int, float],
    transition_width: float | None,
    transition_gradient: float | None,
    local_sources: tuple[LocatedLocalRefinementSource, ...],
    iteration_count: int,
    topology: EdgeTopology,
) -> RefinementResult:
    return RefinementResult(
        mesh=final_mesh,
        diagnostics=_refinement_diagnostics(
            initial_mesh=initial_mesh,
            final_mesh=final_mesh,
            effective_dx=effective_dx,
            transition_width=transition_width,
            transition_gradient=transition_gradient,
            local_sources=local_sources,
            iteration_count=iteration_count,
            topology=topology,
        ),
    )


def _canonical_segment_set(segments: np.ndarray) -> set[tuple[int, int]]:
    canonical: set[tuple[int, int]] = set()
    for raw_first, raw_second in np.asarray(segments):
        first, second = int(raw_first), int(raw_second)
        if first == second:
            raise ValueError("PSLG 含零长度 segment")
        edge = (first, second) if first < second else (second, first)
        if edge in canonical:
            raise ValueError(f"PSLG 含重复 segment：{edge}")
        canonical.add(edge)
    return canonical


def _validate_output_pslg(
    topology: EdgeTopology,
    native_segments: np.ndarray,
) -> np.ndarray:
    segments = np.asarray(native_segments, dtype=np.int64)
    if segments.ndim != 2 or segments.shape[1] != 2 or len(segments) == 0:
        raise ValueError("Triangle refinement 输出缺少合法 PSLG segments")
    if (
        np.any(segments < 0)
        or np.any(segments > np.iinfo(np.int32).max)
    ):
        raise ValueError("Triangle refinement 输出 PSLG index 非法")

    output_segments = np.ascontiguousarray(segments, dtype=np.int32)
    expected = _canonical_segment_set(topology.pslg_segments)
    actual = _canonical_segment_set(output_segments)
    if actual != expected:
        raise ValueError(
            "Triangle refinement 输出 PSLG 与最终 mesh topology 不一致："
            f"缺失={sorted(expected - actual)}，多余={sorted(actual - expected)}"
        )
    return output_segments


def _load_triangle_runtime() -> tuple[Any, Any, Any]:
    """Load the installed OFT 26.6 Triangle binding without importing at startup."""
    try:
        meshing = importlib.import_module(
            "OpenFUSIONToolkit.TokaMaker.meshing"
        )
        triangle_struct = meshing.triangle_struct
        triangulate = meshing.oft_triangulate
        triangle_library = meshing.oft_triangle_lib
    except Exception as exc:
        raise ImportError(
            "无法导入 OFT 26.6 Triangle runtime：需要已安装且可加载的 "
            "OpenFUSIONToolkit.TokaMaker.meshing.triangle_struct、"
            "oft_triangulate 和 oft_triangle_lib"
        ) from exc

    try:
        trifree = triangle_library.trifree
        trifree.argtypes = [ctypes.c_void_p]
        trifree.restype = None
    except Exception as exc:
        raise ImportError(
            "OFT 26.6 liboft_triangle 未提供可绑定的 trifree(void*)"
        ) from exc

    return triangle_struct, triangulate, trifree


def _native_refine(
    mesh: Mesh2D,
    area_bounds: np.ndarray,
    segments: np.ndarray,
) -> tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
    """Execute one ``rpqazQ`` call and return only Python-owned output copies."""
    nodes = np.ascontiguousarray(mesh.nodes, dtype=np.float64)
    triangles = np.ascontiguousarray(mesh.triangles, dtype=np.int32)
    attributes = np.ascontiguousarray(
        mesh.region_ids.astype(np.float64).reshape(-1, 1)
    )
    area_bounds = np.ascontiguousarray(area_bounds, dtype=np.float64)
    segments = np.ascontiguousarray(segments, dtype=np.int32)

    if nodes.shape != (mesh.n_nodes, 2):
        raise ValueError("native pointlist shape 非法")
    if triangles.shape != (mesh.n_triangles, 3):
        raise ValueError("native trianglelist shape 非法")
    if attributes.shape != (mesh.n_triangles, 1):
        raise ValueError("native triangleattributelist shape 非法")
    if area_bounds.shape != (mesh.n_triangles,):
        raise ValueError("native trianglearealist shape 非法")
    if segments.ndim != 2 or segments.shape[1] != 2 or len(segments) == 0:
        raise ValueError("native segmentlist 必须为非空 (S,2)")
    if triangles.min() < 0 or segments.min() < 0:
        raise ValueError("native Triangle 输入必须使用 0-based 非负索引")
    if triangles.max() >= len(nodes) or segments.max() >= len(nodes):
        raise ValueError("native Triangle 输入索引越界")
    if not np.all(np.isfinite(area_bounds)) or np.any(area_bounds <= 0.0):
        raise ValueError("native trianglearealist 必须有限且严格正")

    triangle_struct, triangulate, trifree = _load_triangle_runtime()
    input_struct = triangle_struct()
    output_struct = triangle_struct()

    input_struct.pointlist = nodes.ctypes.data_as(
        type(input_struct.pointlist)
    )
    input_struct.numberofpoints = nodes.shape[0]
    input_struct.numberofpointattributes = 0
    input_struct.trianglelist = triangles.ctypes.data_as(
        type(input_struct.trianglelist)
    )
    input_struct.triangleattributelist = attributes.ctypes.data_as(
        type(input_struct.triangleattributelist)
    )
    input_struct.trianglearealist = area_bounds.ctypes.data_as(
        type(input_struct.trianglearealist)
    )
    input_struct.numberoftriangles = triangles.shape[0]
    input_struct.numberofcorners = 3
    input_struct.numberoftriangleattributes = 1
    input_struct.segmentlist = segments.ctypes.data_as(
        type(input_struct.segmentlist)
    )
    input_struct.numberofsegments = segments.shape[0]
    input_struct.numberofholes = 0
    input_struct.numberofregions = 0

    owned_output_fields = (
        "pointlist",
        "pointmarkerlist",
        "trianglelist",
        "triangleattributelist",
        "segmentlist",
        "segmentmarkerlist",
    )
    copied_output: tuple[
        np.ndarray,
        np.ndarray,
        np.ndarray,
        np.ndarray,
    ] | None = None
    native_error: BaseException | None = None
    cleanup_errors: list[str] = []

    try:
        return_code = int(
            triangulate(
                _OPTION_STRING,
                ctypes.byref(input_struct),
                ctypes.byref(output_struct),
                None,
            )
        )
        if return_code != 0:
            raise ValueError(
                "Triangle rpqazQ refinement 失败："
                f"native return code={return_code}"
            )

        n_points = int(output_struct.numberofpoints)
        n_triangles = int(output_struct.numberoftriangles)
        n_corners = int(output_struct.numberofcorners)
        n_attributes = int(output_struct.numberoftriangleattributes)
        n_segments = int(output_struct.numberofsegments)
        if n_points <= 0 or n_triangles <= 0 or n_segments <= 0:
            raise ValueError("Triangle rpqazQ 返回空 points/triangles/segments")
        if n_corners != 3:
            raise ValueError(
                f"Triangle rpqazQ numberofcorners 应为 3，实际={n_corners}"
            )
        if n_attributes != 1:
            raise ValueError(
                "Triangle rpqazQ 必须保留唯一 Region attribute，"
                f"实际 attributes={n_attributes}"
            )
        for field_name in (
            "pointlist",
            "trianglelist",
            "triangleattributelist",
            "segmentlist",
        ):
            if not bool(getattr(output_struct, field_name)):
                raise ValueError(
                    f"Triangle rpqazQ 缺少输出指针 {field_name}"
                )

        output_nodes = np.ctypeslib.as_array(
            output_struct.pointlist,
            shape=(n_points * 2,),
        ).reshape(n_points, 2).copy()
        output_triangles = np.ctypeslib.as_array(
            output_struct.trianglelist,
            shape=(n_triangles * 3,),
        ).reshape(n_triangles, 3).copy()
        output_attributes = np.ctypeslib.as_array(
            output_struct.triangleattributelist,
            shape=(n_triangles,),
        ).copy()
        output_segments = np.ctypeslib.as_array(
            output_struct.segmentlist,
            shape=(n_segments * 2,),
        ).reshape(n_segments, 2).copy()

        copied_output = (
            np.ascontiguousarray(output_nodes, dtype=np.float64),
            np.ascontiguousarray(output_triangles, dtype=np.int32),
            np.ascontiguousarray(output_attributes, dtype=np.float64),
            np.ascontiguousarray(output_segments, dtype=np.int32),
        )
    except BaseException as exc:
        native_error = exc
    finally:
        for field_name in owned_output_fields:
            pointer = getattr(output_struct, field_name)
            if not bool(pointer):
                continue
            try:
                trifree(ctypes.cast(pointer, ctypes.c_void_p))
            except Exception as exc:
                cleanup_errors.append(f"{field_name}: {exc}")
            finally:
                setattr(output_struct, field_name, type(pointer)())

    if cleanup_errors:
        message = (
            "Triangle rpqazQ native ownership 清理失败："
            + "; ".join(cleanup_errors)
        )
        if native_error is not None:
            raise RuntimeError(message) from native_error
        raise RuntimeError(message)
    if native_error is not None:
        raise native_error
    if copied_output is None:
        raise RuntimeError("Triangle rpqazQ 未产生 Python-owned 输出副本")
    return copied_output


def refine_transition_mesh(
    mesh: Mesh2D,
    dx_by_region_id: Mapping[int, float],
    transition_width: float | None = 0.0,
    *,
    transition_gradient: float | None = None,
    local_sources: Any = (),
) -> RefinementResult:
    """Return the unique final Mesh2D and immutable refinement diagnostics.

    A zero width or absence of different-dx interfaces returns an equivalent
    independently owned Mesh2D without loading or calling the native runtime.
    """
    if transition_width is not None and transition_gradient is not None:
        raise ValueError("transition_width 与 transition_gradient 互斥")

    width: float | None
    gradient: float | None
    if transition_gradient is not None:
        if isinstance(transition_gradient, bool):
            raise ValueError("transition_gradient 必须是有限正数")
        try:
            gradient = float(transition_gradient)
        except (TypeError, ValueError) as exc:
            raise ValueError("transition_gradient 必须是有限正数") from exc
        if not (math.isfinite(gradient) and gradient > 0.0):
            raise ValueError("transition_gradient 必须是有限正数")
        width = None
    else:
        gradient = None
        if isinstance(transition_width, bool):
            raise ValueError("transition_width 必须是有限非负数")
        try:
            width = float(transition_width)
        except (TypeError, ValueError) as exc:
            raise ValueError("transition_width 必须是有限非负数") from exc
        if not (math.isfinite(width) and width >= 0.0):
            raise ValueError("transition_width 必须是有限非负数")

    current = _validate_mesh_arrays(mesh)
    original_region_ids = {
        int(value) for value in current.region_ids.tolist()
    }
    effective_dx = _validate_effective_dx(
        current.region_ids,
        dx_by_region_id,
    )
    initial = _owned_mesh_copy(current)
    normalized_sources = _normalize_local_sources(local_sources)
    located_sources = _locate_local_sources(
        initial,
        effective_dx,
        normalized_sources,
    )
    topology = _extract_edge_topology(current, effective_dx)
    transition_enabled = (
        bool(topology.interfaces)
        and (gradient is not None or width > 0.0)
    )
    if not transition_enabled and not located_sources:
        return _result(
            initial_mesh=initial,
            final_mesh=current,
            effective_dx=effective_dx,
            transition_width=width,
            transition_gradient=gradient,
            local_sources=located_sources,
            iteration_count=0,
            topology=topology,
        )

    native_segments = np.array(
        topology.pslg_segments,
        dtype=np.int32,
        order="C",
        copy=True,
    )

    for refinement_round in range(1, _MAX_REFINEMENT_ROUNDS + 1):
        area_bounds = _triangle_area_bounds(
            current,
            effective_dx,
            width,
            topology,
            transition_gradient=gradient,
            local_sources=located_sources,
        )
        if not np.any(_area_violations(current, area_bounds)):
            return _result(
                initial_mesh=initial,
                final_mesh=current,
                effective_dx=effective_dx,
                transition_width=width,
                transition_gradient=gradient,
                local_sources=located_sources,
                iteration_count=refinement_round - 1,
                topology=topology,
            )

        previous_size = (current.n_nodes, current.n_triangles)
        (
            output_nodes,
            output_triangles,
            output_attributes,
            output_segments,
        ) = _native_refine(current, area_bounds, native_segments)

        candidate = Mesh2D(
            nodes=output_nodes,
            triangles=output_triangles,
            region_ids=output_attributes,
            boundary_segments=np.empty((0, 2), dtype=np.int32),
            boundary_markers=np.empty((0,), dtype=np.int32),
            node_markers=np.empty((0,), dtype=np.int32),
            outer_boundary=(
                None
                if mesh.outer_boundary is None
                else np.array(
                    mesh.outer_boundary,
                    dtype=np.float64,
                    copy=True,
                )
            ),
        )
        candidate = _validate_mesh_arrays(
            candidate,
            expected_region_ids=original_region_ids,
        )
        topology = _extract_edge_topology(candidate, effective_dx)
        native_segments = _validate_output_pslg(
            topology,
            output_segments,
        )

        next_bounds = _triangle_area_bounds(
            candidate,
            effective_dx,
            width,
            topology,
            transition_gradient=gradient,
            local_sources=located_sources,
        )
        violations = _area_violations(candidate, next_bounds)
        if not np.any(violations):
            return _result(
                initial_mesh=initial,
                final_mesh=candidate,
                effective_dx=effective_dx,
                transition_width=width,
                transition_gradient=gradient,
                local_sources=located_sources,
                iteration_count=refinement_round,
                topology=topology,
            )

        current_size = (candidate.n_nodes, candidate.n_triangles)
        if current_size == previous_size:
            raise ValueError(
                "Triangle rpqazQ 输出 mesh 规模未变化，但仍存在 "
                f"{int(np.count_nonzero(violations))} 个面积违规 triangle"
            )

        current = candidate

    final_bounds = _triangle_area_bounds(
        current,
        effective_dx,
        width,
        topology,
        transition_gradient=gradient,
        local_sources=located_sources,
    )
    violation_count = int(
        np.count_nonzero(_area_violations(current, final_bounds))
    )
    raise ValueError(
        f"Triangle transition refinement 达到 {_MAX_REFINEMENT_ROUNDS} 轮上限，"
        f"仍有 {violation_count} 个面积违规 triangle"
    )
