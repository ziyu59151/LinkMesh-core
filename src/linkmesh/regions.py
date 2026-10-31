"""Multi-array Region inspection models and deterministic business logic."""

from __future__ import annotations

import math
import os
import re
import tempfile
import tomllib
from dataclasses import dataclass, replace
from pathlib import Path
from typing import Any, Mapping, Sequence

import numpy as np


REGIONS_MAP_SCHEMA_VERSION = 1

_COLORS = (
    "#0072B2",
    "#D55E00",
    "#009E73",
    "#CC79A7",
    "#E69F00",
    "#56B4E9",
    "#F0E442",
    "#000000",
    "#332288",
    "#88CCEE",
    "#44AA99",
    "#117733",
    "#999933",
    "#DDCC77",
    "#CC6677",
    "#882255",
    "#AA4499",
    "#661100",
    "#6699CC",
    "#888888",
)
_COLOR_PATTERN = re.compile(r"#[0-9A-Fa-f]{6}")


class RegionsError(ValueError):
    """Raised when Region inspection data violates the formal contract."""


@dataclass(frozen=True)
class RegionEntry:
    """One input curve and its derived inspection metadata."""

    curve_id: int
    points: np.ndarray
    parent_curve_id: int
    child_curve_ids: tuple[int, ...]
    depth: int
    technical_role: str
    region_id: int | None = None
    color: str = ""
    physical_label: str = ""


@dataclass(frozen=True)
class RegionPlan:
    """A deterministic unique-root Region tree in input curve order."""

    entries: tuple[RegionEntry, ...]
    root_curve_id: int

    def by_curve_id(self) -> dict[int, RegionEntry]:
        return {entry.curve_id: entry for entry in self.entries}


@dataclass(frozen=True)
class RegionMapEntry:
    """One public TOML regions-map record."""

    region_id: int
    curve_id: int
    parent_region_id: int
    child_region_ids: tuple[int, ...]
    color: str
    physical_label: str = ""


def _signed_area(points: np.ndarray) -> float:
    following = np.roll(points, -1, axis=0)
    return 0.5 * float(
        np.sum(
            points[:, 0] * following[:, 1]
            - following[:, 0] * points[:, 1]
        )
    )


def _validate_curve(points_in: Any, curve_id: int) -> tuple[np.ndarray, float]:
    points = np.array(points_in, dtype=np.float64, copy=True)
    if points.ndim != 2 or points.shape[1] != 2:
        raise RegionsError(
            f"curve_id={curve_id} 必须为 (N,2)，实际 shape={points.shape}"
        )
    if points.shape[0] < 3:
        raise RegionsError(f"curve_id={curve_id} 至少需要 3 个点")
    if not np.all(np.isfinite(points)):
        raise RegionsError(f"curve_id={curve_id} 含 NaN/Inf")
    if np.array_equal(points[0], points[-1]):
        raise RegionsError(
            f"curve_id={curve_id} 必须隐式闭合，末点不得重复首点"
        )

    differences = np.roll(points, -1, axis=0) - points
    if np.any(np.sum(differences * differences, axis=1) == 0.0):
        raise RegionsError(
            f"curve_id={curve_id} 存在相邻重复点或零长度闭合边"
        )
    if np.unique(points, axis=0).shape[0] < 3:
        raise RegionsError(f"curve_id={curve_id} 的互异点不足 3 个")

    area = _signed_area(points)
    r_span = float(np.ptp(points[:, 0]))
    z_span = float(np.ptp(points[:, 1]))
    scale = max(1.0, r_span, z_span, float(np.max(np.abs(points))))
    if abs(area) <= 1.0e-12 * scale * scale:
        raise RegionsError(
            f"curve_id={curve_id} 面积明显退化：signed_area={area}"
        )
    return points, abs(area)


def _point_on_segment(
    point: np.ndarray,
    first: np.ndarray,
    second: np.ndarray,
) -> bool:
    scale = max(
        1.0,
        float(np.max(np.abs(point))),
        float(np.max(np.abs(first))),
        float(np.max(np.abs(second))),
    )
    coordinate_tolerance = 1.0e-12 * scale
    orientation_tolerance = 1.0e-12 * scale * scale
    cross = (
        (second[0] - first[0]) * (point[1] - first[1])
        - (second[1] - first[1]) * (point[0] - first[0])
    )
    return (
        abs(float(cross)) <= orientation_tolerance
        and min(first[0], second[0]) - coordinate_tolerance
        <= point[0]
        <= max(first[0], second[0]) + coordinate_tolerance
        and min(first[1], second[1]) - coordinate_tolerance
        <= point[1]
        <= max(first[1], second[1]) + coordinate_tolerance
    )


def _point_in_polygon(point: np.ndarray, polygon: np.ndarray) -> bool | None:
    """Return True inside, False outside, and None on the polygon boundary."""
    inside = False
    previous = polygon[-1]

    for current in polygon:
        if _point_on_segment(point, previous, current):
            return None
        if (previous[1] > point[1]) != (current[1] > point[1]):
            intersection_r = previous[0] + (
                (point[1] - previous[1])
                * (current[0] - previous[0])
                / (current[1] - previous[1])
            )
            if point[0] < intersection_r:
                inside = not inside
        previous = current

    return inside


def _interior_point(polygon: np.ndarray, curve_id: int) -> np.ndarray:
    """Find a deterministic scan-line point strictly inside a simple polygon."""
    candidate = np.mean(polygon, axis=0)
    if _point_in_polygon(candidate, polygon) is True:
        return np.array(candidate, dtype=np.float64, copy=True)

    z_values = np.unique(polygon[:, 1])
    scan_levels = [
        0.5 * float(z_values[index] + z_values[index + 1])
        for index in range(len(z_values) - 1)
        if z_values[index] < z_values[index + 1]
    ]

    for z_value in scan_levels:
        intersections: list[float] = []
        previous = polygon[-1]
        for current in polygon:
            if (previous[1] > z_value) != (current[1] > z_value):
                intersections.append(
                    float(
                        previous[0]
                        + (z_value - previous[1])
                        * (current[0] - previous[0])
                        / (current[1] - previous[1])
                    )
                )
            previous = current

        intersections.sort()
        intervals = [
            (intersections[index], intersections[index + 1])
            for index in range(0, len(intersections) - 1, 2)
            if intersections[index] < intersections[index + 1]
        ]
        for first, second in sorted(
            intervals,
            key=lambda interval: (-(interval[1] - interval[0]), interval[0]),
        ):
            candidate = np.array(
                [0.5 * (first + second), z_value],
                dtype=np.float64,
            )
            if _point_in_polygon(candidate, polygon) is True:
                return candidate

    raise RegionsError(
        f"curve_id={curve_id} 无法取得无歧义内部点；请检查曲线是否简单闭合"
    )


def build_region_plan(blocks: Sequence[np.ndarray]) -> RegionPlan:
    """Build a deterministic unique-root containment tree from boundary records.

    ``blocks`` may contain the legacy ordered ``(N, 2)`` arrays or objects
    exposing ``curve_id`` and ``points`` (the boundary-list reader record).
    Both forms enter the same containment and Region-plan algorithm.
    """
    if len(blocks) < 2:
        raise RegionsError(
            "multi-region 输入至少需要两个 blocks；"
            "单一区域请使用 array_file + single_region"
        )

    curves: dict[int, np.ndarray] = {}
    areas: dict[int, float] = {}
    interior_points: dict[int, np.ndarray] = {}

    for index, block in enumerate(blocks, start=1):
        if hasattr(block, "curve_id") and hasattr(block, "points"):
            curve_id = int(block.curve_id)
            points_in = block.points
        else:
            curve_id = index
            points_in = block

        if curve_id <= 0 or curve_id in curves:
            raise RegionsError(
                f"curve_id={curve_id} 非法或重复"
            )
        points, area = _validate_curve(points_in, curve_id)
        curves[curve_id] = points
        areas[curve_id] = area
        interior_points[curve_id] = _interior_point(points, curve_id)

    containing: dict[int, set[int]] = {
        curve_id: set() for curve_id in curves
    }
    for child_id, child_point in interior_points.items():
        for candidate_id, candidate_polygon in curves.items():
            if (
                child_id == candidate_id
                or areas[candidate_id] <= areas[child_id]
            ):
                continue
            relation = _point_in_polygon(child_point, candidate_polygon)
            if relation is None:
                raise RegionsError(
                    f"curve_id={child_id} 的内部点落在 curve_id={candidate_id} "
                    "边界上，包含关系有歧义"
                )
            if relation:
                containing[child_id].add(candidate_id)

    parent_by_id: dict[int, int] = {}
    for curve_id in curves:
        candidates = containing[curve_id]
        if candidates:
            parent_by_id[curve_id] = min(
                candidates,
                key=lambda candidate_id: (areas[candidate_id], candidate_id),
            )
        else:
            parent_by_id[curve_id] = 0

    roots = sorted(
        curve_id
        for curve_id, parent_id in parent_by_id.items()
        if parent_id == 0
    )
    if len(roots) != 1:
        raise RegionsError(
            "multi_array_file 必须形成唯一 root，"
            f"实际 root curve_id={roots}；请检查分离、相交或歧义输入"
        )
    root_id = roots[0]

    children_by_id: dict[int, list[int]] = {
        curve_id: [] for curve_id in curves
    }
    for curve_id, parent_id in parent_by_id.items():
        if parent_id:
            children_by_id[parent_id].append(curve_id)
    for children in children_by_id.values():
        children.sort()

    depth_by_id: dict[int, int] = {}

    def resolve_depth(curve_id: int, visiting: set[int]) -> int:
        if curve_id in depth_by_id:
            return depth_by_id[curve_id]
        if curve_id in visiting:
            raise RegionsError("包含关系存在环")
        parent_id = parent_by_id[curve_id]
        depth = (
            0
            if parent_id == 0
            else resolve_depth(parent_id, visiting | {curve_id}) + 1
        )
        depth_by_id[curve_id] = depth
        return depth

    for curve_id in curves:
        resolve_depth(curve_id, set())

    if any(
        curve_id != root_id and depth_by_id[curve_id] <= 0
        for curve_id in curves
    ):
        raise RegionsError("存在未连接到唯一 root 的曲线")

    internal_leaves = [
        curve_id
        for curve_id in curves
        if curve_id != root_id and not children_by_id[curve_id]
    ]
    if not internal_leaves:
        raise RegionsError("multi-array Region tree 缺少内部 leaf")
    plasma_curve_id = min(
        internal_leaves,
        key=lambda curve_id: (-depth_by_id[curve_id], curve_id),
    )

    entries = tuple(
        RegionEntry(
            curve_id=curve_id,
            points=np.array(curves[curve_id], dtype=np.float64, copy=True),
            parent_curve_id=parent_by_id[curve_id],
            child_curve_ids=tuple(children_by_id[curve_id]),
            depth=depth_by_id[curve_id],
            technical_role=(
                "boundary"
                if curve_id == root_id
                else "plasma"
                if curve_id == plasma_curve_id
                else "vacuum"
            ),
        )
        for curve_id in sorted(curves)
    )
    return RegionPlan(entries=entries, root_curve_id=root_id)


def assign_final_region_ids(
    plan: RegionPlan,
    region_id_by_curve_id: Mapping[int, int],
) -> RegionPlan:
    """Attach unique positive OFT final IDs and deterministic display colors."""
    curve_ids = {entry.curve_id for entry in plan.entries}
    if set(region_id_by_curve_id) != curve_ids:
        raise RegionsError(
            "OFT Region ID 映射的 curve_id 集合与 Region plan 不一致"
        )

    normalized: dict[int, int] = {}
    for curve_id, raw_region_id in region_id_by_curve_id.items():
        if isinstance(raw_region_id, bool):
            raise RegionsError("OFT Region ID 必须是正整数")
        numeric = float(raw_region_id)
        if not math.isfinite(numeric) or numeric != round(numeric) or numeric <= 0:
            raise RegionsError(
                f"curve_id={curve_id} 的 OFT Region ID 非法：{raw_region_id!r}"
            )
        normalized[curve_id] = int(numeric)

    if len(set(normalized.values())) != len(normalized):
        raise RegionsError("OFT 最终 Region ID 存在重复")

    color_by_region_id = {
        region_id: _COLORS[index % len(_COLORS)]
        for index, region_id in enumerate(sorted(normalized.values()))
    }
    return RegionPlan(
        entries=tuple(
            replace(
                entry,
                region_id=normalized[entry.curve_id],
                color=color_by_region_id[normalized[entry.curve_id]],
            )
            for entry in plan.entries
        ),
        root_curve_id=plan.root_curve_id,
    )


def region_map_entries(plan: RegionPlan) -> tuple[RegionMapEntry, ...]:
    """Convert a numbered plan to validated public map records."""
    by_curve = plan.by_curve_id()
    if any(entry.region_id is None for entry in plan.entries):
        raise RegionsError("Region plan 尚未取得完整 OFT 最终 Region ID")

    records = tuple(
        RegionMapEntry(
            region_id=int(entry.region_id),
            curve_id=entry.curve_id,
            parent_region_id=(
                0
                if entry.parent_curve_id == 0
                else int(by_curve[entry.parent_curve_id].region_id)
            ),
            child_region_ids=tuple(
                sorted(
                    int(by_curve[child_id].region_id)
                    for child_id in entry.child_curve_ids
                )
            ),
            color=entry.color,
            physical_label=entry.physical_label,
        )
        for entry in sorted(
            plan.entries,
            key=lambda item: int(item.region_id),
        )
    )
    validate_region_map(records)
    return records


def validate_region_map(records: Sequence[RegionMapEntry]) -> None:
    """Validate unique IDs, references, colors, and bidirectional topology."""
    if len(records) < 2:
        raise RegionsError("regions map 至少需要两个 Region")

    by_region_id: dict[int, RegionMapEntry] = {}
    curve_ids: set[int] = set()
    for record in records:
        if record.region_id <= 0 or record.region_id in by_region_id:
            raise RegionsError("regions map 含非正或重复 region_id")
        if record.curve_id <= 0 or record.curve_id in curve_ids:
            raise RegionsError("regions map 含非正或重复 curve_id")
        if record.parent_region_id == record.region_id:
            raise RegionsError("regions map 不允许 Region 自引用为 parent")
        if record.region_id in record.child_region_ids:
            raise RegionsError("regions map 不允许 Region 自引用为 child")
        if tuple(sorted(set(record.child_region_ids))) != record.child_region_ids:
            raise RegionsError("child_region_ids 必须唯一且升序")
        if _COLOR_PATTERN.fullmatch(record.color) is None:
            raise RegionsError(f"非法 Region 颜色：{record.color!r}")
        if not isinstance(record.physical_label, str):
            raise RegionsError("physical_label 必须是 TOML 字符串")
        by_region_id[record.region_id] = record
        curve_ids.add(record.curve_id)

    roots = [
        record.region_id
        for record in records
        if record.parent_region_id == 0
    ]
    if len(roots) != 1:
        raise RegionsError("regions map 必须恰好包含一个 parent_region_id=0 的 root")

    known_ids = set(by_region_id)
    for record in records:
        if record.parent_region_id not in known_ids | {0}:
            raise RegionsError("regions map 引用了未知 parent_region_id")
        if not set(record.child_region_ids) <= known_ids:
            raise RegionsError("regions map 引用了未知 child_region_id")
        for child_id in record.child_region_ids:
            child_parent_id = by_region_id[child_id].parent_region_id
            if child_parent_id != record.region_id:
                raise RegionsError(
                    f"Region ID {record.region_id} 的 child_region_ids 包含 "
                    f"{child_id}，但该 child 的 parent_region_id="
                    f"{child_parent_id}"
                )
        if record.parent_region_id:
            parent = by_region_id[record.parent_region_id]
            if record.region_id not in parent.child_region_ids:
                raise RegionsError(
                    f"Region ID {record.region_id} 的 parent_region_id="
                    f"{record.parent_region_id}，但该 parent 的 "
                    f"child_region_ids 不包含 {record.region_id}"
                )

    visit_state: dict[int, int] = {
        region_id: 0 for region_id in known_ids
    }

    def check_cycle(region_id: int) -> None:
        state = visit_state[region_id]
        if state == 1:
            raise RegionsError(
                f"regions map 存在 cycle，涉及 Region ID {region_id}"
            )
        if state == 2:
            return

        visit_state[region_id] = 1
        for child_id in by_region_id[region_id].child_region_ids:
            check_cycle(child_id)
        visit_state[region_id] = 2

    for region_id in sorted(known_ids):
        if visit_state[region_id] == 0:
            check_cycle(region_id)

    reachable: set[int] = set()

    def visit_from_root(region_id: int) -> None:
        if region_id in reachable:
            raise RegionsError(
                "regions map 从 root 遍历时重复访问 "
                f"Region ID {region_id}"
            )
        reachable.add(region_id)
        for child_id in by_region_id[region_id].child_region_ids:
            visit_from_root(child_id)

    visit_from_root(roots[0])
    if reachable != known_ids:
        unreachable = sorted(known_ids - reachable)
        raise RegionsError(
            "regions map 含 root 不可达或 disconnected 的 Region："
            f"{unreachable}"
        )


def _toml_string(value: str) -> str:
    escaped = (
        value.replace("\\", "\\\\")
        .replace('"', '\\"')
        .replace("\n", "\\n")
        .replace("\r", "\\r")
        .replace("\t", "\\t")
    )
    return f'"{escaped}"'


def serialize_regions_map(records: Sequence[RegionMapEntry]) -> str:
    """Serialize the minimal schema deterministically in Region-ID order."""
    ordered = tuple(sorted(records, key=lambda record: record.region_id))
    validate_region_map(ordered)

    lines = [f"schema_version = {REGIONS_MAP_SCHEMA_VERSION}", ""]
    for record in ordered:
        children = ", ".join(str(value) for value in record.child_region_ids)
        lines.extend(
            (
                "[[regions]]",
                f"region_id = {record.region_id}",
                f"curve_id = {record.curve_id}",
                f"parent_region_id = {record.parent_region_id}",
                f"child_region_ids = [{children}]",
                f"color = {_toml_string(record.color)}",
                f"physical_label = {_toml_string(record.physical_label)}",
                "",
            )
        )
    return "\n".join(lines)


def validate_plan_against_map(
    plan: RegionPlan,
    records: Sequence[RegionMapEntry],
) -> None:
    """Validate rebuilt Region identity/topology against an inspected map.

    Display color and physical_label are intentionally excluded because they
    are user-editable metadata. Any structural or numbering mismatch requires
    the user to run ``linkmesh regions inspect`` again.
    """
    validate_region_map(records)

    if any(entry.region_id is None for entry in plan.entries):
        raise RegionsError("重建 Region plan 尚未取得完整 OFT 最终 Region ID")

    rebuilt = {
        record.curve_id: record
        for record in region_map_entries(plan)
    }
    mapped = {
        record.curve_id: record
        for record in records
    }

    if set(rebuilt) != set(mapped):
        raise RegionsError(
            "regions map 的 curve_id 集合与重建结果不一致；"
            "请重新执行 linkmesh regions inspect"
        )

    rebuilt_region_ids = {
        record.region_id for record in rebuilt.values()
    }
    mapped_region_ids = {
        record.region_id for record in mapped.values()
    }
    if rebuilt_region_ids != mapped_region_ids:
        raise RegionsError(
            "regions map 的 Region ID 集合与 OFT 重建结果不一致；"
            "请重新执行 linkmesh regions inspect"
        )

    for curve_id in sorted(rebuilt):
        expected = rebuilt[curve_id]
        actual = mapped[curve_id]
        if actual.region_id != expected.region_id:
            raise RegionsError(
                f"curve_id={curve_id} 的 OFT Region ID 已变化："
                f"map={actual.region_id}，当前={expected.region_id}；"
                "请重新执行 linkmesh regions inspect"
            )
        if actual.parent_region_id != expected.parent_region_id:
            raise RegionsError(
                f"Region ID {actual.region_id} 的 parent_region_id 已变化；"
                "请重新执行 linkmesh regions inspect"
            )
        if actual.child_region_ids != expected.child_region_ids:
            raise RegionsError(
                f"Region ID {actual.region_id} 的 child_region_ids 已变化；"
                "请重新执行 linkmesh regions inspect"
            )


def resolve_region_dx(
    records: Sequence[RegionMapEntry],
    mesh_dx: float | None,
    region_dx: Mapping[int, float],
) -> dict[int, float]:
    """Resolve a complete independent final-Region-ID to dx mapping."""
    validate_region_map(records)
    known_ids = {record.region_id for record in records}

    default_dx: float | None = None
    if mesh_dx is not None:
        if isinstance(mesh_dx, bool):
            raise RegionsError("mesh_dx 必须是有限正数")
        try:
            default_dx = float(mesh_dx)
        except (TypeError, ValueError) as exc:
            raise RegionsError("mesh_dx 必须是有限正数") from exc
        if not (math.isfinite(default_dx) and default_dx > 0.0):
            raise RegionsError("mesh_dx 必须是有限正数")

    normalized_overrides: dict[int, float] = {}
    for raw_region_id, raw_dx in region_dx.items():
        if isinstance(raw_region_id, bool):
            raise RegionsError("region_dx 的 Region ID 必须是正整数")
        try:
            numeric_id = float(raw_region_id)
        except (TypeError, ValueError) as exc:
            raise RegionsError(
                f"region_dx 的 Region ID 非法：{raw_region_id!r}"
            ) from exc
        if (
            not math.isfinite(numeric_id)
            or numeric_id != round(numeric_id)
            or numeric_id <= 0
        ):
            raise RegionsError(
                f"region_dx 的 Region ID 非法：{raw_region_id!r}"
            )
        region_id = int(numeric_id)
        if region_id not in known_ids:
            raise RegionsError(f"region_dx 引用了未知 Region ID {region_id}")

        if isinstance(raw_dx, bool):
            raise RegionsError(
                f"Region ID {region_id} 的 dx 必须是有限正数"
            )
        try:
            dx = float(raw_dx)
        except (TypeError, ValueError) as exc:
            raise RegionsError(
                f"Region ID {region_id} 的 dx 必须是有限正数"
            ) from exc
        if not (math.isfinite(dx) and dx > 0.0):
            raise RegionsError(
                f"Region ID {region_id} 的 dx 必须是有限正数"
            )
        normalized_overrides[region_id] = dx

    missing = known_ids - set(normalized_overrides)
    if default_dx is None and missing:
        raise RegionsError(
            "未提供 mesh_dx 时，region_dx 必须完整覆盖全部 Region ID；"
            f"缺失={sorted(missing)}"
        )

    return {
        region_id: normalized_overrides.get(region_id, default_dx)
        for region_id in sorted(known_ids)
    }


def write_regions_map(
    records: Sequence[RegionMapEntry],
    filepath: str | os.PathLike[str],
) -> None:
    """Atomically write a deterministic TOML regions map."""
    path = Path(filepath)
    if not path.parent.is_dir():
        raise FileNotFoundError(f"regions map 父目录不存在：{path.parent}")

    content = serialize_regions_map(records)
    temporary_path: Path | None = None
    try:
        with tempfile.NamedTemporaryFile(
            "w",
            encoding="utf-8",
            newline="\n",
            dir=path.parent,
            prefix=f".{path.name}.",
            suffix=".tmp",
            delete=False,
        ) as file_object:
            file_object.write(content)
            file_object.flush()
            os.fsync(file_object.fileno())
            temporary_path = Path(file_object.name)
        os.replace(temporary_path, path)
    finally:
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def read_regions_map(
    filepath: str | os.PathLike[str],
) -> tuple[RegionMapEntry, ...]:
    """Read and strictly validate the minimal TOML regions-map schema."""
    path = Path(filepath)
    try:
        with path.open("rb") as file_object:
            raw = tomllib.load(file_object)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise RegionsError(f"无法读取 regions map {path}：{exc}") from exc

    if set(raw) != {"schema_version", "regions"}:
        raise RegionsError("regions map 顶层只允许 schema_version 和 regions")
    if raw["schema_version"] != REGIONS_MAP_SCHEMA_VERSION:
        raise RegionsError("regions map schema_version 必须为整数 1")
    if not isinstance(raw["regions"], list):
        raise RegionsError("regions map 的 [[regions]] 必须是数组表")

    required_keys = {
        "region_id",
        "curve_id",
        "parent_region_id",
        "child_region_ids",
        "color",
        "physical_label",
    }
    records: list[RegionMapEntry] = []
    for index, item in enumerate(raw["regions"], start=1):
        if not isinstance(item, dict) or set(item) != required_keys:
            raise RegionsError(
                f"regions map 第 {index} 项字段必须精确为 {sorted(required_keys)}"
            )
        if (
            any(
                isinstance(item[key], bool) or not isinstance(item[key], int)
                for key in ("region_id", "curve_id", "parent_region_id")
            )
            or not isinstance(item["child_region_ids"], list)
            or any(
                isinstance(value, bool) or not isinstance(value, int)
                for value in item["child_region_ids"]
            )
            or not isinstance(item["color"], str)
            or not isinstance(item["physical_label"], str)
        ):
            raise RegionsError(f"regions map 第 {index} 项字段类型非法")
        records.append(
            RegionMapEntry(
                region_id=item["region_id"],
                curve_id=item["curve_id"],
                parent_region_id=item["parent_region_id"],
                child_region_ids=tuple(item["child_region_ids"]),
                color=item["color"],
                physical_label=item["physical_label"],
            )
        )

    ordered = tuple(sorted(records, key=lambda record: record.region_id))
    validate_region_map(ordered)
    return ordered
