"""Reader for the approved multi-file boundary-list input."""

from __future__ import annotations

import os
import re
import tomllib
from dataclasses import dataclass
from pathlib import Path
from typing import Any

import numpy as np

from linkmesh.mesh import (
    _signed_area_closed,
    _validate_outer_boundary,
)


_COUNT_PATTERN = re.compile(r"[0-9]+")
_COMMENT_PATTERN = re.compile(r"^(.*?)(?:\s+#.*)?$")


class BoundaryListError(ValueError):
    """Raised when a boundary-list or referenced coordinate file is invalid."""


@dataclass(frozen=True)
class BoundaryRecord:
    """One ordered boundary-list record with its stable input curve ID."""

    curve_id: int
    points: np.ndarray
    source_path: Path


def _error(list_path: Path, message: str) -> BoundaryListError:
    return BoundaryListError(f"[boundary list {list_path}] {message}")


def _strip_comment(line: str) -> str:
    return line.split("#", 1)[0].strip()


def _parse_curve_id(value: Any, list_path: Path, index: int) -> int:
    if isinstance(value, bool) or not isinstance(value, int) or value <= 0:
        raise _error(
            list_path,
            f"第 {index} 个 boundary record 的 curve_id 必须是正整数",
        )
    return value


def _resolve_unique_path(
    list_path: Path,
    raw_path: str,
    index: int,
) -> Path:
    if not isinstance(raw_path, str) or not raw_path:
        raise _error(
            list_path,
            f"第 {index} 个 boundary record 的 file 必须是非空字符串",
        )

    candidate = Path(raw_path)
    if not candidate.is_absolute():
        candidate = list_path.parent / candidate
    resolved = candidate.resolve(strict=False)

    try:
        identity = resolved.stat().st_ino, resolved.stat().st_dev
    except OSError:
        identity = None

    return resolved if identity is None else _PathIdentity(resolved, identity)


@dataclass(frozen=True)
class _PathIdentity:
    path: Path
    identity: tuple[int, int]

    def __fspath__(self) -> str:
        return os.fspath(self.path)

    def __str__(self) -> str:
        return str(self.path)

    def __getattr__(self, name: str) -> Any:
        return getattr(self.path, name)


def _read_coordinate_file(path: Path, list_path: Path, index: int) -> np.ndarray:
    if not path.is_file():
        raise _error(
            list_path,
            f"第 {index} 个 boundary record 引用的坐标文件不存在或不是普通文件：{path}",
        )

    try:
        raw_lines = path.read_text(encoding="utf-8").splitlines()
    except (OSError, UnicodeError) as exc:
        raise _error(
            list_path,
            f"第 {index} 个 boundary record 无法读取坐标文件 {path}：{exc}",
        ) from exc

    lines = [
        (line_number, _strip_comment(line))
        for line_number, line in enumerate(raw_lines, start=1)
    ]
    lines = [(line_number, line) for line_number, line in lines if line]

    if not lines or _COUNT_PATTERN.fullmatch(lines[0][1]) is None:
        raise _error(
            list_path,
            f"第 {index} 个 boundary record 的坐标文件 {path} "
            "缺少合法计数头",
        )

    count = int(lines[0][1], 10)
    if count < 3:
        raise _error(
            list_path,
            f"第 {index} 个 boundary record 的坐标文件 {path} "
            f"N 必须不小于 3，实际 N={count}",
        )

    data_lines = lines[1:]
    if len(data_lines) != count:
        raise _error(
            list_path,
            f"第 {index} 个 boundary record 的坐标文件 {path} "
            f"声明 N={count}，但有 {len(data_lines)} 行有效坐标",
        )

    points = np.empty((count, 2), dtype=np.float64)
    for point_index, (line_number, line) in enumerate(data_lines, start=1):
        fields = line.split()
        if len(fields) != 2:
            raise _error(
                list_path,
                f"第 {index} 个 boundary record 的坐标文件 {path} "
                f"第 {line_number} 行必须恰好包含两个字段",
            )
        try:
            r_value, z_value = (float(value) for value in fields)
        except ValueError as exc:
            raise _error(
                list_path,
                f"第 {index} 个 boundary record 的坐标文件 {path} "
                f"第 {line_number} 行含非法数值",
            ) from exc
        if not np.isfinite(r_value) or not np.isfinite(z_value):
            raise _error(
                list_path,
                f"第 {index} 个 boundary record 的坐标文件 {path} "
                f"第 {line_number} 行含 NaN/Inf",
            )
        points[point_index - 1] = (r_value, z_value)

    if np.array_equal(points[0], points[-1]):
        raise _error(
            list_path,
            f"第 {index} 个 boundary record 的坐标文件 {path} "
            "末点不得重复首点",
        )

    if _signed_area_closed(points) > 0.0:
        points = np.concatenate((points[:1], points[:0:-1]), axis=0)

    try:
        validated = _validate_outer_boundary(points)
    except ValueError as exc:
        raise _error(
            list_path,
            f"第 {index} 个 boundary record 的坐标文件 {path} 几何非法：{exc}",
        ) from exc
    return np.array(validated, dtype=np.float64, copy=True)


def read_boundary_list(filepath: str | Path) -> tuple[BoundaryRecord, ...]:
    """Read a schema-v1 boundary list and all referenced coordinate files."""
    list_path = Path(filepath).resolve(strict=False)
    if not list_path.is_file():
        raise FileNotFoundError(f"boundary list 文件不存在或不是普通文件：{list_path}")

    try:
        with list_path.open("rb") as file_object:
            raw = tomllib.load(file_object)
    except (OSError, tomllib.TOMLDecodeError) as exc:
        raise BoundaryListError(
            f"[boundary list {list_path}] 无法读取或解析 TOML：{exc}"
        ) from exc

    if set(raw) != {"schema_version", "boundaries"}:
        raise _error(
            list_path,
            "顶层只允许 schema_version 和 boundaries",
        )
    if (
        isinstance(raw["schema_version"], bool)
        or raw["schema_version"] != 1
    ):
        raise _error(list_path, "schema_version 必须是整数 1")
    if not isinstance(raw["boundaries"], list) or len(raw["boundaries"]) < 2:
        raise _error(list_path, "boundaries 至少需要两个记录")

    explicit_ids: set[int] = set()
    for index, item in enumerate(raw["boundaries"], start=1):
        if not isinstance(item, dict):
            raise _error(list_path, f"第 {index} 个 boundary record 必须是表")
        if set(item) - {"file", "curve_id"}:
            raise _error(
                list_path,
                f"第 {index} 个 boundary record 含未知字段",
            )
        if "curve_id" in item:
            curve_id = _parse_curve_id(item["curve_id"], list_path, index)
            if curve_id in explicit_ids:
                raise _error(
                    list_path,
                    f"第 {index} 个 boundary record 的 curve_id={curve_id} 重复",
                )
            explicit_ids.add(curve_id)

    used_ids = set(explicit_ids)
    next_id = 1
    records: list[BoundaryRecord] = []
    seen_paths: dict[tuple[int, int] | Path, tuple[int, Path]] = {}

    for index, item in enumerate(raw["boundaries"], start=1):
        if "file" not in item:
            raise _error(
                list_path,
                f"第 {index} 个 boundary record 缺少 file",
            )

        if "curve_id" in item:
            curve_id = _parse_curve_id(item["curve_id"], list_path, index)
        else:
            while next_id in used_ids:
                next_id += 1
            curve_id = next_id
            used_ids.add(curve_id)
            next_id += 1

        resolved = _resolve_unique_path(list_path, item["file"], index)
        identity = (
            resolved.identity
            if isinstance(resolved, _PathIdentity)
            else Path(resolved)
        )
        path = resolved.path if isinstance(resolved, _PathIdentity) else Path(resolved)

        if identity in seen_paths:
            previous_index, previous_path = seen_paths[identity]
            raise _error(
                list_path,
                f"第 {index} 个 boundary record 与第 {previous_index} 个 "
                f"record 重复引用文件；解析路径均为 {path}，"
                f"此前路径为 {previous_path}",
            )
        seen_paths[identity] = (index, path)

        records.append(
            BoundaryRecord(
                curve_id=curve_id,
                points=_read_coordinate_file(path, list_path, index),
                source_path=path,
            )
        )

    return tuple(records)
