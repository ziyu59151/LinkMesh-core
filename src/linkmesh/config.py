"""Strict configuration loading for root and isolated regions workflows.

The root command resolves one single-region configuration. Multi-region
settings remain isolated in the regions inspect/generate configuration types.
This module performs no business-file I/O and invokes no readers, meshers,
writers, or plotters.

Path-key convention:
    - CliOverrides 使用 CLI 短键名：gfile/array_file/hdf5/plot/model/smb（与 argparse
      dest 一致）；
    - _load_toml 输出与 EffectiveConfig 字段名一致的长键名：gfile_path/array_file_path/
      hdf5_path/plot_path/model_path/smb_path；
    - 因此路径合并时必须分别用「CLI 键」查 CliOverrides、用「EffectiveConfig 键」查
      TOML 映射，二者不可混用同一名称（标量字段两侧键名恰好相同，不受影响）。

"""

from __future__ import annotations

import math
import tomllib
from dataclasses import dataclass
from pathlib import Path
from types import MappingProxyType
from typing import Any, Literal, Mapping, Optional


# --------------------------------------------------------------------------- #
# 枚举与内建默认值（正式唯一来源，第 6A.2 / 6A.3 节）                           #
# --------------------------------------------------------------------------- #

SCHEMA_VERSION = 1

TARGETS = ("mesh_only", "trimeg", "m3dc1")
BOUNDARY_SOURCES = (
    "lcfs",
    "first_wall",
    "array_file",
    "flux_surface",
)

DEFAULT_TARGET = "trimeg"
DEFAULT_BOUNDARY_SOURCE = "lcfs"
DEFAULT_MESH_DX = 0.08
DEFAULT_VERBOSE = False
DEFAULT_PLOT_LCFS = False
DEFAULT_TRANSITION_WIDTH = 0.0

# 路径默认值（第 6A.3 节，经只读核对实时 cli.py）：
#   - hdf5 默认 mesh.h5（仅当 target=trimeg 需要时作为默认输出）；
#   - plot 默认 None（不生成 PNG）；
#   - model/smb 不制造尚未存在的可执行默认路径（首批 None）。
DEFAULT_HDF5_NAME = "mesh.h5"

# 允许的 TOML section 与逐 section 键（第 6A.4 节）。
_ALLOWED_TOP_LEVEL_KEYS = {"schema_version"}
_ALLOWED_SECTIONS = {"task", "input", "domain", "mesh", "output", "runtime"}
_ALLOWED_SECTION_KEYS = {
    "task": {"target"},
    "input": {"gfile", "array_file"},
    "domain": {"boundary_source"},
    "mesh": {"mesh_dx", "local_refinement"},
    "output": {
        "hdf5",
        "plot",
        "quality_plot",
        "model",
        "smb",
        "plot_lcfs",
    },
    "runtime": {"verbose"},
}


# --------------------------------------------------------------------------- #
# 异常                                                                          #
# --------------------------------------------------------------------------- #

class ConfigError(ValueError):
    """配置读取、校验或合并阶段的错误（ValueError 子类，便于 CLI 统一捕获）。"""


# --------------------------------------------------------------------------- #
# EffectiveConfig（frozen dataclass，13 字段）                                 #
# --------------------------------------------------------------------------- #

@dataclass(frozen=True)
class LocalRefinementConfig:
    """One immutable R-Z point local-refinement configuration record."""

    r: float
    z: float
    dx: float
    width: float


@dataclass(frozen=True)
class EffectiveConfig:
    """LinkMesh 只读/不可变有效配置（合并「默认 < TOML < 显式 CLI」的最终结果）。

    字段逐项对齐当前蓝图字段表（共 13 字段）。所有非 None Path
    均为规范化绝对路径。不保存值来源字典、NumPy 数组、句柄或 Namespace。
    """

    schema_version: int
    config_path: Optional[Path]
    target: Literal["mesh_only", "trimeg", "m3dc1"]
    boundary_source: Literal[
        "lcfs", "first_wall", "array_file", "flux_surface"
    ]
    gfile_path: Optional[Path]
    array_file_path: Optional[Path]
    mesh_dx: float
    hdf5_path: Optional[Path]
    plot_path: Optional[Path]
    quality_plot_path: Optional[Path]
    model_path: Optional[Path]
    smb_path: Optional[Path]
    verbose: bool
    plot_lcfs: bool
    local_refinement: tuple[LocalRefinementConfig, ...]


# --------------------------------------------------------------------------- #
# CLI 覆盖载体（内部 sentinel：未提供的字段保持缺省，不进入最终对象）          #
# --------------------------------------------------------------------------- #

class _Unset:
    """哨兵：表示某 CLI 字段未由用户显式提供。"""

    _instance = None

    def __new__(cls):
        if cls._instance is None:
            cls._instance = super().__new__(cls)
        return cls._instance

    def __repr__(self) -> str:  # pragma: no cover - 仅调试
        return "UNSET"


UNSET = _Unset()


@dataclass
class RegionsInspectOverrides:
    """Explicit CLI values for the isolated regions-inspect configuration."""

    multi_array_file: Any = UNSET
    boundary_list: Any = UNSET
    regions_map: Any = UNSET
    plot: Any = UNSET
    verbose: Any = UNSET


@dataclass(frozen=True)
class RegionsInspectConfig:
    """Resolved configuration for ``linkmesh regions inspect`` only."""

    schema_version: int
    config_path: Optional[Path]
    mode: Literal["inspect"]
    multi_array_file_path: Optional[Path]
    regions_map_path: Path
    plot_path: Path
    verbose: bool
    boundary_list_path: Optional[Path] = None


@dataclass
class RegionsGenerateOverrides:
    """Explicit CLI values for the isolated regions-generate configuration."""

    target: Any = UNSET
    multi_array_file: Any = UNSET
    boundary_list: Any = UNSET
    regions_map: Any = UNSET
    plot: Any = UNSET
    quality_plot: Any = UNSET
    hdf5: Any = UNSET
    model: Any = UNSET
    smb: Any = UNSET
    mesh_dx: Any = UNSET
    region_dx: Any = UNSET
    transition_width: Any = UNSET
    transition_gradient: Any = UNSET
    local_refinement: Any = UNSET
    verbose: Any = UNSET


@dataclass(frozen=True)
class RegionsGenerateConfig:
    """Resolved configuration for ``linkmesh regions generate`` only."""

    schema_version: int
    config_path: Optional[Path]
    mode: Literal["generate"]
    target: Literal["mesh_only", "trimeg", "m3dc1"]
    multi_array_file_path: Optional[Path]
    regions_map_path: Path
    plot_path: Optional[Path]
    quality_plot_path: Optional[Path]
    hdf5_path: Optional[Path]
    model_path: Optional[Path]
    smb_path: Optional[Path]
    mesh_dx: Optional[float]
    region_dx: Mapping[int, float]
    transition_mode: Literal["disabled", "width", "gradient"]
    transition_width: Optional[float]
    transition_gradient: Optional[float]
    local_refinement: tuple[LocalRefinementConfig, ...]
    verbose: bool
    boundary_list_path: Optional[Path] = None


@dataclass
class CliOverrides:
    """CLI 收集的用户显式输入（未提供者保持 UNSET）。

    该对象是合并输入，不是最终配置；仅传给 load_effective_config。
    路径字段以字符串保存（相对进程当前工作目录解析）。

    注意：cli.py 会在构造后对 verbose 字段赋值（overrides.verbose = True），
    因此本类必须为可变 dataclass（不加 frozen）。

    路径字段键名使用 CLI 短名（gfile/array_file/hdf5/plot/model/smb），与
    EffectiveConfig 的长名（gfile_path 等）不同，合并逻辑需分别映射。
    """

    target: Any = UNSET
    boundary_source: Any = UNSET
    gfile: Any = UNSET
    array_file: Any = UNSET
    mesh_dx: Any = UNSET
    hdf5: Any = UNSET
    plot: Any = UNSET
    quality_plot: Any = UNSET
    model: Any = UNSET
    smb: Any = UNSET
    verbose: Any = UNSET
    plot_lcfs: Any = UNSET
    local_refinement: Any = UNSET


# --------------------------------------------------------------------------- #
# 内部辅助：类型/枚举校验                                                       #
# --------------------------------------------------------------------------- #

def _fail(config_path: Optional[Path], where: str, msg: str) -> None:
    prefix = f"[配置文件 {config_path}] " if config_path is not None else ""
    raise ConfigError(f"{prefix}{where}：{msg}")


def _check_enum(value: Any, allowed: tuple, name: str,
                config_path: Optional[Path]) -> str:
    if not isinstance(value, str) or value not in allowed:
        _fail(
            config_path, name,
            f"取值非法，实际={value!r}，允许={list(allowed)}",
        )
    return value


def _check_positive_float(value: Any, name: str,
                          config_path: Optional[Path]) -> float:
    # TOML 中数值可能被解析为 int 或 float；bool 是 int 子类，需排除。
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(config_path, name, f"必须是数值，实际类型={type(value).__name__}")
    v = float(value)
    if not (math.isfinite(v) and v > 0.0):
        _fail(config_path, name, f"必须是有限正数，实际={value}")
    return v


def _check_nonnegative_float(
    value: Any,
    name: str,
    config_path: Optional[Path],
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(config_path, name, f"必须是数值，实际类型={type(value).__name__}")
    numeric = float(value)
    if not (math.isfinite(numeric) and numeric >= 0.0):
        _fail(config_path, name, f"必须是有限非负数，实际={value}")
    return numeric


def _check_finite_float(
    value: Any,
    name: str,
    config_path: Optional[Path],
) -> float:
    if isinstance(value, bool) or not isinstance(value, (int, float)):
        _fail(config_path, name, f"必须是数值，实际类型={type(value).__name__}")
    numeric = float(value)
    if not math.isfinite(numeric):
        _fail(config_path, name, f"必须是有限数，实际={value}")
    return numeric


def _parse_local_refinement_records(
    records: Any,
    name: str,
    config_path: Optional[Path],
) -> tuple[LocalRefinementConfig, ...]:
    if records is None:
        return ()
    if not isinstance(records, list):
        _fail(config_path, name, "必须是 TOML array-of-tables")

    expected_keys = {"r", "z", "dx", "width"}
    parsed: list[LocalRefinementConfig] = []
    for index, record in enumerate(records, start=1):
        item_name = f"{name}[{index}]"
        if not isinstance(record, dict):
            _fail(config_path, item_name, "必须是 TOML table")
        actual_keys = set(record)
        if actual_keys != expected_keys:
            missing = sorted(expected_keys - actual_keys)
            unknown = sorted(actual_keys - expected_keys)
            _fail(
                config_path,
                item_name,
                f"字段必须精确为 {sorted(expected_keys)}；"
                f"缺失={missing}，未知={unknown}",
            )
        parsed.append(
            LocalRefinementConfig(
                r=_check_finite_float(
                    record["r"], f"{item_name}.r", config_path
                ),
                z=_check_finite_float(
                    record["z"], f"{item_name}.z", config_path
                ),
                dx=_check_positive_float(
                    record["dx"], f"{item_name}.dx", config_path
                ),
                width=_check_positive_float(
                    record["width"], f"{item_name}.width", config_path
                ),
            )
        )
    return tuple(parsed)


def _parse_local_refinement_tokens(
    tokens: Any,
    config_path: Optional[Path],
) -> tuple[LocalRefinementConfig, ...]:
    if tokens is UNSET or tokens is None:
        return ()
    if not isinstance(tokens, (list, tuple)):
        _fail(config_path, "--local-refinement", "内部值必须是 token 序列")

    records: list[LocalRefinementConfig] = []
    for token in tokens:
        if not isinstance(token, str):
            _fail(
                config_path,
                "--local-refinement",
                f"必须使用 R,Z,DX,W 格式，实际={token!r}",
            )
        parts = token.split(",")
        if len(parts) != 4 or any(part.strip() == "" for part in parts):
            _fail(
                config_path,
                "--local-refinement",
                f"必须使用 R,Z,DX,W 格式，实际={token!r}",
            )
        try:
            r, z, dx, width = (float(part.strip()) for part in parts)
        except ValueError as exc:
            raise ConfigError(
                "--local-refinement：必须使用四个浮点数 R,Z,DX,W，"
                f"实际={token!r}"
            ) from exc
        records.append(
            LocalRefinementConfig(
                r=_check_finite_float(
                    r, "--local-refinement R", config_path
                ),
                z=_check_finite_float(
                    z, "--local-refinement Z", config_path
                ),
                dx=_check_positive_float(
                    dx, "--local-refinement DX", config_path
                ),
                width=_check_positive_float(
                    width, "--local-refinement W", config_path
                ),
            )
        )
    return tuple(records)


def _check_bool(value: Any, name: str, config_path: Optional[Path]) -> bool:
    if not isinstance(value, bool):
        _fail(config_path, name, f"必须是布尔值，实际类型={type(value).__name__}")
    return value


def _check_str_path(value: Any, name: str, config_path: Optional[Path]) -> str:
    if not isinstance(value, str) or value == "":
        _fail(config_path, name, f"必须是非空字符串路径，实际={value!r}")
    return value


# --------------------------------------------------------------------------- #
# TOML 读取与严格 schema 校验（第 6A.4 节）                                     #
# --------------------------------------------------------------------------- #

def _load_toml(config_path: Path) -> dict:
    """读取并严格校验 linkmesh.in，返回已按 EffectiveConfig 字段名归一的原始值字典。

    返回的字典只包含 TOML 中实际出现的键（映射到 EffectiveConfig 字段名，
    路径字段用长名 gfile_path/array_file_path/hdf5_path/plot_path/model_path/smb_path），
    未出现的键不放入（表示未提供）。路径值保持字符串（相对配置目录解析在后续步骤）。
    """
    try:
        with open(config_path, "rb") as f:
            raw = tomllib.load(f)
    except tomllib.TOMLDecodeError as e:
        raise ConfigError(
            f"[配置文件 {config_path}] TOML 语法错误：{e}"
        ) from e
    except OSError as e:
        raise ConfigError(
            f"[配置文件 {config_path}] 无法读取：{e}"
        ) from e

    if not isinstance(raw, dict):
        _fail(config_path, "顶层", "TOML 顶层必须是表（键值映射）")

    # 顶层：schema_version 必须存在且为整数 1；其余顶层非 section 键立即失败。
    if "schema_version" not in raw:
        _fail(config_path, "schema_version", "缺失，v1 schema 要求顶层 schema_version = 1")

    sv = raw["schema_version"]
    if isinstance(sv, bool) or not isinstance(sv, int) or sv != SCHEMA_VERSION:
        _fail(
            config_path, "schema_version",
            f"必须是整数 {SCHEMA_VERSION}，实际={sv!r}",
        )

    # 分离 section 与非 section 顶层键。
    for key, val in raw.items():
        if key in _ALLOWED_TOP_LEVEL_KEYS:
            continue
        # 允许的 section 是表；其余顶层键（无论是否为表）都视为未知。
        if key in _ALLOWED_SECTIONS:
            if not isinstance(val, dict):
                _fail(config_path, f"[{key}]", "必须是 TOML 表（section）")
            continue
        if isinstance(val, dict):
            _fail(config_path, f"[{key}]", "未知 section，不被 v1 schema 允许")
        _fail(config_path, key, "未知顶层键，不被 v1 schema 允许")

    # 逐 section 校验未知键。
    for section, allowed_keys in _ALLOWED_SECTION_KEYS.items():
        if section not in raw:
            continue
        body = raw[section]
        for k in body.keys():
            if k not in allowed_keys:
                _fail(
                    config_path, f"[{section}].{k}",
                    f"未知键，[{section}] 只允许 {sorted(allowed_keys)}",
                )

    # 抽取并类型校验，映射到 EffectiveConfig 字段名。
    out: dict = {"schema_version": SCHEMA_VERSION}

    task = raw.get("task", {})
    if "target" in task:
        out["target"] = _check_enum(task["target"], TARGETS, "[task].target", config_path)

    inp = raw.get("input", {})
    if "gfile" in inp:
        out["gfile_path"] = _check_str_path(inp["gfile"], "[input].gfile", config_path)
    if "array_file" in inp:
        out["array_file_path"] = _check_str_path(
            inp["array_file"], "[input].array_file", config_path
        )

    dom = raw.get("domain", {})
    if "boundary_source" in dom:
        out["boundary_source"] = _check_enum(
            dom["boundary_source"], BOUNDARY_SOURCES,
            "[domain].boundary_source", config_path,
        )

    mesh = raw.get("mesh", {})
    if "mesh_dx" in mesh:
        out["mesh_dx"] = _check_positive_float(
            mesh["mesh_dx"], "[mesh].mesh_dx", config_path
        )
    if "local_refinement" in mesh:
        out["local_refinement"] = _parse_local_refinement_records(
            mesh["local_refinement"],
            "[[mesh.local_refinement]]",
            config_path,
        )

    output = raw.get("output", {})
    if "hdf5" in output:
        out["hdf5_path"] = _check_str_path(output["hdf5"], "[output].hdf5", config_path)
    if "plot" in output:
        out["plot_path"] = _check_str_path(output["plot"], "[output].plot", config_path)
    if "quality_plot" in output:
        out["quality_plot_path"] = _check_str_path(
            output["quality_plot"],
            "[output].quality_plot",
            config_path,
        )
    if "model" in output:
        out["model_path"] = _check_str_path(
            output["model"], "[output].model", config_path
        )
    if "smb" in output:
        out["smb_path"] = _check_str_path(output["smb"], "[output].smb", config_path)
    if "plot_lcfs" in output:
        out["plot_lcfs"] = _check_bool(
            output["plot_lcfs"], "[output].plot_lcfs", config_path
        )

    runtime = raw.get("runtime", {})
    if "verbose" in runtime:
        out["verbose"] = _check_bool(
            runtime["verbose"], "[runtime].verbose", config_path
        )

    return out


# --------------------------------------------------------------------------- #
# regions inspect 独立配置模型                                                 #
# --------------------------------------------------------------------------- #

def _load_regions_inspect_toml(config_path: Path) -> dict[str, Any]:
    try:
        with config_path.open("rb") as file_object:
            raw = tomllib.load(file_object)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"[配置文件 {config_path}] TOML 语法错误：{exc}"
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"[配置文件 {config_path}] 无法读取：{exc}"
        ) from exc

    if set(raw) - {"schema_version", "regions", "runtime"}:
        unknown = sorted(set(raw) - {"schema_version", "regions", "runtime"})
        _fail(config_path, "顶层", f"未知 section 或键：{unknown}")
    if raw.get("schema_version") != SCHEMA_VERSION or isinstance(
        raw.get("schema_version"), bool
    ):
        _fail(config_path, "schema_version", "必须是整数 1")

    regions = raw.get("regions", {})
    runtime = raw.get("runtime", {})
    if not isinstance(regions, dict):
        _fail(config_path, "[regions]", "必须是 TOML table")
    if not isinstance(runtime, dict):
        _fail(config_path, "[runtime]", "必须是 TOML table")

    allowed_regions = {
        "mode",
        "multi_array_file",
        "boundary_list",
        "regions_map",
        "plot",
    }
    unknown_regions = set(regions) - allowed_regions
    if unknown_regions:
        _fail(
            config_path,
            "[regions]",
            f"未知键：{sorted(unknown_regions)}",
        )
    unknown_runtime = set(runtime) - {"verbose"}
    if unknown_runtime:
        _fail(
            config_path,
            "[runtime]",
            f"未知键：{sorted(unknown_runtime)}",
        )

    values: dict[str, Any] = {}
    if "mode" in regions:
        if regions["mode"] != "inspect":
            _fail(
                config_path,
                "[regions].mode",
                "regions inspect 配置必须为字符串 \"inspect\"",
            )
        values["mode"] = "inspect"
    for key in ("multi_array_file", "boundary_list", "regions_map", "plot"):
        if key in regions:
            values[key] = _check_str_path(
                regions[key],
                f"[regions].{key}",
                config_path,
            )
    if "verbose" in runtime:
        values["verbose"] = _check_bool(
            runtime["verbose"],
            "[runtime].verbose",
            config_path,
        )
    return values


def _parse_region_dx_key(
    raw_region_id: Any,
    name: str,
    config_path: Optional[Path],
) -> int:
    if isinstance(raw_region_id, bool):
        _fail(config_path, name, "Region ID 必须是正整数")
    token = str(raw_region_id)
    if not token.isdecimal():
        _fail(config_path, name, f"Region ID 必须是正整数，实际={token!r}")
    region_id = int(token, 10)
    if region_id <= 0:
        _fail(config_path, name, f"Region ID 必须是正整数，实际={token!r}")
    return region_id


def _parse_region_dx_tokens(
    tokens: Any,
    config_path: Optional[Path],
) -> dict[int, float]:
    if tokens is UNSET or tokens is None:
        return {}
    if not isinstance(tokens, (list, tuple)):
        _fail(config_path, "--region-dx", "内部值必须是 token 序列")

    parsed: dict[int, float] = {}
    for token in tokens:
        if not isinstance(token, str) or token.count("=") != 1:
            _fail(
                config_path,
                "--region-dx",
                f"必须使用 ID=DX 格式，实际={token!r}",
            )
        raw_region_id, raw_dx = token.split("=", 1)
        region_id = _parse_region_dx_key(
            raw_region_id,
            "--region-dx",
            config_path,
        )
        if region_id in parsed:
            _fail(
                config_path,
                "--region-dx",
                f"重复提供 Region ID {region_id}",
            )
        try:
            dx_value: Any = float(raw_dx)
        except ValueError as exc:
            raise ConfigError(
                f"--region-dx：Region ID {region_id} 的 DX 必须是有限正数，"
                f"实际={raw_dx!r}"
            ) from exc
        parsed[region_id] = _check_positive_float(
            dx_value,
            f"--region-dx {region_id}",
            config_path,
        )
    return parsed


def _load_regions_generate_toml(
    config_path: Path,
) -> dict[str, Any]:
    try:
        with config_path.open("rb") as file_object:
            raw = tomllib.load(file_object)
    except tomllib.TOMLDecodeError as exc:
        raise ConfigError(
            f"[配置文件 {config_path}] TOML 语法错误：{exc}"
        ) from exc
    except OSError as exc:
        raise ConfigError(
            f"[配置文件 {config_path}] 无法读取：{exc}"
        ) from exc

    allowed_top = {"schema_version", "regions", "runtime"}
    unknown_top = set(raw) - allowed_top
    if unknown_top:
        _fail(
            config_path,
            "顶层",
            f"未知 section 或键：{sorted(unknown_top)}",
        )
    schema_version = raw.get("schema_version")
    if (
        isinstance(schema_version, bool)
        or schema_version != SCHEMA_VERSION
    ):
        _fail(config_path, "schema_version", "必须是整数 1")

    regions = raw.get("regions", {})
    runtime = raw.get("runtime", {})
    if not isinstance(regions, dict):
        _fail(config_path, "[regions]", "必须是 TOML table")
    if not isinstance(runtime, dict):
        _fail(config_path, "[runtime]", "必须是 TOML table")

    allowed_regions = {
        "mode",
        "target",
        "multi_array_file",
        "boundary_list",
        "regions_map",
        "plot",
        "quality_plot",
        "hdf5",
        "model",
        "smb",
        "mesh",
    }
    unknown_regions = set(regions) - allowed_regions
    if unknown_regions:
        _fail(
            config_path,
            "[regions]",
            f"未知键：{sorted(unknown_regions)}",
        )
    unknown_runtime = set(runtime) - {"verbose"}
    if unknown_runtime:
        _fail(
            config_path,
            "[runtime]",
            f"未知键：{sorted(unknown_runtime)}",
        )

    if regions.get("mode") != "generate":
        _fail(
            config_path,
            "[regions].mode",
            'regions generate 配置必须为字符串 "generate"',
        )

    values: dict[str, Any] = {"mode": "generate"}
    if "target" in regions:
        values["target"] = _check_enum(
            regions["target"],
            ("mesh_only", "trimeg", "m3dc1"),
            "[regions].target",
            config_path,
        )
    for key in (
        "multi_array_file",
        "boundary_list",
        "regions_map",
        "plot",
        "quality_plot",
        "hdf5",
        "model",
        "smb",
    ):
        if key in regions:
            values[key] = _check_str_path(
                regions[key],
                f"[regions].{key}",
                config_path,
            )

    mesh = regions.get("mesh", {})
    if not isinstance(mesh, dict):
        _fail(config_path, "[regions.mesh]", "必须是 TOML table")
    unknown_mesh = set(mesh) - {
        "mesh_dx",
        "region_dx",
        "transition_width",
        "transition_gradient",
        "local_refinement",
    }
    if unknown_mesh:
        _fail(
            config_path,
            "[regions.mesh]",
            f"未知键：{sorted(unknown_mesh)}",
        )
    if "mesh_dx" in mesh:
        values["mesh_dx"] = _check_positive_float(
            mesh["mesh_dx"],
            "[regions.mesh].mesh_dx",
            config_path,
        )
    if (
        "transition_width" in mesh
        and "transition_gradient" in mesh
    ):
        _fail(
            config_path,
            "[regions.mesh]",
            "transition_width 与 transition_gradient 互斥，不能同时定义",
        )
    if "transition_width" in mesh:
        values["transition_width"] = _check_nonnegative_float(
            mesh["transition_width"],
            "[regions.mesh].transition_width",
            config_path,
        )
    if "transition_gradient" in mesh:
        values["transition_gradient"] = _check_positive_float(
            mesh["transition_gradient"],
            "[regions.mesh].transition_gradient",
            config_path,
        )
    if "local_refinement" in mesh:
        values["local_refinement"] = _parse_local_refinement_records(
            mesh["local_refinement"],
            "[[regions.mesh.local_refinement]]",
            config_path,
        )

    raw_region_dx = mesh.get("region_dx", {})
    if not isinstance(raw_region_dx, dict):
        _fail(
            config_path,
            "[regions.mesh.region_dx]",
            "必须是 TOML table",
        )
    normalized_region_dx: dict[int, float] = {}
    for raw_region_id, raw_dx in raw_region_dx.items():
        region_id = _parse_region_dx_key(
            raw_region_id,
            "[regions.mesh.region_dx]",
            config_path,
        )
        if region_id in normalized_region_dx:
            _fail(
                config_path,
                "[regions.mesh.region_dx]",
                f"规范化后重复 Region ID {region_id}",
            )
        normalized_region_dx[region_id] = _check_positive_float(
            raw_dx,
            f"[regions.mesh.region_dx].{raw_region_id}",
            config_path,
        )
    values["region_dx"] = normalized_region_dx

    if "verbose" in runtime:
        values["verbose"] = _check_bool(
            runtime["verbose"],
            "[runtime].verbose",
            config_path,
        )
    return values


def load_regions_inspect_config(
    overrides: RegionsInspectOverrides,
    *,
    config_arg: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> RegionsInspectConfig:
    """Resolve isolated inspect config using defaults < TOML < explicit CLI."""
    base_cwd = Path(cwd) if cwd is not None else Path.cwd()
    config_path: Optional[Path] = None
    values: dict[str, Any] = {}

    if config_arg is not None:
        config_path = _resolve_path(config_arg, base_cwd)
        if not config_path.is_file():
            raise ConfigError(
                f"--config 指定的 regions 配置文件不存在：{config_path}"
            )
        values = _load_regions_inspect_toml(config_path)

    config_dir = config_path.parent if config_path is not None else base_cwd

    def merged(name: str, default: Any = None) -> Any:
        explicit = getattr(overrides, name)
        if explicit is not UNSET:
            return explicit
        return values.get(name, default)

    def required_path(name: str) -> Path:
        explicit = getattr(overrides, name)
        value = merged(name)
        if value is None:
            option_name = name.replace("_", "-")
            raise ConfigError(
                "regions inspect 缺少必需路径："
                f"--{option_name} 或 [regions].{name}"
            )
        base = base_cwd if explicit is not UNSET else config_dir
        return _resolve_path(
            _check_str_path(value, name, config_path),
            base,
        )

    verbose = _check_bool(
        merged("verbose", False),
        "verbose",
        config_path,
    )
    explicit_multi_array = overrides.multi_array_file is not UNSET
    explicit_boundary_list = overrides.boundary_list is not UNSET
    if explicit_multi_array and explicit_boundary_list:
        raise ConfigError(
            "regions inspect 的 multi_array_file 与 boundary_list 互斥，"
            "不能同时通过 CLI 提供"
        )

    toml_multi_array = values.get("multi_array_file")
    toml_boundary_list = values.get("boundary_list")
    if (
        not explicit_multi_array
        and not explicit_boundary_list
        and toml_multi_array is not None
        and toml_boundary_list is not None
    ):
        raise ConfigError(
            "regions inspect 的 multi_array_file 与 boundary_list 互斥，"
            "不能同时在 TOML 中定义"
        )

    if explicit_boundary_list or (
        not explicit_multi_array
        and toml_boundary_list is not None
    ):
        multi_array_file_path = None
        boundary_list_path = required_path("boundary_list")
    else:
        boundary_list_path = None
        multi_array_file_path = required_path("multi_array_file")

    regions_map_path = required_path("regions_map")
    plot_path = required_path("plot")

    resolved_paths = (
        (
            "boundary_list"
            if boundary_list_path is not None
            else "multi_array_file",
            boundary_list_path
            if boundary_list_path is not None
            else multi_array_file_path,
        ),
        ("regions_map", regions_map_path),
        ("plot", plot_path),
    )

    for index, (first_name, first_path) in enumerate(resolved_paths):
        for second_name, second_path in resolved_paths[index + 1:]:
            if first_path == second_path:
                raise ConfigError(
                    "regions inspect 路径冲突："
                    f"{first_name} 与 {second_name} 解析为同一路径 "
                    f"{first_path}"
                )

    return RegionsInspectConfig(
        schema_version=SCHEMA_VERSION,
        config_path=config_path,
        mode="inspect",
        multi_array_file_path=multi_array_file_path,
        boundary_list_path=boundary_list_path,
        regions_map_path=regions_map_path,
        plot_path=plot_path,
        verbose=verbose,
    )


def load_regions_generate_config(
    overrides: RegionsGenerateOverrides,
    *,
    config_arg: Optional[str] = None,
    cwd: Optional[Path] = None,
) -> RegionsGenerateConfig:
    """Resolve isolated generate config using defaults < TOML < CLI."""
    base_cwd = Path(cwd) if cwd is not None else Path.cwd()
    config_path: Optional[Path] = None
    values: dict[str, Any] = {}

    if config_arg is not None:
        config_path = _resolve_path(config_arg, base_cwd)
        if not config_path.is_file():
            raise ConfigError(
                f"--config 指定的 regions 配置文件不存在：{config_path}"
            )
        values = _load_regions_generate_toml(config_path)

    config_dir = config_path.parent if config_path is not None else base_cwd

    def merged(name: str, default: Any = None) -> Any:
        explicit = getattr(overrides, name)
        if explicit is not UNSET:
            return explicit
        return values.get(name, default)

    def required_path(name: str) -> Path:
        explicit = getattr(overrides, name)
        value = merged(name)
        if value is None:
            option_name = name.replace("_", "-")
            raise ConfigError(
                "regions generate 缺少必需路径："
                f"--{option_name} 或 [regions].{name}"
            )
        base = base_cwd if explicit is not UNSET else config_dir
        return _resolve_path(
            _check_str_path(value, name, config_path),
            base,
        )

    target = _check_enum(
        merged("target", "mesh_only"),
        ("mesh_only", "trimeg", "m3dc1"),
        "regions target",
        config_path,
    )

    explicit_multi_array = overrides.multi_array_file is not UNSET
    explicit_boundary_list = overrides.boundary_list is not UNSET
    if explicit_multi_array and explicit_boundary_list:
        raise ConfigError(
            "regions generate 的 multi_array_file 与 boundary_list 互斥，"
            "不能同时通过 CLI 提供"
        )

    toml_multi_array = values.get("multi_array_file")
    toml_boundary_list = values.get("boundary_list")
    if (
        not explicit_multi_array
        and not explicit_boundary_list
        and toml_multi_array is not None
        and toml_boundary_list is not None
    ):
        raise ConfigError(
            "regions generate 的 multi_array_file 与 boundary_list 互斥，"
            "不能同时在 TOML 中定义"
        )

    if explicit_boundary_list or (
        not explicit_multi_array
        and toml_boundary_list is not None
    ):
        multi_array_file_path = None
        boundary_list_path = required_path("boundary_list")
    else:
        boundary_list_path = None
        multi_array_file_path = required_path("multi_array_file")

    regions_map_path = required_path("regions_map")

    def optional_output_path(name: str) -> Optional[Path]:
        raw_value = merged(name)
        if raw_value is None:
            return None
        base = (
            base_cwd
            if getattr(overrides, name) is not UNSET
            else config_dir
        )
        return _resolve_path(
            _check_str_path(raw_value, name, config_path),
            base,
        )

    plot_path = optional_output_path("plot")
    quality_plot_path = optional_output_path("quality_plot")
    hdf5_path = optional_output_path("hdf5")
    model_path = optional_output_path("model")
    smb_path = optional_output_path("smb")

    if target == "mesh_only":
        if any(
            value is not None
            for value in (hdf5_path, model_path, smb_path)
        ):
            raise ConfigError(
                "regions generate target=mesh_only 禁止提供 hdf5、model 或 smb"
            )
    elif target == "trimeg":
        if hdf5_path is None:
            raise ConfigError(
                "regions generate target=trimeg 需要 hdf5，"
                "必须提供 --hdf5 或 [regions].hdf5"
            )
        if model_path is not None or smb_path is not None:
            raise ConfigError(
                "regions generate target=trimeg 禁止提供 model 或 smb"
            )
    else:
        if hdf5_path is not None:
            raise ConfigError(
                "regions generate target=m3dc1 禁止提供 hdf5"
            )
        if model_path is None or smb_path is None:
            raise ConfigError(
                "regions generate target=m3dc1 需要同时提供 model 和 smb"
            )

    source_name = (
        "boundary_list"
        if boundary_list_path is not None
        else "multi_array_file"
    )
    source_path = (
        boundary_list_path
        if boundary_list_path is not None
        else multi_array_file_path
    )
    resolved_paths = [
        (source_name, source_path),
        ("regions_map", regions_map_path),
    ]
    for name, path in (
        ("plot", plot_path),
        ("quality_plot", quality_plot_path),
        ("hdf5", hdf5_path),
        ("model", model_path),
        ("smb", smb_path),
    ):
        if path is not None:
            resolved_paths.append((name, path))

    for index, (first_name, first_path) in enumerate(resolved_paths):
        for second_name, second_path in resolved_paths[index + 1:]:
            if first_path == second_path:
                raise ConfigError(
                    "regions generate 路径冲突："
                    f"{first_name} 与 {second_name} 解析为同一路径 "
                    f"{first_path}"
                )

    raw_mesh_dx = merged("mesh_dx")
    mesh_dx = (
        None
        if raw_mesh_dx is None
        else _check_positive_float(
            raw_mesh_dx,
            "regions mesh_dx",
            config_path,
        )
    )

    toml_region_dx = dict(values.get("region_dx", {}))
    cli_region_dx = _parse_region_dx_tokens(
        overrides.region_dx,
        config_path,
    )
    region_dx = {**toml_region_dx, **cli_region_dx}
    if mesh_dx is None and not region_dx:
        raise ConfigError(
            "regions generate 需要 --mesh-dx，或通过 --region-dx/"
            "[regions.mesh.region_dx] 提供完整 Region 覆盖"
        )

    cli_width = overrides.transition_width
    cli_gradient = overrides.transition_gradient
    if cli_width is not UNSET and cli_gradient is not UNSET:
        raise ConfigError(
            "--transition-width 与 --transition-gradient 互斥，不能同时提供"
        )

    if cli_width is not UNSET:
        transition_width = _check_nonnegative_float(
            cli_width,
            "regions transition_width",
            config_path,
        )
        transition_gradient = None
        transition_mode = (
            "disabled" if transition_width == 0.0 else "width"
        )
    elif cli_gradient is not UNSET:
        transition_width = None
        transition_gradient = _check_positive_float(
            cli_gradient,
            "regions transition_gradient",
            config_path,
        )
        transition_mode = "gradient"
    elif "transition_gradient" in values:
        transition_width = None
        transition_gradient = values["transition_gradient"]
        transition_mode = "gradient"
    else:
        transition_width = _check_nonnegative_float(
            values.get("transition_width", DEFAULT_TRANSITION_WIDTH),
            "regions transition_width",
            config_path,
        )
        transition_gradient = None
        transition_mode = (
            "disabled" if transition_width == 0.0 else "width"
        )

    local_refinement = (
        _parse_local_refinement_tokens(
            overrides.local_refinement,
            config_path,
        )
        if overrides.local_refinement is not UNSET
        else tuple(values.get("local_refinement", ()))
    )

    verbose = _check_bool(
        merged("verbose", False),
        "verbose",
        config_path,
    )
    return RegionsGenerateConfig(
        schema_version=SCHEMA_VERSION,
        config_path=config_path,
        mode="generate",
        target=target,
        multi_array_file_path=multi_array_file_path,
        boundary_list_path=boundary_list_path,
        regions_map_path=regions_map_path,
        plot_path=plot_path,
        quality_plot_path=quality_plot_path,
        hdf5_path=hdf5_path,
        model_path=model_path,
        smb_path=smb_path,
        mesh_dx=mesh_dx,
        region_dx=MappingProxyType(dict(sorted(region_dx.items()))),
        transition_mode=transition_mode,
        transition_width=transition_width,
        transition_gradient=transition_gradient,
        local_refinement=local_refinement,
        verbose=verbose,
    )


# --------------------------------------------------------------------------- #
# 配置发现（第 6A.6 节）                                                        #
# --------------------------------------------------------------------------- #

def discover_config(
    *,
    config_arg: Optional[str],
    no_config: bool,
    cwd: Optional[Path] = None,
) -> Optional[Path]:
    """按第 6A.6 节发现配置文件，返回绝对路径或 None。

    - config_arg 与 no_config 互斥（由调用方保证；此处再次防御）；
    - config_arg 提供：只加载该路径，不存在立即失败；
    - no_config 提供：不自动发现；
    - 都未提供：只检查当前工作目录下 linkmesh.in；不递归父目录、不搜主目录。

    异常：
        ConfigError: --config 与 --no-config 同时提供；或 --config 指定文件不存在。
    """
    if config_arg is not None and no_config:
        raise ConfigError("--config 与 --no-config 互斥，不能同时提供")

    base = Path(cwd) if cwd is not None else Path.cwd()

    if config_arg is not None:
        p = Path(config_arg)
        if not p.is_absolute():
            p = base / p
        p = p.resolve()
        if not p.is_file():
            raise ConfigError(f"--config 指定的配置文件不存在：{p}")
        return p

    if no_config:
        return None

    candidate = (base / "linkmesh.in").resolve()
    if candidate.is_file():
        return candidate
    return None


# --------------------------------------------------------------------------- #
# 路径解析辅助                                                                  #
# --------------------------------------------------------------------------- #

def _resolve_path(value: str, base_dir: Path) -> Path:
    """把 value（可能相对）相对 base_dir 解析为规范化绝对路径。

    不展开 ~、不展开环境变量、不要求存在、不创建父目录。
    """
    p = Path(value)
    if not p.is_absolute():
        p = base_dir / p
    # resolve(strict=False)：规范化但不要求存在。
    return p.resolve()


# --------------------------------------------------------------------------- #
# 三层合并 + required fields + 兼容矩阵                                         #
# --------------------------------------------------------------------------- #

def load_effective_config(
    overrides: CliOverrides,
    *,
    config_arg: Optional[str] = None,
    no_config: bool = False,
    cwd: Optional[Path] = None,
) -> EffectiveConfig:
    """合并「内建默认值 < linkmesh.in < 用户显式 CLI」，返回冻结的 EffectiveConfig。

    路径解析基准：TOML 相对路径相对配置目录；CLI 相对路径相对进程当前工作目录。
    随后校验 required fields 与兼容矩阵（第 6A.9/6A.10 节）；当前阶段未实现的
    组合在此明确失败，不进入业务执行。

    参数：
        overrides: CLI 收集的用户显式输入（UNSET 表示未提供）。
        config_arg: --config 值或 None。
        no_config: --no-config 是否提供。
        cwd: 进程当前工作目录（测试可注入）；None 时用 Path.cwd()。

    返回：
        EffectiveConfig（frozen）。

    异常：
        ConfigError: 发现/读取/校验/合并/兼容矩阵任一环节失败。
        NotImplementedError: 当前阶段未实现的组合（single_region/flux_surface）。
    """
    base_cwd = Path(cwd) if cwd is not None else Path.cwd()

    # Step 1：发现配置文件。
    config_path = discover_config(
        config_arg=config_arg, no_config=no_config, cwd=base_cwd
    )

    # Step 2：加载 TOML（若有）。toml_values 使用 EffectiveConfig 字段名（长名）。
    toml_values: dict = {}
    if config_path is not None:
        toml_values = _load_toml(config_path)
        config_dir = config_path.parent
    else:
        config_dir = None

    def _explicit(cli_key: str) -> Any:
        """读取 CliOverrides 上的显式值；UNSET 归一为 None（表示未提供）。"""
        val = getattr(overrides, cli_key)
        return val if val is not UNSET else None

    # ------------------------------------------------------------------ #
    # 标量字段合并：默认 < TOML < 显式 CLI                                  #
    # 标量字段的 CLI 键名与 TOML/EffectiveConfig 键名相同，可共用一个名称。 #
    # ------------------------------------------------------------------ #
    def _merge_scalar(field: str, default: Any) -> Any:
        cli_val = _explicit(field)
        if cli_val is not None:
            return cli_val
        if field in toml_values:
            return toml_values[field]
        return default

    target = _merge_scalar("target", DEFAULT_TARGET)
    boundary_source = _merge_scalar("boundary_source", DEFAULT_BOUNDARY_SOURCE)
    mesh_dx = _merge_scalar("mesh_dx", DEFAULT_MESH_DX)
    verbose = _merge_scalar("verbose", DEFAULT_VERBOSE)
    plot_lcfs = _merge_scalar("plot_lcfs", DEFAULT_PLOT_LCFS)
    local_refinement = (
        _parse_local_refinement_tokens(
            overrides.local_refinement,
            config_path,
        )
        if overrides.local_refinement is not UNSET
        else tuple(toml_values.get("local_refinement", ()))
    )

    # 枚举/类型再校验（覆盖 CLI 直接注入的情况；TOML 侧已在 _load_toml 校验）。
    target = _check_enum(target, TARGETS, "target", config_path)
    boundary_source = _check_enum(
        boundary_source, BOUNDARY_SOURCES, "boundary_source", config_path
    )
    mesh_dx = _check_positive_float(mesh_dx, "mesh_dx", config_path)
    verbose = _check_bool(verbose, "verbose", config_path)
    plot_lcfs = _check_bool(plot_lcfs, "plot_lcfs", config_path)

    # ------------------------------------------------------------------ #
    # 路径字段合并 + 基准解析                                               #
    #   CLI 显式路径优先（相对 cwd）；否则 TOML 路径（相对配置目录）；        #
    #   否则内建默认（仅 hdf5 有默认名，且仅按需求处理）。                    #
    #                                                                      #
    # 关键：CLI 载体键名（cli_key，如 "gfile"）与 TOML/EffectiveConfig 键名  #
    # （toml_key，如 "gfile_path"）不同，必须分别查询，不能共用同一名称，    #
    # 否则 TOML 路径值会因 `cli_key in toml_values` 恒为 False 而被丢弃。    #
    # ------------------------------------------------------------------ #
    def _merge_path(cli_key: str, toml_key: str,
                    default_name: Optional[str]) -> Optional[Path]:
        cli_val = _explicit(cli_key)
        if cli_val is not None:
            return _resolve_path(str(cli_val), base_cwd)
        if toml_key in toml_values and config_dir is not None:
            return _resolve_path(str(toml_values[toml_key]), config_dir)
        if default_name is not None:
            # 内建默认路径相对当前工作目录。
            return _resolve_path(default_name, base_cwd)
        return None

    # gfile/array_file/plot/model/smb 无内建默认；hdf5 默认名仅在 target=trimeg
    # 时作为默认输出，其它 target 不制造默认可执行路径。
    gfile_path = _merge_path("gfile", "gfile_path", None)
    array_file_path = _merge_path("array_file", "array_file_path", None)
    plot_path = _merge_path("plot", "plot_path", None)
    quality_plot_path = _merge_path(
        "quality_plot",
        "quality_plot_path",
        None,
    )
    model_path = _merge_path("model", "model_path", None)
    smb_path = _merge_path("smb", "smb_path", None)

    hdf5_default_name = DEFAULT_HDF5_NAME if target == "trimeg" else None
    hdf5_path = _merge_path("hdf5", "hdf5_path", hdf5_default_name)

    cfg = EffectiveConfig(
        schema_version=SCHEMA_VERSION,
        config_path=config_path,
        target=target,
        boundary_source=boundary_source,
        gfile_path=gfile_path,
        array_file_path=array_file_path,
        mesh_dx=mesh_dx,
        hdf5_path=hdf5_path,
        plot_path=plot_path,
        quality_plot_path=quality_plot_path,
        model_path=model_path,
        smb_path=smb_path,
        verbose=verbose,
        plot_lcfs=plot_lcfs,
        local_refinement=local_refinement,
    )

    if (
        cfg.plot_path is not None
        and cfg.quality_plot_path is not None
        and cfg.plot_path == cfg.quality_plot_path
    ):
        raise ConfigError(
            "根命令路径冲突：plot 与 quality_plot 解析为同一路径 "
            f"{cfg.plot_path}"
        )

    # required-field 校验发生在路径合并完成之后。
    _validate_compatibility(cfg)
    _validate_required_fields(cfg)
    _validate_current_stage(cfg)

    return cfg


# --------------------------------------------------------------------------- #
# 根命令组合校验                                                                #
# --------------------------------------------------------------------------- #

def _combo(cfg: EffectiveConfig) -> str:
    return f"target={cfg.target}, boundary_source={cfg.boundary_source}"


def _validate_compatibility(cfg: EffectiveConfig) -> None:
    """Validate root-command combinations without a profile dimension."""
    if cfg.plot_lcfs and cfg.boundary_source != "array_file":
        raise ConfigError(
            f"禁止的组合（{_combo(cfg)}）：plot_lcfs 只适用于 "
            "boundary_source=array_file"
        )


# --------------------------------------------------------------------------- #
# required fields（第 6A.10 节）                                               #
# --------------------------------------------------------------------------- #

def _validate_required_fields(cfg: EffectiveConfig) -> None:
    """校验 required fields（第 6A.10 节）。

    仅校验字段存在/冲突关系；具体文件内容有效性（如第一壁是否有效）留待
    对应输入层实现阶段。array_file 内容读取本轮不实现。
    """
    b = cfg.boundary_source

    if b in ("lcfs", "first_wall"):
        if cfg.gfile_path is None:
            raise ConfigError(
                f"boundary_source={b} 需要 gfile_path（--gfile 或 [input].gfile）"
            )
        if cfg.array_file_path is not None:
            raise ConfigError(
                f"boundary_source={b} 禁止同时提供 array_file_path"
            )
    elif b == "array_file":
        if cfg.array_file_path is None:
            raise ConfigError(
                "boundary_source=array_file 需要 array_file_path"
                "（--array-file 或 [input].array_file）"
            )
        if cfg.plot_lcfs:
            if cfg.gfile_path is None:
                raise ConfigError(
                    "boundary_source=array_file 且 plot_lcfs=True 时需要 "
                    "gfile_path（--gfile 或 [input].gfile）"
                )
            if cfg.plot_path is None:
                raise ConfigError(
                    "boundary_source=array_file 且 plot_lcfs=True 时需要 "
                    "plot_path（--plot 或 [output].plot）"
                )
        elif cfg.gfile_path is not None:
            raise ConfigError(
                "boundary_source=array_file 仅在 plot_lcfs=True 时允许提供 "
                "gfile_path"
            )
    # flux_surface 的 required 判定由 _validate_current_stage 统一报未实现。

    # target=trimeg 需要 hdf5_path（默认已提供 mesh.h5）。
    if cfg.target == "trimeg" and cfg.hdf5_path is None:
        raise ConfigError("target=trimeg 需要 hdf5_path")

    # M5 第一条 M3DC1 CLI 闭环需要配套 TXT model 与 SMB 基础输出路径。
    if cfg.target == "m3dc1":
        if cfg.model_path is None:
            raise ConfigError(
                "target=m3dc1 需要 model_path（--model 或 [output].model）"
            )
        if cfg.smb_path is None:
            raise ConfigError(
                "target=m3dc1 需要 smb_path（--smb 或 [output].smb）"
            )


# --------------------------------------------------------------------------- #
# 当前阶段实现边界（输入层 I + 输入层 II）                                       #
# --------------------------------------------------------------------------- #

# 当前已实现的根命令单区域来源：lcfs 与 array_file。
# first_wall/flux_surface 仍明确未实现。
_IMPLEMENTED_M3DC1_SOURCES = ("lcfs", "array_file")
_IMPLEMENTED_SINGLE_REGION_SOURCES = ("lcfs", "array_file")


def _validate_current_stage(cfg: EffectiveConfig) -> None:
    """Validate the currently implemented root single-region sources."""
    if cfg.boundary_source == "flux_surface":
        raise NotImplementedError(
            f"boundary_source=flux_surface 尚未实现（{_combo(cfg)}）："
            "真实磁通面提取未实现，禁止径向缩放 LCFS 伪造磁面"
        )

    if (
        cfg.target == "m3dc1"
        and cfg.boundary_source not in _IMPLEMENTED_M3DC1_SOURCES
    ):
        raise NotImplementedError(
            "target=m3dc1 当前仅支持 lcfs/array_file 单区域来源"
            f"（{_combo(cfg)}）"
        )

    if cfg.boundary_source not in _IMPLEMENTED_SINGLE_REGION_SOURCES:
        raise NotImplementedError(
            f"boundary_source={cfg.boundary_source} 尚未实现（{_combo(cfg)}）："
            "first_wall 当前阶段未实现，不回退其他边界来源"
        )
