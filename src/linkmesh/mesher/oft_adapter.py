"""
OFT mesh generation adapter.

职责：
    - mesh_from_oft_domain(domain) -> Mesh2D
          通用 N 区域核心：接收已构造的 OFT gs_Domain，调用一次 build_mesh()，
          将 pts/lc/reg 规范化并验证为 Mesh2D。对任意正区域数量 N 通用（含 N=1）。
    - build_single_region_domain_from_lcfs(gfile_data, ...) -> gs_Domain
          （v2.3 输入层 II）LCFS 单区域来源策略层：以 LCFS 为唯一计算域边界，
          用一个 add_polygon() Region 构造 OFT gs_Domain，不创建第一壁/vacuum/
          人工 outer rectangle，不调用 build_mesh()。single-region 网格尺度用 mesh_dx。
    - generate_mesh(boundary, mesh_dx=...) -> Mesh2D
          根命令便利入口：只构造一个 polygon Region。

正式数据流（single_region + lcfs，v2.3 输入层 II）：
    GFileData(LCFS) → build_single_region_domain_from_lcfs → OFT gs_Domain
             → mesh_from_oft_domain/build_mesh → pts/lc/reg → Mesh2D

outer boundary 数据链：
    1. 策略层通过 add_polygon() 创建唯一 outer Region；
    2. 策略层在创建前后用 domain.regions 长度差异捕获新增的 Region 对象引用
       （不写死整数 region ID、不按名称猜测、不遍历私有 map），作为 LinkMesh
       私有属性 domain._linkmesh_outer_region 附加在返回的 domain 上（该属性
       不是 OFT 字段，OFT 不消费它）；捕获或保存失败即停，不静默 pass；
    3. mesh_from_oft_domain 在 build_mesh() 之前把该 Region 引用保存到局部变量；
       调用 domain.build_mesh()；OFT Mesh.__init__() 内部完成实际重采样、共享
       边界处理、点合并、全局重编号与 segment 方向处理，并把该 Region 的有序
       离散点列回写到 outer_region._resampled_points；
    4. build_mesh() 完成后，受控读取 build 前保存的同一个 Region 对象的
       _resampled_points（Triangle 输入前的真实有序离散点列，与最终 vert 同源
       同批），删除可能的显式闭合重复末点、规范化为顺时针（保持起点）、创建
       float64 独立副本；即使 build_mesh() 期间 domain 上的私有属性被替换，也使用
       build 前保存的原对象；
    5. 由 Mesh2D 统一验证完整几何契约，注入 Mesh2D.outer_boundary。

    该私有依赖是版本锁定（openfusiontoolkit==26.6）、范围受控、失败即停止的兼容
    接口，不是通用稳定公共 API；只服务于获取 outer_boundary。

区域 ID 时序（重要）：
    OFT gs_Domain.build_mesh() 会根据区域类型和定义顺序重新排列区域 ID，
    并原地更新 domain.region_info[name]["id"]。因此 Adapter 必须在 build_mesh()
    成功返回之后再读取 domain.region_info，得到 OFT 最终区域 ID 集合，
    并以该最终集合校验输出 reg。绝不能在 build_mesh() 之前提取区域 ID 集合。

显式闭合末点容差（v2.3 LCFS 单区域来源真实回归修复）：
    OFT 26.6 回写的 outer Region _resampled_points 在真实 DIII-D GFile 下可能包含
    与首点在 Mesh2D 正式坐标尺度容差内重复的显式闭合末点。删除该末点的容差必须与
    Mesh2D._validate_outer_boundary() 的正式坐标尺度语义一致——
    length_scale = max(1.0, R 跨度, Z 跨度, 坐标绝对量级)，
    coordinate_tol = 1e-12 * length_scale（长度量纲），
    绝不使用 NumPy allclose/array_equal 的隐式默认容差，绝不用 Adapter 私有
    的、比 Mesh2D 更窄或语义不同的尺度公式。只删除一个末点；若删除后首末点仍在
    该容差内重复，不继续循环删除，交由 Mesh2D 拒绝。

禁止事项：
    - 不复制 OFT 的区域拓扑、点合并、重采样、区域种子或 Triangle 调用逻辑；
    - 不修改 OFT 源码/安装目录/fork；
    - 不把 OFT 源码快照加入 sys.path/PYTHONPATH；
    - 不在通用核心中判断 plasma/vacuum/outer 语义，不写死区域数量/名称/类型/整数 ID；
    - 不重编号、不排序、不压缩、不执行 +1；
    - 不解析 domain.mesh._reg_seg_map / _resampled_segments / Mesh._resampled_points；
    - 不调用 Region.get_resampled_points() 作为 outer_boundary 来源；
    - 不用 build_mesh() 后的 pts/lc、最终 idxevert 或原始 LCFS 控制点反推 outer boundary；
    - single_region + lcfs 不径向缩放 LCFS，不使用第一壁，不创建人工外框/annulus；
    - 不引入 Protocol、抽象基类、插件系统、通用区域配置类或额外模块。

Mesh2D 兼容说明：
    Mesh2D 从 linkmesh.mesh 导入（唯一正式定义）。其构造函数当前允许核心三数组
    与可选兼容字段；OFT Adapter 显式传入核心三数组，并对
    boundary_segments/boundary_markers/node_markers 传入空数组以维持与旧构造方式一致。
    空数组只是兼容占位，不表示已生成任何边界语义。

"""

from __future__ import annotations

from typing import TYPE_CHECKING, Any, Optional

import numpy as np

from linkmesh.mesh import Mesh2D

if TYPE_CHECKING:
    from linkmesh.readers.gfile_reader import GFileData


# LinkMesh 私有属性名：附加在 domain 上的、策略层捕获的 outer Region 对象引用。
# 该属性不是 OFT gs_Domain 的字段，OFT 不读取、不消费它。
# 注意：它保存的是 outer Region 对象引用（供 build_mesh() 后读取回写点列），
# 绝不保存人工矩形四角数组或原始 LCFS 控制点。
_OUTER_REGION_ATTR = "_linkmesh_outer_region"

# 版本锁定兼容接口绑定的 OFT 版本（见蓝图第 3.2A 节）。
_REQUIRED_OFT_VERSION = "26.6"

# OFT 26.6 Mesh.__init__()/gs_Domain.build_mesh() 的锁定默认点合并阈值。
# 仅用于 single-region polygon 输入的相邻近重复控制点规范化，不向公共 API 暴露。
_SINGLE_REGION_POLYGON_MERGE_THRESH = 1e-4


# --------------------------------------------------------------------------- #
# OFT 延迟导入                                                                  #
# --------------------------------------------------------------------------- #

def _load_gs_domain_class() -> Any:
    """延迟导入 OFT gs_Domain 类。

    在函数内部执行导入，避免模块顶层静态导入完整 OFT meshing 模块时
    触发 Windows 平台检查或动态库加载失败。

    正式 OFT 类来自 Linux Conda 环境安装的：
        OpenFUSIONToolkit.TokaMaker.meshing.gs_Domain

    不把 OFT 源码目录加入 sys.path 或 PYTHONPATH。

    返回：
        gs_Domain 类对象

    异常：
        ImportError: OFT 未安装、不在 PYTHONPATH 中，或共享库加载失败。
                     保留原始异常链（__cause__）。
    """
    try:
        from OpenFUSIONToolkit.TokaMaker.meshing import gs_Domain
    except Exception as e:
        raise ImportError(
            "无法导入 OFT gs_Domain。正式运行环境为 Linux Conda 环境（linkmesh），"
            "需通过 pip 安装正式的 openfusiontoolkit==26.6，且 liboft_triangle.so 可加载。"
            "不要将 OFT 源码目录加入 PYTHONPATH。"
            f"\n原始错误：{e}"
        ) from e
    return gs_Domain


def _get_installed_oft_version() -> Optional[str]:
    """读取当前环境安装的 openfusiontoolkit 版本字符串。

    使用仓库现有依赖（标准库 importlib.metadata）读取，不引入第三方依赖。
    读取失败时返回 None（由调用方转化为明确失败）。

    返回：
        版本字符串（如 "26.6"），或 None（无法取得）
    """
    try:
        from importlib.metadata import version as _pkg_version

        return str(_pkg_version("openfusiontoolkit"))
    except Exception:
        return None


def _check_oft_version_locked() -> None:
    """集中、可测试的 OFT 版本兼容检查（版本锁定兼容接口前置条件）。

    该检查在受控读取 outer Region 私有属性 _resampled_points 之前执行：
        - 版本无法取得 → 明确失败；
        - 版本不等于 26.6 → 明确失败。
    错误信息包含 openfusiontoolkit、检测值与要求版本。

    版本判断只在本函数集中实现，不在多个函数中重复。

    异常：
        ValueError: 版本无法取得或不等于 26.6
    """
    detected = _get_installed_oft_version()
    if detected is None:
        raise ValueError(
            "无法读取 openfusiontoolkit 版本：outer boundary 私有兼容接口要求"
            f" openfusiontoolkit=={_REQUIRED_OFT_VERSION}（版本锁定），"
            "检测值=None（未取得）。"
        )
    if detected != _REQUIRED_OFT_VERSION:
        raise ValueError(
            "openfusiontoolkit 版本与 outer boundary 私有兼容接口锁定版本不一致："
            f"检测值={detected}，要求版本={_REQUIRED_OFT_VERSION}。"
            "该私有依赖为版本锁定兼容接口，升级前必须重新核对 OFT 内部数据流。"
        )


# --------------------------------------------------------------------------- #
# 内部数值校验辅助                                                              #
# --------------------------------------------------------------------------- #

def _as_finite_float_2d(arr: Any, name: str) -> np.ndarray:
    """将输入转换为有限的二维 float64 数组，非法时抛 ValueError。"""
    a = np.asarray(arr, dtype=np.float64)
    if a.ndim != 2:
        raise ValueError(f"{name} 必须是二维数组，实际 ndim={a.ndim}")
    if not np.all(np.isfinite(a)):
        raise ValueError(f"{name} 含非有限值（NaN/Inf）")
    return a


def _validate_integer_semantics(arr_float: np.ndarray, name: str) -> None:
    """验证浮点数组具有整数语义（有限且等于其四舍五入值）。

    在 astype(np.int32) 之前调用，防止 astype 静默截断 1.5、NaN 或 Inf。
    """
    if not np.all(np.isfinite(arr_float)):
        raise ValueError(f"{name} 含非有限值（NaN/Inf），不具备整数语义")
    if not np.all(arr_float == np.round(arr_float)):
        raise ValueError(f"{name} 含非整数值，不具备整数语义")


def _signed_area_closed(boundary: np.ndarray) -> float:
    """对隐式闭合（末尾不重复首点）的轮廓计算带符号闭合面积（鞋带公式）。

    正值为逆时针，负值为顺时针。
    """
    r = boundary[:, 0]
    z = boundary[:, 1]
    r_next = np.roll(r, -1)
    z_next = np.roll(z, -1)
    return 0.5 * float(np.sum(r * z_next - r_next * z))


def _mesh2d_coordinate_tol(pts: np.ndarray) -> float:
    """按 Mesh2D._validate_outer_boundary() 的正式坐标尺度语义计算坐标容差。

    与 Mesh2D 统一验证保持一致的尺度语义（长度量纲）：
        length_scale   = max(1.0, R 跨度, Z 跨度, 坐标绝对量级)
        coordinate_tol = 1e-12 * length_scale

    该函数只提供 Adapter 删除显式闭合重复末点所需的坐标容差，使其与 Mesh2D
    的首末点/相邻点判据处于同一尺度语义，避免 Adapter 使用比 Mesh2D 更窄或
    语义不同的私有容差（不使用 NumPy allclose/array_equal 隐式默认容差）。

    参数：
        pts: (B,2) float64 点列（至少 1 个点）

    返回：
        坐标容差（float，长度量纲）
    """
    r_span = float(pts[:, 0].max() - pts[:, 0].min())
    z_span = float(pts[:, 1].max() - pts[:, 1].min())
    abs_mag = float(np.max(np.abs(pts)))
    length_scale = max(1.0, r_span, z_span, abs_mag)
    if not (np.isfinite(length_scale) and length_scale > 0):
        length_scale = 1.0
    return 1e-12 * length_scale


# --------------------------------------------------------------------------- #
# outer Region 回写点列的读取与规范化（版本锁定兼容接口，见蓝图第 3.2A 节）      #
# --------------------------------------------------------------------------- #

def _normalize_resampled_points(raw: Any) -> np.ndarray:
    """把 outer Region 回写的 _resampled_points 规范化为顺时针 (B,2) float64 副本。

    仅做：
        - 转 (B,2) float64 独立副本；
        - 删除可能存在的显式闭合重复末点（末点在 Mesh2D 正式坐标尺度容差内
          ≈首点时只删一个）；
        - 用带符号闭合面积判断方向，逆时针则反转为顺时针，且保持原点 0 为输出点 0：
              顺时针  → 保持全部点序不变；
              逆时针  → 输出为 [p0, p_(B-1), p_(B-2), ..., p1]
                        （不使用 pts[::-1]，因为那会把起点改成最后一点）。
    不做任何新的边界插值/重采样，不循环移位、不排序、不删除内部点、不删除相邻
    重复点、不静默修补自交/退化/重复点（完整几何契约交由 Mesh2D 统一验证）。

    显式闭合末点判据（v2.3 真实回归修复）：
        使用 _mesh2d_coordinate_tol()（length_scale = max(1.0, R 跨度, Z 跨度,
        坐标绝对量级)，coordinate_tol = 1e-12 * length_scale），与 Mesh2D 正式
        坐标尺度语义一致。逐位完全相等的首末点其欧氏距离为 0，天然落在该容差内，
        亦被删除。只删除一个末点；删除后至少保留 3 个点；若删除一个末点后首末点
        仍在该容差内重复，不继续循环删除，交由 Mesh2D 拒绝异常点列。

    参数：
        raw: outer_region._resampled_points（OFT 回写点列）

    返回：
        (B,2) float64 顺时针独立副本

    异常：
        ValueError: None、shape 非法、空、NaN/Inf、删末点后点数不足、方向不可判定
    """
    if raw is None:
        raise ValueError(
            "outer_region._resampled_points 为 None：OFT 未回写有序离散边界点列"
        )

    pts = np.array(raw, dtype=np.float64, copy=True)
    if pts.ndim != 2:
        raise ValueError(
            f"outer_region._resampled_points 必须是二维数组，实际 ndim={pts.ndim}"
        )
    if pts.shape[1] != 2:
        raise ValueError(
            f"outer_region._resampled_points 必须为 (B,2)，实际 shape={pts.shape}"
        )
    if pts.shape[0] == 0:
        raise ValueError("outer_region._resampled_points 为空数组")
    if not np.all(np.isfinite(pts)):
        raise ValueError("outer_region._resampled_points 含非有限值（NaN/Inf）")

    # 删除可能存在的显式闭合重复末点（末点与首点在 Mesh2D 正式坐标尺度容差内
    # 重复时只删一个）。容差语义与 Mesh2D._validate_outer_boundary() 一致，
    # 逐位完全相等（距离 0）亦被覆盖。只删一个末点，不循环删除。
    if pts.shape[0] >= 2:
        coordinate_tol = _mesh2d_coordinate_tol(pts)
        gap = float(np.hypot(pts[0, 0] - pts[-1, 0], pts[0, 1] - pts[-1, 1]))
        if gap <= coordinate_tol:
            pts = pts[:-1]

    if pts.shape[0] < 3:
        raise ValueError(
            "outer_region._resampled_points 删除显式闭合重复末点后不足 3 个点，"
            f"实际={pts.shape[0]}"
        )

    # 方向规范化为顺时针（带符号闭合面积应为负）。
    #   - 退化方向（面积为 0）留给 Mesh2D 统一验证明确拒绝，不在此静默处理；
    #   - 逆时针（面积>0）反转为顺时针，且保持原点 0 为输出点 0：
    #         输出 = [p0, p_(B-1), p_(B-2), ..., p1]
    #     等价实现：先整体反向 pts[::-1] 得到 [p_(B-1), ..., p0]，
    #     再 np.roll(..., 1) 把 p0 移回索引 0，得到 [p0, p_(B-1), ..., p1]。
    #     不使用裸 pts[::-1]（会改变起点），不排序、不循环移位到固定起点。
    area = _signed_area_closed(pts)
    if area == 0.0:
        return pts
    if area > 0.0:
        pts = np.roll(pts[::-1], 1, axis=0).copy()
    return pts


def _extract_outer_boundary_from_region(
    outer_region: Optional[Any],
) -> Optional[np.ndarray]:
    """build_mesh() 之后，从 build 前保存的 outer Region 读取回写点列并规范化。

    该 helper 收紧为接收 outer Region 对象本身（而非整个 domain），确保它无法在
    build 后重新从 domain 定位或猜测 Region；调用方必须在 build_mesh() 之前保存
    该对象引用并在 build 之后原样传入。

    仅当 outer_region 非 None 时才尝试读取；否则返回 None（通用 domain 不携带
    outer Region 引用时 outer_boundary 保持 None）。

    读取路径严格限定为 outer_region._resampled_points（版本锁定兼容接口，
    见蓝图第 3.2A 节）：
        - 不读取 domain.mesh._reg_seg_map；
        - 不读取 domain.mesh._resampled_segments；
        - 不读取 domain.mesh._resampled_points；
        - 不调用 outer_region.get_resampled_points()；
        - 不从最终 pts/lc/idxevert 反推；
        - 不使用人工矩形四角或原始 LCFS 控制点。

    在读取私有属性前执行集中的 OFT 版本锁定检查。

    参数：
        outer_region: build 前保存的 outer Region 对象引用，或 None

    返回：
        (B,2) float64 顺时针独立副本，或 None（未捕获 outer Region）

    异常：
        ValueError: 版本不符、缺 _resampled_points、None、shape/有限性/点数非法、
                    方向不可判定等（失败即停，禁止回退四角/idxevert/自行重采样）
    """
    if outer_region is None:
        # 通用 domain（非策略层构造）：不猜测 outer Region，outer_boundary 保持 None。
        return None

    # 受控读取私有属性前，先做版本锁定检查（集中实现）。
    _check_oft_version_locked()

    if not hasattr(outer_region, "_resampled_points"):
        raise ValueError(
            "捕获的 outer Region 缺少 _resampled_points 属性："
            "OFT 26.6 应在 build_mesh() 内回写该属性，缺失即失败，"
            "禁止回退四角/idxevert/自行重采样"
        )

    raw = getattr(outer_region, "_resampled_points")
    return _normalize_resampled_points(raw)


# --------------------------------------------------------------------------- #
# outer Region 引用捕获                                                         #
# --------------------------------------------------------------------------- #

def _capture_and_attach_outer_region(domain: Any, n_before: int) -> Any:
    """在 domain.regions 恰好新增一个 Region 后，捕获该 Region 并附加到 domain。

    要求策略层在创建 outer/唯一 Region 之前先记录 len(domain.regions) 为 n_before，
    创建之后调用本 helper 完成：
        - 校验 domain.regions 存在且是可计数容器；
        - 校验恰好新增一个 Region（len == n_before + 1）；
        - 取 regions[-1] 作为捕获的 Region 对象引用（不写死整数 region ID、
          不按名称猜测、不遍历私有 map）；
        - 通过 setattr 附加为 LinkMesh 私有属性 _linkmesh_outer_region；
        - 捕获或保存失败即停，不静默 pass，保留原始异常链。

    参数：
        domain:   OFT gs_Domain 对象（策略层已创建 Region）
        n_before: 创建 Region 之前的 len(domain.regions)

    返回：
        捕获的 Region 对象引用

    异常：
        ValueError:   regions 缺失/非计数容器，或未恰好新增一个 Region
        RuntimeError: 无法把私有属性附加到 domain（保留原始异常链）
    """
    regions_after = getattr(domain, "regions", None)
    if regions_after is None or not hasattr(regions_after, "__len__"):
        raise ValueError(
            "创建 Region 后 domain.regions 不存在或不是预期容器，无法捕获 outer Region"
        )
    n_after = len(regions_after)
    if n_after != n_before + 1:
        raise ValueError(
            "创建 Region 后 domain.regions 未恰好新增一个 Region："
            f"新增前={n_before}，新增后={n_after}；无法可靠捕获 outer Region 引用，"
            "失败即停，不回退四角/idxevert/自行重采样"
        )
    outer_region = regions_after[-1]

    # 将捕获的 outer Region 对象引用作为 LinkMesh 私有属性附加在 domain 上。
    # OFT 不消费该属性；仅供 mesh_from_oft_domain 在 build_mesh() 之后读取回写点列。
    # 保存失败即停：绝不静默 pass，绝不回退四角/idxevert/自行重采样。
    try:
        setattr(domain, _OUTER_REGION_ATTR, outer_region)
    except (AttributeError, TypeError) as e:
        raise RuntimeError(
            "无法将捕获的 outer Region 引用保存到 domain."
            f"{_OUTER_REGION_ATTR}："
            "domain 不允许附加该私有属性，无法建立 outer boundary 数据链，"
            "失败即停，禁止回退四角/idxevert/自行重采样。"
            f"\n原始错误：{e}"
        ) from e

    return outer_region


# --------------------------------------------------------------------------- #
# multi-array inspect：只执行 OFT setup_only 编号                              #
# --------------------------------------------------------------------------- #

def _multi_region_name(curve_id: int) -> str:
    """Return the shared deterministic OFT Region name for one input curve."""
    return f"LM_CURVE_{curve_id}"


def _validate_multi_region_plan(plan: Any, *, require_ids: bool) -> tuple[Any, ...]:
    entries = tuple(plan.entries)
    if len(entries) < 2:
        raise ValueError("multi-region plan 至少需要两个 Region")

    curve_ids = [entry.curve_id for entry in entries]
    if (
        any(isinstance(curve_id, bool) or curve_id <= 0 for curve_id in curve_ids)
        or len(set(curve_ids)) != len(curve_ids)
    ):
        raise ValueError("Region plan 含非正或重复 curve_id")

    by_curve_id = {entry.curve_id: entry for entry in entries}
    roots = [
        entry.curve_id
        for entry in entries
        if entry.parent_curve_id == 0
    ]
    if len(roots) != 1 or roots[0] != plan.root_curve_id:
        raise ValueError("Region plan 必须恰好包含一个一致的 root")
    known_curve_ids = set(by_curve_id)
    if any(
        entry.parent_curve_id not in known_curve_ids | {0}
        for entry in entries
    ):
        raise ValueError("Region plan 引用了未知 parent_curve_id")

    role_counts = {
        role: sum(entry.technical_role == role for entry in entries)
        for role in ("boundary", "plasma", "vacuum")
    }
    if role_counts["boundary"] != 1 or role_counts["plasma"] != 1:
        raise ValueError(
            "Region plan 必须恰好包含一个 boundary 和一个 synthetic plasma "
            "technical role"
        )
    if any(
        entry.technical_role not in ("boundary", "plasma", "vacuum")
        for entry in entries
    ):
        raise ValueError("Region plan 含未批准的 OFT technical role")

    if require_ids:
        region_ids: list[int] = []
        for entry in entries:
            if entry.region_id is None:
                raise ValueError("Region plan 缺少 OFT 最终 Region ID")
            region_id = _coerce_region_id(entry.region_id)
            if region_id <= 0:
                raise ValueError("Region plan 的 OFT 最终 Region ID 必须为正")
            region_ids.append(region_id)
        if len(set(region_ids)) != len(region_ids):
            raise ValueError("Region plan 含重复 OFT 最终 Region ID")

    return entries


def _add_multi_region_polygons(
    domain: Any,
    entries: tuple[Any, ...],
    root_curve_id: int,
) -> None:
    """Add polygons in curve order and capture the unique root Region."""
    for entry in sorted(entries, key=lambda item: item.curve_id):
        parent_name = (
            None
            if entry.parent_curve_id == 0
            else _multi_region_name(entry.parent_curve_id)
        )
        regions = getattr(domain, "regions", None)
        if regions is None or not hasattr(regions, "__len__"):
            raise ValueError(
                "domain.regions 不存在或不是预期容器，无法构造 multi-region Domain"
            )
        n_before = len(regions)
        domain.add_polygon(
            np.array(entry.points, dtype=np.float64, copy=True),
            _multi_region_name(entry.curve_id),
            parent_name=parent_name,
        )
        if entry.curve_id == root_curve_id:
            _capture_and_attach_outer_region(domain, n_before)


def inspect_multi_region_ids(plan: Any) -> dict[int, int]:
    """Build a fresh OFT Domain and return final IDs from setup_only.

    This function never requests Triangle nodes, cells, or attributes.
    Technical roles are consumed only as OFT adapter compatibility metadata.
    """
    entries = _validate_multi_region_plan(plan, require_ids=False)
    by_curve_id = {entry.curve_id: entry for entry in entries}

    gs_domain = _load_gs_domain_class()
    domain = gs_domain()

    # setup-only 编号与 dx 数值无关；第一批使用固定正值，不开放网格尺度配置。
    setup_dx = 1.0
    for entry in sorted(entries, key=lambda item: item.curve_id):
        domain.define_region(
            _multi_region_name(entry.curve_id),
            setup_dx,
            entry.technical_role,
        )

    _add_multi_region_polygons(
        domain,
        entries,
        plan.root_curve_id,
    )

    result = domain.build_mesh(setup_only=True)
    if result != (None, None, None):
        raise ValueError(
            "OFT build_mesh(setup_only=True) 返回了非预期结果；"
            "inspect 禁止执行完整 Triangle 网格生成"
        )

    raw_info = getattr(domain, "region_info", None)
    if not hasattr(raw_info, "items"):
        raise ValueError("OFT setup_only 后缺少可读取的 region_info")

    info_by_name = {
        str(name).upper(): info
        for name, info in raw_info.items()
    }
    mapping: dict[int, int] = {}
    for entry in entries:
        name = _multi_region_name(entry.curve_id).upper()
        if name not in info_by_name:
            raise ValueError(
                f"OFT setup_only 后缺少 curve_id={entry.curve_id} 对应 Region"
            )
        mapping[entry.curve_id] = _coerce_region_id(info_by_name[name])

    if len(set(mapping.values())) != len(mapping):
        raise ValueError("OFT setup_only 返回重复最终 Region ID")
    if set(mapping) != set(by_curve_id):
        raise ValueError("OFT setup_only 的 curve/Region 对应不完整")
    return mapping


def generate_multi_region_mesh(
    plan: Any,
    dx_by_region_id: dict[int, float],
) -> Mesh2D:
    """Build a fresh full OFT Domain from a numbered plan and resolved dx."""
    entries = _validate_multi_region_plan(plan, require_ids=True)
    expected_ids = {
        int(entry.region_id)
        for entry in entries
    }
    if set(dx_by_region_id) != expected_ids:
        raise ValueError(
            "完整 Region dx 映射必须与编号 plan 的最终 Region ID 集合一致"
        )

    normalized_dx: dict[int, float] = {}
    for region_id, raw_dx in dx_by_region_id.items():
        if isinstance(raw_dx, bool):
            raise ValueError(f"Region ID {region_id} 的 dx 必须是有限正数")
        dx = float(raw_dx)
        if not (np.isfinite(dx) and dx > 0.0):
            raise ValueError(f"Region ID {region_id} 的 dx 必须是有限正数")
        normalized_dx[int(region_id)] = dx

    gs_domain = _load_gs_domain_class()
    domain = gs_domain()
    for entry in sorted(entries, key=lambda item: item.curve_id):
        domain.define_region(
            _multi_region_name(entry.curve_id),
            normalized_dx[int(entry.region_id)],
            entry.technical_role,
        )

    _add_multi_region_polygons(
        domain,
        entries,
        plan.root_curve_id,
    )
    mesh = mesh_from_oft_domain(domain)

    raw_info = getattr(domain, "region_info", None)
    if not hasattr(raw_info, "items"):
        raise ValueError("OFT 完整构建后缺少可读取的 region_info")
    info_by_name = {
        str(name).upper(): info
        for name, info in raw_info.items()
    }
    actual_mapping: dict[int, int] = {}
    for entry in entries:
        name = _multi_region_name(entry.curve_id).upper()
        if name not in info_by_name:
            raise ValueError(
                f"OFT 完整构建后缺少 curve_id={entry.curve_id} 对应 Region"
            )
        actual_mapping[entry.curve_id] = _coerce_region_id(
            info_by_name[name]
        )

    expected_mapping = {
        entry.curve_id: int(entry.region_id)
        for entry in entries
    }
    if actual_mapping != expected_mapping:
        raise ValueError(
            "OFT 完整构建后的 curve_id -> Region ID 与 setup-only/map "
            f"不一致：期望={expected_mapping}，实际={actual_mapping}"
        )
    if set(int(value) for value in mesh.region_ids.tolist()) != expected_ids:
        raise ValueError("Mesh2D.region_ids 与编号 plan 的 Region ID 集合不一致")
    return mesh


# --------------------------------------------------------------------------- #
# 1. 通用 N 区域核心                                                            #
# --------------------------------------------------------------------------- #

def mesh_from_oft_domain(domain: Any) -> Mesh2D:
    """通用 N 区域核心：将已构造 OFT gs_Domain 转换为轻量 Mesh2D。

    仅负责：
        1. 在调用前验证 domain 提供可调用的 build_mesh() 和可读取的 region_info 属性；
        2. 在 build_mesh() 前把策略层附加的 outer Region 引用保存到局部变量（若有）；
        3. 调用一次 domain.build_mesh()；
        4. 接收 pts, lc, reg；
        5. build_mesh() 成功后重新读取 domain.region_info，
           提取 OFT 最终区域 ID 集合（build_mesh 会原地重排区域 ID）；
        6. build_mesh() 之后，从 build 前保存的同一个 outer Region 对象受控读取
           回写点列 outer_region._resampled_points，规范化为顺时针 (B,2) float64
           独立副本（保持起点）；
        7. 规范化并验证输出，用最终区域 ID 集合校验 reg；
        8. 构造 Mesh2D（由 Mesh2D 统一验证 outer_boundary 完整几何契约）。

    对任意正区域数量 N 通用，包括 N=1、N=2、N=3、N>3，无特殊分支。
    不判断 LCFS/第一壁/plasma/vacuum/outer，不写死区域名称/类型/数量/整数 ID。
    不重编号、不排序、不压缩、不执行 +1。

    outer boundary 时序（关键，v2.2）：
        outer boundary 来自策略层捕获的 outer Region 对象在 build_mesh() 之后被
        OFT 回写的 _resampled_points（Triangle 输入前、Steiner 插点之前的有序离散
        点列，与最终 vert 同源同批）。读取发生在 build_mesh() 之后（不得在之前）。
        必须使用 build 前保存的局部引用，即使 build_mesh() 期间 domain 上的私有属性
        被替换，也不得在 build 后重新从 domain 定位或猜测 Region。
        当 domain 未携带该 outer Region 引用（例如受控测试直接构造 Domain，或通用
        N 区域 domain）时，Mesh2D.outer_boundary 保持 None（未提供），核心不猜测
        outer Region、不读取 OFT 私有 map、不从最终 pts/lc/idxevert 反推。

    区域 ID 时序（关键）：
        OFT build_mesh() 会根据区域类型和定义顺序重排区域 ID，并原地更新
        domain.region_info[name]["id"]。因此最终区域 ID 集合必须在 build_mesh()
        成功返回之后从 domain.region_info 读取，绝不能在 build_mesh() 之前提取。

    参数：
        domain: 已由调用方构造完成的 OFT gs_Domain 对象

    返回：
        Mesh2D 对象（核心三数组 nodes/triangles/region_ids，兼容空边界数组，
        可选 outer_boundary）

    异常：
        ValueError: 输出不满足契约（形状、dtype、索引范围、退化、区域集合、
                    outer boundary 私有接口/版本/点列等），
                    或 build_mesh() 后 region_info 为空/非法/含重复或非正 ID
        TypeError:  domain 缺少 build_mesh() 或 region_info
    """
    # --- Step 1：调用前只验证 domain 接口存在（不在此处提取最终区域 ID）---
    build_mesh = getattr(domain, "build_mesh", None)
    if not callable(build_mesh):
        raise TypeError("domain 必须提供可调用的 build_mesh() 方法")

    # 仅验证 region_info 属性存在且可读取；不在 build_mesh() 前提取最终 ID，
    # 因为 build_mesh() 会原地重排区域 ID。
    if not hasattr(domain, "region_info"):
        raise TypeError("domain 必须提供可读取的 region_info 属性")

    # --- Step 2：build_mesh() 前把 outer Region 引用保存到局部变量（若策略层附加）---
    # 此处只取并保存引用，不读取 _resampled_points；点列必须在 build_mesh() 之后读取，
    # 因为 OFT 在 Mesh.__init__() 期间才回写 region._resampled_points。
    # 保存到局部变量后，即使 build_mesh() 期间 domain 上的该属性被替换，也使用原对象。
    saved_outer_region = getattr(domain, _OUTER_REGION_ATTR, None)

    # --- Step 3：调用且只调用一次 build_mesh() ---
    result = build_mesh()
    try:
        pts, lc, reg = result
    except (TypeError, ValueError) as e:
        raise ValueError(
            f"build_mesh() 应返回 (pts, lc, reg) 三元组，实际无法解包：{e}"
        ) from e

    # --- Step 5：build_mesh() 成功后重新读取 region_info，提取 OFT 最终区域 ID 集合 ---
    region_info = getattr(domain, "region_info", None)
    if region_info is None:
        raise TypeError("build_mesh() 后 domain.region_info 为 None，无法读取最终区域 ID")

    # 从生成后的 region_info 读取最终区域 ID 集合（不推断物理语义，仅取整数 ID）
    expected_ids = _extract_region_ids(region_info)
    if len(expected_ids) == 0:
        raise ValueError(
            "build_mesh() 后 domain.region_info 为空，未定义任何区域"
        )

    # --- Step 6：build_mesh() 之后从 build 前保存的同一个 outer Region 读取回写点列 ---
    # 失败即停（版本不符/缺属性/None/非法），禁止回退四角/idxevert/自行重采样。
    # 使用 build 前保存的 saved_outer_region，不在 build 后重新从 domain 定位。
    outer_boundary = _extract_outer_boundary_from_region(saved_outer_region)

    # --- Step 7：规范化并验证 nodes ---
    nodes = _as_finite_float_2d(pts, "pts")
    if nodes.shape[0] == 0:
        raise ValueError("build_mesh() 返回节点数为 0")
    if nodes.shape[1] != 2:
        raise ValueError(f"pts 必须为 (N,2)，实际 shape={nodes.shape}")
    n_nodes = nodes.shape[0]

    # 验证并转换 triangles（lc）
    lc_float = np.asarray(lc, dtype=np.float64)
    if lc_float.ndim != 2 or lc_float.shape[1] != 3:
        raise ValueError(f"lc 必须为 (T,3)，实际 shape={lc_float.shape}")
    if lc_float.shape[0] == 0:
        raise ValueError("build_mesh() 返回三角形数为 0")
    _validate_integer_semantics(lc_float, "lc")
    triangles = lc_float.astype(np.int32)
    n_triangles = triangles.shape[0]

    # 索引范围校验（0-based，M2-0 确认）
    if triangles.min() < 0 or triangles.max() >= n_nodes:
        raise ValueError(
            f"三角形节点索引越界：应满足 0 <= index < {n_nodes}，"
            f"实际范围 [{int(triangles.min())}, {int(triangles.max())}]"
        )

    # 退化检查：三个顶点索引互不相同
    for i in range(n_triangles):
        a, b, c = int(triangles[i, 0]), int(triangles[i, 1]), int(triangles[i, 2])
        if a == b or b == c or a == c:
            raise ValueError(f"三角形 {i} 存在重复顶点索引：{[a, b, c]}")

    # 退化检查：绝对面积严格大于 1e-12
    for i in range(n_triangles):
        p0 = nodes[triangles[i, 0]]
        p1 = nodes[triangles[i, 1]]
        p2 = nodes[triangles[i, 2]]
        area = abs(
            (p1[0] - p0[0]) * (p2[1] - p0[1])
            - (p2[0] - p0[0]) * (p1[1] - p0[1])
        ) / 2.0
        if not (area > 1e-12):
            raise ValueError(f"三角形 {i} 面积过小或退化：area={area}")

    # 验证并转换 region_ids（reg）
    reg_float = np.asarray(reg, dtype=np.float64).reshape(-1)
    _validate_integer_semantics(reg_float, "reg")
    region_ids = reg_float.astype(np.int32).reshape(-1)

    if len(region_ids) != n_triangles:
        raise ValueError(
            f"reg 长度（{len(region_ids)}）与三角形数（{n_triangles}）不一致"
        )

    # 所有区域编号必须为正
    if region_ids.min() <= 0:
        raise ValueError(
            f"region_ids 含非正编号（0/负数）：最小值={int(region_ids.min())}；"
            "非正区域编号表示未认领或非法区域，不是合法标签"
        )

    # --- 区域集合一致性校验（与 build_mesh 后的 OFT 最终区域 ID 集合完全一致）---
    actual_ids = set(int(x) for x in region_ids.tolist())
    if actual_ids != expected_ids:
        missing = expected_ids - actual_ids
        extra = actual_ids - expected_ids
        raise ValueError(
            "输出区域 ID 集合与 build_mesh 后的 OFT 最终 region_info 不一致："
            f"缺失={sorted(missing)}，多余={sorted(extra)}；"
            f"期望（最终）={sorted(expected_ids)}，实际={sorted(actual_ids)}"
        )

    # 每个已定义区域至少对应一个三角形（actual_ids == expected_ids 已隐含，
    # 但显式检查以覆盖 region_info 提供 ID 而输出缺该 ID 的情况）
    for rid in expected_ids:
        if rid not in actual_ids:
            raise ValueError(f"区域 ID {rid} 未对应任何三角形")

    # --- Step 8：构造 Mesh2D（兼容旧六字段构造签名，附加 outer_boundary）---
    mesh = Mesh2D(
        nodes=nodes,
        triangles=triangles,
        region_ids=region_ids,
        boundary_segments=np.empty((0, 2), dtype=np.int32),
        boundary_markers=np.empty((0,), dtype=np.int32),
        node_markers=np.empty((0,), dtype=np.int32),
        outer_boundary=outer_boundary,
    )
    return mesh


def _extract_region_ids(region_info: Any) -> set:
    """从 domain.region_info 提取最终区域整数 ID 集合。

    注意：本函数应在 build_mesh() 成功返回之后调用，读取的是 OFT 重排后的
    最终区域 ID。只读取整数 ID，不推断物理语义。region_info 结构由 OFT 决定；
    这里做宽松适配：支持形如 {name: {'id': int, ...}} 或 {name: id} 的映射，
    以及提供 .id 属性的区域对象。

    参数：
        region_info: OFT domain.region_info（build_mesh 后的最终状态）

    返回：
        区域 ID 的 set（int）

    异常：
        ValueError: 无法从 region_info 提取合法整数 ID，或出现 ID 重复
    """
    ids: list = []

    if hasattr(region_info, "items"):
        for _name, info in region_info.items():
            ids.append(_coerce_region_id(info))
    elif isinstance(region_info, (list, tuple)):
        for info in region_info:
            ids.append(_coerce_region_id(info))
    else:
        raise ValueError(
            "无法解析 domain.region_info：既非映射也非序列"
        )

    if len(ids) == 0:
        return set()

    id_set = set(ids)
    if len(id_set) != len(ids):
        raise ValueError(f"domain.region_info 存在重复区域 ID：{sorted(ids)}")

    for rid in id_set:
        if rid <= 0:
            raise ValueError(
                f"domain.region_info 含非正区域 ID：{rid}"
            )
    return id_set


def _coerce_region_id(info: Any) -> int:
    """从单个区域记录中提取整数 ID（宽松适配）。"""
    candidate: Any = None

    if isinstance(info, dict):
        if "id" in info:
            candidate = info["id"]
        elif "reg_id" in info:
            candidate = info["reg_id"]
        else:
            raise ValueError(f"区域记录缺少 id 字段：{info}")
    elif hasattr(info, "id"):
        candidate = getattr(info, "id")
    elif isinstance(info, (int, np.integer)):
        candidate = info
    elif isinstance(info, float):
        candidate = info
    else:
        raise ValueError(f"无法从区域记录提取 ID：{info!r}")

    cf = float(candidate)
    if not np.isfinite(cf) or cf != round(cf):
        raise ValueError(f"区域 ID 不是合法整数：{candidate!r}")
    return int(cf)


# --------------------------------------------------------------------------- #
# 已退役的 nested_three_region 实现从此处删除                                   #
# --------------------------------------------------------------------------- #

def _retired_build_oft_domain_from_gfile(
    gfile_data: "GFileData",
    *,
    plasma_dx: float = 0.08,
    vacuum_dx: float = 0.10,
    outer_dx: float = 0.15,
    outer_padding: float = 0.12,
) -> Any:
    """当前 GFile 默认区域策略层：构造 OFT gs_Domain（不调用 build_mesh）。

    默认策略：
        - 有有效第一壁：OUTER(boundary) + PLASMA(plasma) + VACUUM(vacuum)，
          使用 add_annulus(lcfs, "plasma", first_wall, "vacuum", parent_name="outer")；
        - 无第一壁：OUTER(boundary) + PLASMA(plasma)，
          使用 add_polygon(lcfs, "plasma", parent_name="outer")。
        两种策略均创建人工 outer rectangle。

    本函数不调用 build_mesh()，不转换 pts/lc/reg，不复制 OFT 的重采样、
    点合并、种子搜索或 Triangle 算法。

    outer Region 捕获（v2.2，见蓝图第 3.2A 节）：
        人工 outer 矩形由 add_rectangle(name="outer") 创建（形状/尺寸/名称/
        outer_dx/区域策略均不变）。策略层在 add_rectangle 前后用 domain.regions
        长度差异捕获新增的 outer Region 对象引用（不写死整数 region ID、不按名称
        猜测、不遍历私有 map），作为 LinkMesh 私有属性附加在返回 domain 上。
        该引用供 mesh_from_oft_domain 在 build_mesh() 之后受控读取
        outer_region._resampled_points，得到 Triangle 输入前的真实有序离散边界。
        本函数不读取任何 _resampled_points（build_mesh() 之前 OFT 尚未回写），
        不使用人工矩形四角作为 outer_boundary。有壁/无壁共用同一捕获逻辑。
        捕获或保存引用失败即停，绝不静默 pass，绝不回退四角/idxevert/自行重采样。

    参数：
        gfile_data:    GFileData 对象（含 lcfs_r/lcfs_z，可选 wall_r/wall_z）
        plasma_dx:     plasma 区域网格尺度（有限正数）
        vacuum_dx:     vacuum 区域网格尺度（有限正数）
        outer_dx:      outer 区域网格尺度（有限正数）
        outer_padding: 人工 outer 矩形四侧扩展比例（有限非负数）

    返回：
        已构造但尚未 build_mesh() 的 OFT gs_Domain 对象，
        并附加 LinkMesh 私有属性（捕获的 outer Region 对象引用）

    异常：
        ValueError: 参数或轮廓非法，或 outer Region 捕获失败（regions 缺失/
                    非预期容器/未恰好新增一个 Region）
        RuntimeError: 无法将 outer Region 引用保存到 domain（保留原始异常链）
        ImportError: OFT 不可用
    """
    # --- 参数校验 ---
    for dxname, dxval in (("plasma_dx", plasma_dx), ("vacuum_dx", vacuum_dx), ("outer_dx", outer_dx)):
        if not (np.isfinite(dxval) and dxval > 0):
            raise ValueError(f"{dxname} 必须是有限正数，实际={dxval}")
    if not (np.isfinite(outer_padding) and outer_padding >= 0):
        raise ValueError(f"outer_padding 必须是有限非负数，实际={outer_padding}")

    # --- LCFS 轮廓校验与薄规范化 ---
    lcfs = _prepare_contour(
        getattr(gfile_data, "lcfs_r", None),
        getattr(gfile_data, "lcfs_z", None),
        name="LCFS",
    )

    # --- 第一壁校验：同时存在或同时缺失 ---
    wall_r = getattr(gfile_data, "wall_r", None)
    wall_z = getattr(gfile_data, "wall_z", None)
    has_wall_r = wall_r is not None and len(np.asarray(wall_r)) > 0
    has_wall_z = wall_z is not None and len(np.asarray(wall_z)) > 0

    if has_wall_r != has_wall_z:
        raise ValueError(
            "第一壁 R/Z 必须同时存在或同时缺失："
            f"wall_r 存在={has_wall_r}，wall_z 存在={has_wall_z}"
        )

    first_wall = None
    if has_wall_r:
        # 第一壁 R/Z 长度不一致、点数不足或含非有限值时，
        # _prepare_contour 在 _load_gs_domain_class() 之前抛出 ValueError。
        first_wall = _prepare_contour(wall_r, wall_z, name="第一壁")

    # --- 计算人工 outer 矩形（LCFS 与可用第一壁的联合包围范围）---
    all_r = [lcfs[:, 0]]
    all_z = [lcfs[:, 1]]
    if first_wall is not None:
        all_r.append(first_wall[:, 0])
        all_z.append(first_wall[:, 1])
    cat_r = np.concatenate(all_r)
    cat_z = np.concatenate(all_z)

    r_min, r_max = float(cat_r.min()), float(cat_r.max())
    z_min, z_max = float(cat_z.min()), float(cat_z.max())
    dr = r_max - r_min
    dz = z_max - z_min
    if not (np.isfinite(dr) and np.isfinite(dz) and dr > 0 and dz > 0):
        raise ValueError(
            f"轮廓包围范围退化：R 跨度={dr}，Z 跨度={dz}"
        )

    pad = outer_padding * max(dr, dz)
    rect_rmin = r_min - pad
    rect_rmax = r_max + pad
    rect_zmin = z_min - pad
    rect_zmax = z_max + pad

    rect_rc = 0.5 * (rect_rmin + rect_rmax)
    rect_zc = 0.5 * (rect_zmin + rect_zmax)
    rect_w = rect_rmax - rect_rmin
    rect_h = rect_zmax - rect_zmin
    if not (np.isfinite(rect_w) and np.isfinite(rect_h) and rect_w > 0 and rect_h > 0):
        raise ValueError(
            f"人工 outer 矩形尺寸非法：宽={rect_w}，高={rect_h}"
        )

    # --- 构造 OFT Domain（延迟导入）---
    gs_Domain = _load_gs_domain_class()
    domain = gs_Domain()

    domain.define_region("outer", outer_dx, "boundary")
    domain.define_region("plasma", plasma_dx, "plasma")

    if first_wall is not None:
        # 有壁三区域：PLASMA + VACUUM + OUTER
        domain.define_region("vacuum", vacuum_dx, "vacuum")
        domain.add_annulus(
            lcfs, "plasma", first_wall, "vacuum", parent_name="outer"
        )
    else:
        # 无壁两区域：PLASMA + OUTER
        domain.add_polygon(lcfs, "plasma", parent_name="outer")

    # --- 捕获 outer Region 对象引用（add_rectangle 前后 domain.regions 长度差异）---
    # 有壁/无壁使用同一捕获逻辑；不写死整数 region ID、不按名称猜测、不遍历私有 map。
    regions = getattr(domain, "regions", None)
    if regions is None or not hasattr(regions, "__len__"):
        raise ValueError(
            "domain.regions 不存在或不是预期容器：无法捕获 outer Region 引用，"
            "失败即停，不回退四角/idxevert/自行重采样"
        )
    n_before = len(regions)

    # 人工 outer rectangle（形状/尺寸/名称/outer_dx 不变）
    domain.add_rectangle(rect_rc, rect_zc, rect_w, rect_h, name="outer")

    # 捕获并附加 outer Region 引用（集中 helper，失败即停并保留异常链）
    _capture_and_attach_outer_region(domain, n_before)

    return domain


# --------------------------------------------------------------------------- #
# single-region polygon 输入规范化（OFT 26.6 版本锁定行为）                    #
# --------------------------------------------------------------------------- #

def _normalize_single_region_polygon(contour_in: Any) -> np.ndarray:
    """稳定删除会被 OFT 26.6 默认 merge_thresh 合并的相邻控制点。

    输入应为已经过基础轮廓处理的 (B,2) 有限 float64 点列。本函数仍执行
    防御性 shape、有限性和最低点数检查，并始终生成独立 float64 副本。

    从点 0 开始稳定遍历：保留第一个点；候选点与最后一个已接受点的欧氏距离
    严格小于 1e-4 m 时删除候选点，否则保留。算法只处理相邻近重复点，不排序、
    不循环移位、不插值、不重采样、不修改保留坐标，也不执行全局去重或几何修补。

    过滤后首末点若相距小于 1e-4 m，则明确失败。首末显式闭合属于基础轮廓处理
    的职责，不在此处继续静默删除，以免改变隐式闭合拓扑语义。
    """
    contour = np.array(contour_in, dtype=np.float64, copy=True)
    if contour.ndim != 2 or contour.shape[1] != 2:
        raise ValueError(
            "single-region polygon 必须为 (B,2)，"
            f"实际 shape={contour.shape}"
        )
    if contour.shape[0] < 3:
        raise ValueError(
            "single-region polygon 至少需要 3 个点，"
            f"实际={contour.shape[0]}"
        )
    if not np.all(np.isfinite(contour)):
        raise ValueError("single-region polygon 含非有限值（NaN/Inf）")

    accepted = [contour[0].copy()]
    for candidate in contour[1:]:
        distance = float(np.linalg.norm(candidate - accepted[-1]))
        if distance < _SINGLE_REGION_POLYGON_MERGE_THRESH:
            continue
        accepted.append(candidate.copy())

    normalized = np.asarray(accepted, dtype=np.float64)
    if normalized.shape[0] < 3:
        raise ValueError(
            "single-region polygon 删除相邻近重复控制点后不足 3 个点，"
            f"实际={normalized.shape[0]}"
        )

    closing_distance = float(np.linalg.norm(normalized[-1] - normalized[0]))
    if closing_distance < _SINGLE_REGION_POLYGON_MERGE_THRESH:
        raise ValueError(
            "single-region polygon 过滤后首末点距离小于 OFT 26.6 默认 "
            f"merge_thresh={_SINGLE_REGION_POLYGON_MERGE_THRESH} m；"
            "基础轮廓处理后仍存在近重复闭合点，拒绝静默修补"
        )

    return np.array(normalized, dtype=np.float64, copy=True)


# --------------------------------------------------------------------------- #
# 2c. 统一 single-region polygon 策略层                                        #
# --------------------------------------------------------------------------- #

def build_single_region_domain_from_boundary(
    boundary: Any,
    *,
    mesh_dx: float = 0.08,
) -> Any:
    """Use one validated polygon as the only OFT plasma region boundary.

    The input polygon is normalized only for OFT 26.6's locked merge
    threshold. The authoritative Mesh2D.outer_boundary is still obtained from
    the captured Region._resampled_points after build_mesh().
    """
    if not (np.isfinite(mesh_dx) and mesh_dx > 0):
        raise ValueError(f"mesh_dx 必须是有限正数，实际={mesh_dx}")

    polygon = np.array(boundary, dtype=np.float64, copy=True)
    if polygon.ndim != 2 or polygon.shape[1] != 2:
        raise ValueError(
            f"single-region boundary 必须为 (B,2)，实际 shape={polygon.shape}"
        )
    if polygon.shape[0] < 3:
        raise ValueError(
            f"single-region boundary 至少需要 3 个点，实际={polygon.shape[0]}"
        )
    if not np.all(np.isfinite(polygon)):
        raise ValueError("single-region boundary 含非有限值（NaN/Inf）")

    polygon = _normalize_single_region_polygon(polygon)

    gs_Domain = _load_gs_domain_class()
    domain = gs_Domain()
    domain.define_region("plasma", mesh_dx, "plasma")

    regions = getattr(domain, "regions", None)
    if regions is None or not hasattr(regions, "__len__"):
        raise ValueError(
            "domain.regions 不存在或不是预期容器：无法捕获 single-region Region 引用，"
            "失败即停，不回退控制点/idxevert/自行重采样"
        )
    n_before = len(regions)

    domain.add_polygon(polygon, "plasma")
    _capture_and_attach_outer_region(domain, n_before)
    return domain


# --------------------------------------------------------------------------- #
# 2d. LCFS 单区域来源策略层（v2.3 输入层 II）                                   #
# --------------------------------------------------------------------------- #

def build_single_region_domain_from_lcfs(
    gfile_data: "GFileData",
    *,
    mesh_dx: float = 0.08,
) -> Any:
    """LCFS 单区域来源策略层：以 LCFS 为唯一计算域边界构造 OFT gs_Domain。

    正式语义（蓝图第 6A.11 节，第 1 项）：
        - 从 GFileData 取 LCFS（lcfs_r/lcfs_z）；
        - LCFS 是唯一计算域边界，也是唯一区域边界；
        - 进入 add_polygon() 前，按 OFT 26.6 默认 merge_thresh=1e-4 m
          稳定删除相邻近重复控制点的后一个点；
        - 使用一个 add_polygon() Region；
        - single-region 网格尺度使用 mesh_dx；
        - 不使用第一壁；
        - 不创建 vacuum 区域；
        - 不创建人工 outer rectangle；
        - 不创建 annulus；
        - 不径向缩放 LCFS，不使用输入控制点冒充实际离散边界。

    本函数不调用 build_mesh()，不转换 pts/lc/reg，不复制 OFT 的重采样/点合并/
    种子搜索/Triangle 算法。

    outer Region 捕获（复用与 nested 相同的集中 helper，见蓝图第 3.2A 节）：
        唯一 Region 由 add_polygon() 创建。策略层在 add_polygon 前后用
        domain.regions 长度差异捕获新增的 Region 对象引用（不写死整数 region ID、
        不按名称猜测、不遍历私有 map），作为 LinkMesh 私有属性附加在返回 domain 上。
        该引用供 mesh_from_oft_domain 在 build_mesh() 之后受控读取
        _resampled_points，得到 Triangle 输入前的真实有序离散边界。
        本函数不读取任何 _resampled_points（build_mesh() 之前 OFT 尚未回写），
        不使用原始 LCFS 控制点作为 outer_boundary。
        捕获或保存引用失败即停，绝不静默 pass，绝不回退控制点/idxevert/自行重采样。

    参数：
        gfile_data: GFileData 对象（含 lcfs_r/lcfs_z）
        mesh_dx:  single-region 网格尺度（有限正数）

    返回：
        已构造但尚未 build_mesh() 的 OFT gs_Domain 对象，
        并附加 LinkMesh 私有属性（捕获的唯一 Region 对象引用）

    异常：
        ValueError: 参数或 LCFS 轮廓非法，或 Region 捕获失败（regions 缺失/
                    非预期容器/未恰好新增一个 Region）
        RuntimeError: 无法将 Region 引用保存到 domain（保留原始异常链）
        ImportError: OFT 不可用
    """
    # --- 参数校验 ---
    if not (np.isfinite(mesh_dx) and mesh_dx > 0):
        raise ValueError(f"mesh_dx 必须是有限正数，实际={mesh_dx}")

    # --- LCFS 基础校验与薄规范化（长度一致、点数>=3、有限、删一个闭合末点）---
    lcfs = _prepare_contour(
        getattr(gfile_data, "lcfs_r", None),
        getattr(gfile_data, "lcfs_z", None),
        name="LCFS",
    )

    return build_single_region_domain_from_boundary(
        lcfs,
        mesh_dx=mesh_dx,
    )


def _prepare_contour(r: Any, z: Any, *, name: str) -> np.ndarray:
    """轮廓薄规范化：转 (N,2) float64，删除首尾近似重复点，验证最低点数。

    不自行重采样、平滑尖角、自交修补、点在多边形判断或区域种子搜索。

    参数：
        r:    R 坐标数组
        z:    Z 坐标数组
        name: 轮廓名称（用于错误信息）

    返回：
        (N,2) float64 轮廓数组

    异常：
        ValueError: R/Z 缺失、长度不一致、有效点少于 3、或含非有限值（NaN/Inf）
    """
    if r is None or z is None:
        raise ValueError(f"{name} 的 R/Z 坐标必须同时提供")

    ra = np.asarray(r, dtype=np.float64).reshape(-1)
    za = np.asarray(z, dtype=np.float64).reshape(-1)

    if len(ra) != len(za):
        raise ValueError(
            f"{name} 的 R/Z 长度不一致：len(R)={len(ra)}，len(Z)={len(za)}"
        )
    if len(ra) < 3:
        raise ValueError(f"{name} 至少需要 3 个点，实际={len(ra)}")
    if not (np.all(np.isfinite(ra)) and np.all(np.isfinite(za))):
        raise ValueError(f"{name} 含非有限值（NaN/Inf）")

    contour = np.column_stack([ra, za]).astype(np.float64)

    # 尺度相关的窄容差判断首尾近似重复；只删除最后一个重复点
    dr = float(contour[:, 0].max() - contour[:, 0].min())
    dz = float(contour[:, 1].max() - contour[:, 1].min())
    scale = max(dr, dz)
    if not (np.isfinite(scale) and scale > 0):
        raise ValueError(f"{name} 包围范围退化：R 跨度={dr}，Z 跨度={dz}")

    tol = 1e-9 * scale
    gap = float(np.hypot(contour[0, 0] - contour[-1, 0], contour[0, 1] - contour[-1, 1]))
    if gap <= tol:
        contour = contour[:-1]

    if len(contour) < 3:
        raise ValueError(f"{name} 删除重复闭合点后不足 3 个点，实际={len(contour)}")

    return contour


# --------------------------------------------------------------------------- #
# 3. CLI 便利入口                                                              #
# --------------------------------------------------------------------------- #

def generate_mesh(
    boundary: Any,
    *,
    mesh_dx: float = 0.08,
) -> Mesh2D:
    """Generate one single-region Mesh2D from the unique actual boundary."""
    domain = build_single_region_domain_from_boundary(
        boundary,
        mesh_dx=mesh_dx,
    )
    return mesh_from_oft_domain(domain)
