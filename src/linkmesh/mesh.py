"""
mesh.py —— LinkMesh 唯一正式内存网格对象模块。

职责：
    - 定义轻量统一内存网格对象 Mesh2D，作为 LinkMesh 各目标格式之间的
      统一表示（OFT build_mesh 输出经 Adapter 适配后的统一对象，也是各
      Writer 的共同输入）。

设计约束（v1.11 M2.5）：
    - Mesh2D 是 LinkMesh 唯一正式内存网格对象；
    - 它与具体网格生成后端（OFT / Python Triangle 等）和目标文件格式
      （TRIMEG HDF5 / M3DC1 .smb）均无关；
    - 内部三角形索引保持 0-based；任何针对目标格式的偏置（如 TRIMEG HDF5
      的 +1）由目标 Writer 在写出阶段自行完成；
    - region_ids 原样保存 OFT 最终区域编号，不重编号、不排序、不压缩、不 +1。

outer boundary 支持：
    - 新增正式字段 outer_boundary（有序、顺时针、隐式闭合的最外层计算域
      几何边界），供 TRIMEG HDF5 Writer 写出 nbcRZ/Rbc/Zbc。
    - outer_boundary 与现有兼容字段 boundary_segments/boundary_markers/
      node_markers 语义不同、名称不同，互不替代。
    - outer_boundary 为可选字段，默认 None（表示"未提供"）；仅当提供时
      在 __post_init__ 中执行集中式验证。空/None 不是合法几何状态，
      不得被当作合法 outer boundary 写入 TRIMEG HDF5。

outer boundary 验证：
    - outer_boundary 集中验证补齐完整非相邻边自交检查（单一简单闭环，
      禁止非相邻边严格交叉、端点接触、共线重叠）；相邻边（含首末相邻边）
      共享端点不误判为自交。

ownership 与容差处理：
    - outer_boundary 始终独立持有 float64 副本（即使输入已是 float64 ndarray，
      也创建独立副本，不复用输入数组、不修改输入）；
    - 修正严格线段交叉判据：严格交叉要求两组 orientation 符号分别真正相反
      （o1*o2 < 0 且 o3*o4 < 0），orientation 为 0 交给共线/on-segment 分支；
    - 区分坐标容差（coordinate_tol，长度量纲）与 orientation 叉积容差
      （orientation_tol，长度平方量纲），由轮廓尺度与坐标绝对量级共同确定。

模块依赖边界（铁律）：
    本模块不得依赖 OFT、Python Triangle、h5py、PUMI、CLI、Plotter、Reader
    或任何 Writer，也不得引入任何网格生成后端。仅依赖 numpy 与标准库。

"""

from __future__ import annotations

from dataclasses import dataclass, field
from typing import Optional

import numpy as np


def _empty_segments() -> np.ndarray:
    """构造空的边界线段数组，shape=(0,2)、dtype=int32。

    每次调用返回独立数组，避免不同 Mesh2D 实例共享同一可变默认数组。
    """
    return np.empty((0, 2), dtype=np.int32)


def _empty_markers() -> np.ndarray:
    """构造空的一维标记数组，shape=(0,)、dtype=int32。

    每次调用返回独立数组，避免不同 Mesh2D 实例共享同一可变默认数组。
    """
    return np.empty((0,), dtype=np.int32)


@dataclass
class Mesh2D:
    """LinkMesh 轻量统一内存网格对象（唯一正式定义，v1.11 迁入独立模块）。

    与网格后端无关的统一表示。该对象是 OFT build_mesh() 返回结果经 LinkMesh
    最小适配后的统一表示，同时也是 writers/hdf5_writer.py 和 writers/smb_writer.py
    的共同输入。本模块不依赖 OFT / Python Triangle / HDF5 / PUMI / CLI / Plotter。

    M2 正式核心字段来源（OFT build_mesh() 返回值）：
        nodes      ← pts[np, 2]，OFT 返回的节点坐标，期望 dtype float64
        triangles  ← lc[nc, 3]，OFT 返回的三角形连接表，期望整数 dtype，内部 0-based
        region_ids ← reg[nc]，OFT 返回的三角形区域编号，期望整数 dtype

    索引约定（M2-0 / M2 已确认）：
        OFT 26.6 build_mesh() 返回合法 0-based lc 连接表，索引范围 0..N-1。
        Adapter 应验证 0 <= lc < N，无需无依据地再次偏置。
        任何针对目标格式的偏置（如 TRIMEG HDF5 的 +1）由目标 Writer 在写出阶段执行。

    区域编号约定（v1.8，v1.11 保持）：
        region_ids 原样保存 OFT 返回的三角形区域编号，不重编号、不压缩、不排序、不执行 +1。
        LinkMesh、HDF5 Writer 和未来 SMB Writer 不得根据整数编号猜测区域物理语义。
        region_ids 不写入 TRIMEG HDF5（v1.11 契约），继续保留在内存对象中供
        Plotter 与 M4 使用。

    兼容字段说明（v1.8 降级；v1.11 提供安全空数组默认值）：
        boundary_segments/boundary_markers/node_markers 不是 TRIMEG HDF5
        TRIMEG HDF5 的直接消费数据集，也不是核心三数组，且不参与 HDF5
        输出契约。保留字段定义以维持兼容性。
        使用 default_factory 避免不同实例共享同一可变 NumPy 默认数组。

    outer_boundary（正式字段）：
        含义   ← 传入网格生成器的、有序、最外层计算域几何边界点，
                 一条简单闭合轮廓，用于写出 TRIMEG HDF5 的 nbcRZ/Rbc/Zbc。
        shape  ← (B, 2)，第 0 列 R，第 1 列 Z，单位 m
        dtype  ← float64
        闭合   ← 末尾不重复首点，最后一点与第一点隐式闭合
        方向   ← 顺时针（带符号闭合面积严格为负）
        约束   ← B>=3，坐标全有限，至少 3 个互异点，无相邻重复点，
                 首尾不同（闭合边非零），面积非退化，单一简单闭环
                 （禁止非相邻边自交），P0 仅一条闭合轮廓
        所有权 ← 构造期始终规范化为 float64 独立副本，不复用输入数组、不修改输入
        非语义 ← 不包含 OFT/Triangle 后续插入的边界 Steiner 点；
                 不等同于最终拓扑外边界节点集合 idxevert；
                 B 可小于最终 nvert_ex；不是按节点编号升序的坐标表。
        与兼容字段区别 ← 与 boundary_segments/boundary_markers/node_markers
                 语义不同、名称不同、互不替代。
        默认   ← None（表示"未提供"，不是合法几何）。仅当提供（非 None）时
                 在 __post_init__ 中执行集中验证并规范化为 float64 独立副本。
                 空数组/None 不得被写入 TRIMEG HDF5。

    本数据类除对 outer_boundary（若提供）执行集中验证并独立复制外，不在构造时
    复制、重排、转换或校验核心/兼容输入数组；这些边界校验分别由 OFT Adapter 和
    各 Writer 承担。
    """

    # M2 正式核心字段（来自 OFT build_mesh() 返回值）
    nodes: np.ndarray                # shape=(N, 2)，节点 [R, Z] 坐标 [m]，dtype=float64
    triangles: np.ndarray            # shape=(T, 3)，三角形三顶点节点索引，整数 dtype，内部 0-based
    region_ids: np.ndarray           # shape=(T,)，每个三角形的区域 ID，整数 dtype，原样保留

    # 非核心、可选、兼容字段（v1.11 提供安全空数组默认值）
    # 使用 default_factory 避免共享可变 NumPy 默认数组。
    boundary_segments: np.ndarray = field(default_factory=_empty_segments)  # shape=(0,2) int32
    boundary_markers: np.ndarray = field(default_factory=_empty_markers)    # shape=(0,) int32
    node_markers: np.ndarray = field(default_factory=_empty_markers)        # shape=(0,) int32

    # v2.0 新增正式字段：有序最外层计算域边界几何（可选，默认 None）
    outer_boundary: Optional[np.ndarray] = None                             # shape=(B,2) float64 或 None

    def __post_init__(self) -> None:
        """仅对 outer_boundary（若提供）执行集中验证并规范化为 float64 独立副本。

        核心字段与兼容字段维持既有"不在构造期校验/复制"的行为，本方法不触碰它们。
        outer_boundary 为 None 时视为"未提供"，跳过全部验证（不作为合法几何）。
        """
        if self.outer_boundary is not None:
            self.outer_boundary = _validate_outer_boundary(self.outer_boundary)

    @property
    def n_nodes(self) -> int:
        """节点总数。"""
        return self.nodes.shape[0]

    @property
    def n_triangles(self) -> int:
        """三角形总数。"""
        return self.triangles.shape[0]

    @property
    def n_boundary_segments(self) -> int:
        """边界线段总数。"""
        return self.boundary_segments.shape[0]


def _signed_area_closed(boundary: np.ndarray) -> float:
    """对隐式闭合（末尾不重复首点）的轮廓计算带符号闭合面积（鞋带公式）。

    约定：正值为逆时针，负值为顺时针。最后一点到第一点的隐式闭合边参与求和。

    参数：
        boundary: (B,2) float64 数组，末尾不重复首点

    返回：
        带符号面积（float）
    """
    r = boundary[:, 0]
    z = boundary[:, 1]
    r_next = np.roll(r, -1)
    z_next = np.roll(z, -1)
    return 0.5 * float(np.sum(r * z_next - r_next * z))


# --------------------------------------------------------------------------- #
# 内部辅助：小型、局部、确定性的二维线段相交判断（用于自交检查）                #
#                                                                             #
# 容差量纲说明：                                                               #
#   - coordinate_tol：坐标比较与 on-segment 包围盒容差，量纲为长度（m）；       #
#   - orientation_tol：orientation 叉积（有向面积 2 倍）符号判断容差，          #
#                      量纲为长度平方（m²）。                                   #
#   两者由轮廓尺度与坐标绝对量级共同确定，量纲不同，不得混用同一个 tol。         #
# --------------------------------------------------------------------------- #

def _orientation(ax, ay, bx, by, cx, cy, orientation_tol: float) -> int:
    """三点 (a,b,c) 的有向面积符号（叉积），带 orientation_tol（长度平方量纲）容差。

    返回：
        +1 逆时针；-1 顺时针；0 共线（在 orientation_tol 容差内）。
    """
    # val 为有向面积的 2 倍，量纲为长度平方，故与 orientation_tol 同量纲比较。
    val = (bx - ax) * (cy - ay) - (by - ay) * (cx - ax)
    if val > orientation_tol:
        return 1
    if val < -orientation_tol:
        return -1
    return 0


def _on_segment(ax, ay, bx, by, px, py, coordinate_tol: float) -> bool:
    """在已知 p 与 (a,b) 共线的前提下，判断 p 是否落在闭线段 [a,b] 上。

    使用带 coordinate_tol（长度量纲）的包围盒判定。
    """
    return (
        min(ax, bx) - coordinate_tol <= px <= max(ax, bx) + coordinate_tol
        and min(ay, by) - coordinate_tol <= py <= max(ay, by) + coordinate_tol
    )


def _segments_intersect(
    p1, p2, p3, p4, coordinate_tol: float, orientation_tol: float
) -> bool:
    """判断闭线段 [p1,p2] 与 [p3,p4] 是否相交（含端点接触与共线重叠）。

    严格交叉判据：
        - 要求两组 orientation 符号分别真正相反，即
          o1*o2 < 0 且 o3*o4 < 0；
        - orientation 为 0 的情况不算严格交叉，交由下方共线/on-segment 特殊分支处理。

    共线/端点接触特殊情况：
        - 某端点与另一条线段共线且落在其闭区间上 → 端点接触或共线重叠。

    参数：
        p1..p4:          线段端点 (x, y) 二元序列；
        coordinate_tol:  坐标/包围盒容差（长度量纲）；
        orientation_tol: 叉积符号容差（长度平方量纲）。
    """
    a1x, a1y = float(p1[0]), float(p1[1])
    a2x, a2y = float(p2[0]), float(p2[1])
    b1x, b1y = float(p3[0]), float(p3[1])
    b2x, b2y = float(p4[0]), float(p4[1])

    o1 = _orientation(a1x, a1y, a2x, a2y, b1x, b1y, orientation_tol)
    o2 = _orientation(a1x, a1y, a2x, a2y, b2x, b2y, orientation_tol)
    o3 = _orientation(b1x, b1y, b2x, b2y, a1x, a1y, orientation_tol)
    o4 = _orientation(b1x, b1y, b2x, b2y, a2x, a2y, orientation_tol)

    # 一般情况：严格交叉（两组符号分别真正相反）
    if o1 * o2 < 0 and o3 * o4 < 0:
        return True

    # 共线/端点接触特殊情况（orientation 为 0 时）
    if o1 == 0 and _on_segment(a1x, a1y, a2x, a2y, b1x, b1y, coordinate_tol):
        return True
    if o2 == 0 and _on_segment(a1x, a1y, a2x, a2y, b2x, b2y, coordinate_tol):
        return True
    if o3 == 0 and _on_segment(b1x, b1y, b2x, b2y, a1x, a1y, coordinate_tol):
        return True
    if o4 == 0 and _on_segment(b1x, b1y, b2x, b2y, a2x, a2y, coordinate_tol):
        return True

    return False


def _check_no_self_intersection(
    boundary: np.ndarray, coordinate_tol: float, orientation_tol: float
) -> None:
    """完整非相邻边自交检查（单一简单闭环）。

    边定义（隐式闭合）：
        边 i 连接点 i 与点 (i+1) mod B。

    检查规则：
        - 遍历所有边对 (i, j)，i < j；
        - 跳过同一条边（i == j）；
        - 跳过相邻边（共享一个正常端点）：j == i+1，或
          (i == 0 且 j == B-1)（第一条边与最后一条边相邻）；
        - 其余非相邻边对：严格交叉、端点接触、共线重叠中任一发生即拒绝；
        - 非相邻边共线但分离不误判（由 on-segment 包围盒判定排除）。

    复杂度 O(B²)，P0 边界规模有限，不引入扫描线或计算几何依赖。

    参数：
        boundary:        (B,2) float64，末尾不重复首点；
        coordinate_tol:  坐标/包围盒容差（长度量纲）；
        orientation_tol: 叉积符号容差（长度平方量纲）。

    异常：
        ValueError: 存在非相邻边自交（含端点接触、共线重叠）
    """
    b = boundary.shape[0]
    # B<3 已在上游拒绝；B==3 三条边两两相邻，不可能存在非相邻边自交。
    if b < 4:
        return

    for i in range(b):
        i_p1 = boundary[i]
        i_p2 = boundary[(i + 1) % b]
        for j in range(i + 1, b):
            # 跳过同一条边
            if j == i:
                continue
            # 跳过相邻边：共享一个端点
            if j == i + 1:
                continue
            if i == 0 and j == b - 1:
                # 第一条边 (0,1) 与最后一条边 (B-1,0) 相邻，共享点 0
                continue
            j_p1 = boundary[j]
            j_p2 = boundary[(j + 1) % b]
            if _segments_intersect(
                i_p1, i_p2, j_p1, j_p2, coordinate_tol, orientation_tol
            ):
                raise ValueError(
                    "outer_boundary 存在非相邻边自交（self-intersection）："
                    f"边 {i}->{(i + 1) % b} 与边 {j}->{(j + 1) % b} 相交、"
                    "端点接触或共线重叠；outer_boundary 必须是单一简单闭环"
                )


def _validate_outer_boundary(outer_boundary_in) -> np.ndarray:
    """验证并返回规范化后的 outer_boundary（float64，(B,2)，顺时针，隐式闭合）。

    集中式验证（不修改坐标顺序，不复制首点，不排序，不规定固定起点）：
        1. 生成独立的 float64 (B,2) 副本（不复用输入数组、不修改输入）；
        2. 二维 (B,2)；
        3. B >= 3；
        4. 所有坐标有限（无 NaN/Inf）；
        5. 首点与末点不同（末尾不重复首点，闭合边长度非零）；
        6. 相邻点不重复；
        7. 至少 3 个互异点；
        8. 带符号闭合面积严格为负（顺时针）且非退化（|area| > 1e-12）；
        9. 单一简单闭环：非相邻边不得严格交叉、端点接触或共线重叠。

    验证顺序说明：
        自交检查安排在 shape/有限性/相邻重复/首尾/方向/面积等基础检查之后，
        使得基础退化输入被各自专属错误明确拒绝；而真正的自交（形状合法、
        方向合法但非简单）轮廓能被明确识别为自交，不被无关错误掩盖。

    所有权：
        使用 np.array(..., copy=True) 生成 float64 独立副本，即使输入已是
        float64 ndarray 也不复用其内存，且不修改输入数组。

    参数：
        outer_boundary_in: 待验证的 outer boundary 输入

    返回：
        (B,2) float64 独立副本数组

    异常：
        ValueError: 不满足上述任一约束
    """
    # 1. 生成独立 float64 副本（不复用输入内存、不修改输入）
    ob = np.array(outer_boundary_in, dtype=np.float64, copy=True)

    # 2. 二维 (B,2)
    if ob.ndim != 2:
        raise ValueError(
            f"outer_boundary 必须是二维数组，实际 ndim={ob.ndim}"
        )
    if ob.shape[1] != 2:
        raise ValueError(
            f"outer_boundary 必须为 (B,2)，实际 shape={ob.shape}"
        )

    b = ob.shape[0]
    # 3. B >= 3
    if b < 3:
        raise ValueError(
            f"outer_boundary 至少需要 3 个点（B>=3），实际 B={b}"
        )

    # 4. 有限性
    if not np.all(np.isfinite(ob)):
        raise ValueError("outer_boundary 含非有限值（NaN/Inf）")

    # 5. 首尾不同（末尾不重复首点，闭合边非零）
    if np.array_equal(ob[0], ob[-1]):
        raise ValueError(
            "outer_boundary 首点与末点相同：应隐式闭合，末尾不得重复首点"
        )

    # 6. 相邻点不重复（相邻显式点对，不含隐式闭合边——闭合边由第 5 步保证非零）
    diffs = np.diff(ob, axis=0)
    seg_len_sq = np.sum(diffs * diffs, axis=1)
    if np.any(seg_len_sq == 0.0):
        raise ValueError("outer_boundary 存在相邻重复点（零长度边）")

    # 7. 至少 3 个互异点
    unique_pts = np.unique(ob, axis=0)
    if unique_pts.shape[0] < 3:
        raise ValueError(
            f"outer_boundary 互异点不足 3 个，实际互异点数={unique_pts.shape[0]}"
        )

    # 8. 带符号闭合面积严格为负（顺时针）且非退化
    area = _signed_area_closed(ob)
    if abs(area) <= 1e-12:
        raise ValueError(
            f"outer_boundary 面积退化（近零或共线）：signed_area={area}"
        )
    if area >= 0:
        raise ValueError(
            "outer_boundary 方向必须为顺时针（带符号闭合面积应严格为负），"
            f"实际 signed_area={area}（逆时针或退化）"
        )

    # 9. 单一简单闭环：非相邻边自交检查（含端点接触、共线重叠）
    #
    # 容差量纲（明确区分，不混用同一个含糊 tol）：
    #   length_scale：轮廓尺度，取 max(1.0, R/Z 跨度, 坐标绝对量级)，
    #                 大坐标下也能提供合理的浮点误差基准。
    #   coordinate_tol  = 1e-12 * length_scale        （长度量纲，用于坐标/on-segment）
    #   orientation_tol = 1e-12 * length_scale^2       （长度平方量纲，用于叉积符号）
    r_span = float(ob[:, 0].max() - ob[:, 0].min())
    z_span = float(ob[:, 1].max() - ob[:, 1].min())
    abs_mag = float(np.max(np.abs(ob)))
    length_scale = max(1.0, r_span, z_span, abs_mag)
    if not np.isfinite(length_scale) or length_scale <= 0:
        length_scale = 1.0

    coordinate_tol = 1e-12 * length_scale
    orientation_tol = 1e-12 * length_scale * length_scale

    _check_no_self_intersection(ob, coordinate_tol, orientation_tol)

    return ob
