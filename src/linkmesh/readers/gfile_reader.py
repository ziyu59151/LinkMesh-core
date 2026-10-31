"""
gfile_reader.py —— GFile/EQDSK 平衡文件读取 Adapter。

职责：
    - 薄 Adapter（接线员）模式：直接调用 OFT GEQDSKEquilibrium 读取 GFile
    - 从 OFT 返回对象的属性中提取 LinkMesh 所需字段
    - 封装为 GFileData 对象返回

禁止事项：
    - 绝对禁止在 LinkMesh 侧手动执行任何原始 GFile 文本 I/O
    - 绝对禁止手动执行字符串切分、Fortran 格式化数值解析或 EQDSK 字段手动提取
    - 不得调用 Triangle 或任何网格生成逻辑
    - 不得写文件

设计原则：
    OFT（OpenFUSIONToolkit）已提供完整、经过验证的 GFile 解析实现。
    LinkMesh 直接复用 GEQDSKEquilibrium 类，不重复造轮子。

"""

from __future__ import annotations

from dataclasses import dataclass
from typing import Optional

import numpy as np


# --------------------------------------------------------------------------- #
# 核心数据结构                                                                  #
# --------------------------------------------------------------------------- #

@dataclass
class GFileData:
    """GFile/EQDSK 平衡文件读取结果（OFT Adapter 输出）。

    所有字段均由 readers/gfile_reader.py 从 OFT GEQDSKEquilibrium 对象属性
    中直接提取，LinkMesh 侧不执行任何原始文本解析。

    所有坐标单位均为米（SI 单位制）。
    R 为大半径方向，Z 为垂直方向（托卡马克柱坐标系）。

    字段来源（对应 GEQDSKEquilibrium 属性名）：
        lcfs_r   ← eq.boundary_R
        lcfs_z   ← eq.boundary_Z
        rmaxis   ← eq.R_mag
        zmaxis   ← eq.Z_mag
        wall_r   ← eq.limiter_R（若为空则 None）
        wall_z   ← eq.limiter_Z（若为空则 None）
        rcentr   ← eq.R_center
        bcentr   ← eq.B_center
        current  ← eq.Ip
    """

    # LCFS 边界坐标（最后闭合磁面）
    # 来源：GEQDSKEquilibrium.boundary_R / boundary_Z
    lcfs_r: np.ndarray               # shape=(N_lcfs,)，LCFS 上各点 R 坐标 [m]
    lcfs_z: np.ndarray               # shape=(N_lcfs,)，LCFS 上各点 Z 坐标 [m]

    # 磁轴坐标
    # 来源：GEQDSKEquilibrium.R_mag / Z_mag
    rmaxis: float                    # 磁轴 R 坐标 [m]
    zmaxis: float                    # 磁轴 Z 坐标 [m]

    # 第一壁点集（可选，若 GFile 不含则为 None）
    # 来源：GEQDSKEquilibrium.limiter_R / limiter_Z
    wall_r: Optional[np.ndarray]     # shape=(N_wall,)，第一壁 R 坐标 [m]
    wall_z: Optional[np.ndarray]     # shape=(N_wall,)，第一壁 Z 坐标 [m]

    # 基本平衡参数（可选，用于后续物理验证与可视化）
    # 来源：GEQDSKEquilibrium 对应属性
    rcentr: Optional[float] = None   # 参考大半径 R0 [m]（eq.R_center）
    bcentr: Optional[float] = None   # 参考磁场强度 B0 [T]（eq.B_center）
    current: Optional[float] = None  # 等离子体电流 Ip [A]（eq.Ip）


# --------------------------------------------------------------------------- #
# 公开接口                                                                      #
# --------------------------------------------------------------------------- #

def read_gfile(filepath: str) -> GFileData:
    """读取 GFile/EQDSK 平衡文件，提取 LinkMesh 所需的边界信息。

    实现策略：薄 Adapter 模式。
    直接调用 OFT GEQDSKEquilibrium 读取 GFile，从其属性中提取字段，
    封装为 GFileData 对象。LinkMesh 侧不执行任何原始文本解析。

    参数：
        filepath: GFile 文件路径（字符串）

    返回：
        GFileData 对象，包含 LCFS 坐标、第一壁坐标、磁轴坐标和基本平衡参数

    异常：
        FileNotFoundError: 文件不存在（由 GEQDSKEquilibrium 内部抛出）
        ValueError: 文件格式不符合 EQDSK 规范（由 GEQDSKEquilibrium 内部抛出）
        ImportError: OFT 未安装或不在 PYTHONPATH 中

    实现要点：
        1. 调用 GEQDSKEquilibrium(filepath) 完成全部 GFile 解析（OFT 负责）
        2. 从 eq.boundary_R / eq.boundary_Z 提取 LCFS 坐标
        3. 从 eq.R_mag / eq.Z_mag 提取磁轴坐标
        4. 从 eq.limiter_R / eq.limiter_Z 提取第一壁坐标（若为空则置 None）
        5. 对 LCFS 坐标执行闭合性检查与逆时针方向检查
        6. 封装为 GFileData 对象返回
    """
    # Import the public OFT equilibrium reader.
    try:
        from OpenFUSIONToolkit.TokaMaker.eqdsk import GEQDSKEquilibrium
    except Exception as exc:
        raise ImportError(
            "无法导入 OFT GFile/EQDSK Reader "
            "OpenFUSIONToolkit.TokaMaker.eqdsk.GEQDSKEquilibrium。"
            f"原始错误：{exc}"
        ) from exc

    # 调用 OFT 完成全部 GFile 解析，LinkMesh 侧不执行任何文本解析
    eq = GEQDSKEquilibrium(filepath)

    # 提取 LCFS 边界坐标
    lcfs_r = np.asarray(eq.boundary_R, dtype=np.float64)
    lcfs_z = np.asarray(eq.boundary_Z, dtype=np.float64)

    # 提取第一壁坐标，若为空则置 None
    lim_r = eq.limiter_R
    lim_z = eq.limiter_Z
    if lim_r is not None and len(lim_r) > 0:
        wall_r = np.asarray(lim_r, dtype=np.float64)
        wall_z = np.asarray(lim_z, dtype=np.float64)
    else:
        wall_r = None
        wall_z = None

    # LCFS 后处理：闭合性检查 + 逆时针方向检查
    # （这是 LinkMesh 侧唯一允许的后处理，不属于原始文本解析）
    lcfs_r, lcfs_z = _ensure_lcfs_closed(lcfs_r, lcfs_z)
    lcfs_r, lcfs_z = _ensure_lcfs_ccw(lcfs_r, lcfs_z)

    # 提取基本平衡参数
    rmaxis = float(eq.R_mag)
    zmaxis = float(eq.Z_mag)
    rcentr = float(eq.R_center) if eq.R_center is not None else None
    bcentr = float(eq.B_center) if eq.B_center is not None else None
    current = float(eq.Ip) if eq.Ip is not None else None

    return GFileData(
        lcfs_r=lcfs_r,
        lcfs_z=lcfs_z,
        rmaxis=rmaxis,
        zmaxis=zmaxis,
        wall_r=wall_r,
        wall_z=wall_z,
        rcentr=rcentr,
        bcentr=bcentr,
        current=current,
    )


# --------------------------------------------------------------------------- #
# 内部辅助函数（LCFS 后处理，LinkMesh 侧唯一允许的后处理逻辑）                  #
# --------------------------------------------------------------------------- #

def _ensure_lcfs_closed(r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """确保 LCFS 边界首尾点相同（闭合）。

    若首尾点不同（欧氏距离超过容差），则追加首点到末尾。

    参数：
        r: LCFS R 坐标数组
        z: LCFS Z 坐标数组

    返回：
        (r, z) 闭合后的坐标数组元组
    """
    if len(r) < 2:
        return r, z
    # 使用相对容差：首尾点距离 < 总周长的 0.1% 视为已闭合
    gap = np.sqrt((r[0] - r[-1]) ** 2 + (z[0] - z[-1]) ** 2)
    perimeter = np.sum(np.sqrt(np.diff(r) ** 2 + np.diff(z) ** 2))
    if perimeter > 0 and gap > 1e-3 * perimeter:
        r = np.append(r, r[0])
        z = np.append(z, z[0])
    elif gap > 1e-10:
        # 绝对容差兜底
        r = np.append(r, r[0])
        z = np.append(z, z[0])
    return r, z


def _ensure_lcfs_ccw(r: np.ndarray, z: np.ndarray) -> tuple[np.ndarray, np.ndarray]:
    """确保 LCFS 边界点按逆时针方向排列。

    使用 Shoelace 公式计算有向面积：
        正值 → 逆时针（保持不变）
        负值 → 顺时针（反转数组）

    参数：
        r: LCFS R 坐标数组（应已闭合，首尾点相同）
        z: LCFS Z 坐标数组（应已闭合，首尾点相同）

    返回：
        (r, z) 逆时针方向的坐标数组元组
    """
    if len(r) < 3:
        return r, z
    # Shoelace 公式：sum(r[i] * z[i+1] - r[i+1] * z[i]) / 2
    # 正值为逆时针（数学正方向），负值为顺时针
    area = 0.5 * np.sum(r[:-1] * z[1:] - r[1:] * z[:-1])
    if area < 0:
        # 顺时针 → 反转为逆时针
        r = r[::-1].copy()
        z = z[::-1].copy()
    return r, z
