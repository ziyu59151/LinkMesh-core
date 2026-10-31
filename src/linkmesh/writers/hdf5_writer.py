"""
hdf5_writer.py —— TRIMEG HDF5 静态网格写出器（完整 11 数据集契约）。

职责：
    - 接收 Mesh2D 对象
    - 按 TRIMEG HDF5 文件结构写出完整 11 个数据集

禁止事项：
    - 不得修改 Mesh2D 对象
    - 不得包含任何网格生成逻辑
    - 不得写 region_ids 到 HDF5
    - 不得对 vert/tria 转置
    - 不得写 tria 第四列
    - 不得访问旧 boundary_segments / boundary_markers / node_markers
    - 不得由 idxevert 反推 Rbc/Zbc；不得把 Steiner 点加入 Rbc/Zbc
    - 不得要求 nbcRZ == nvert_ex；不得把 Rbc/Zbc 转置成 (1,B)；不得重复首点

TRIMEG HDF5 数据布局：
    完整 11 数据集（全部磁盘 dtype float64，位于 /femmesh/）：
        nvert ntria nvert_in nvert_ex vert idxivert idxevert tria nbcRZ Rbc Zbc
    - 8 核心数据集契约保持不变：
          nvert/ntria/nvert_in/nvert_ex 均 (1,1)
          vert     = (N,2)，每行 [R,Z]，不转置
          tria     = (T,3)，1-based（triangles+1），无第四列
          idxivert = (1,nvert_in)，1-based，升序
          idxevert = (1,nvert_ex)，1-based，升序（只出现一次的无向边端点集合）
    - 新增 3 数据集：
          nbcRZ = (1,1)，值为 B（整数语义），float64
          Rbc   = (B,1)，内容为 mesh.outer_boundary[:,0]
          Zbc   = (B,1)，内容为 mesh.outer_boundary[:,1]
    - Rbc/Zbc 来自有序 mesh.outer_boundary（网格生成前几何），
      不由升序 idxevert 反推，不含生成器后插入的 Steiner 边界点；
      nbcRZ 可等于或小于 nvert_ex（不要求相等）。

outer_boundary 验证统一化：
    Writer 不再维护自己那套不完整的 outer_boundary 几何验证（缺少非相邻边自交检查），
    而是复用 linkmesh.mesh 中 Mesh2D 使用的同一个统一验证实现
    _validate_outer_boundary（含 shape/有限性/相邻重复/首尾/方向/退化面积/完整
    非相邻边自交检查：严格交叉、端点接触、共线重叠）。该 helper 是仓库内部统一
    验证实现，不是对外公共 API；Writer 直接在运行时导入它，避免两处验证逻辑分叉。
    Writer 侧仅保留一层很薄的 wrapper，用于对 outer_boundary=None 给出 Writer 专用
    错误信息；复杂几何验证（含自交算法）一律委托给 mesh.py 的统一实现，Writer 不
    复制自交算法。统一 helper 返回独立 float64 副本，Writer 后续只使用该副本，
    不修改 mesh.outer_boundary，不重采样、不排序、不循环移位、不添加闭合点。

关于 MATLAB 中的 h5read(...).' ：那是 MATLAB/HDF5 维度解释（列优先）问题，
不能反向推导 Python Writer 必须转置。h5py 磁盘 shape 直接按上表写出。

"""

from __future__ import annotations

from typing import TYPE_CHECKING

import numpy as np

# 复用 Mesh2D 使用的同一个统一 outer_boundary 验证实现（仓库内部统一验证，
# 非对外公共 API）。Writer 直接在运行时导入该内部 helper，确保 Writer 与
# Mesh2D 使用完全一致的几何契约（含完整非相邻边自交检查），避免验证逻辑分叉。
from linkmesh.mesh import _validate_outer_boundary as _mesh_validate_outer_boundary

if TYPE_CHECKING:
    from linkmesh.mesh import Mesh2D


# --------------------------------------------------------------------------- #
# 内部辅助：节点数组验证                                                        #
# --------------------------------------------------------------------------- #

def _validate_nodes(nodes_in) -> np.ndarray:
    """验证并返回节点坐标数组的 float64 副本视图，不修改原数组。

    要求：
        - 可转换为 NumPy 数组；
        - 二维 (N, 2)；
        - N > 0；
        - 全部有限（无 NaN/Inf）。

    返回：
        (N, 2) float64 数组（asarray 得到，不保证是副本，但绝不原地修改）

    异常：
        ValueError: 不满足上述任一约束
    """
    nodes = np.asarray(nodes_in, dtype=np.float64)
    if nodes.ndim != 2:
        raise ValueError(f"nodes 必须是二维数组，实际 ndim={nodes.ndim}")
    if nodes.shape[1] != 2:
        raise ValueError(f"nodes 必须为 (N,2)，实际 shape={nodes.shape}")
    if nodes.shape[0] == 0:
        raise ValueError("nodes 为空：N=0")
    if not np.all(np.isfinite(nodes)):
        raise ValueError("nodes 含非有限值（NaN/Inf）")
    return nodes


# --------------------------------------------------------------------------- #
# 内部辅助：三角形连接表验证（0-based）                                         #
# --------------------------------------------------------------------------- #

def _validate_triangles(triangles_in, n_nodes: int) -> np.ndarray:
    """验证并返回 0-based 整数三角形连接表，不修改原数组。

    要求：
        - 二维 (T, 3)；
        - T > 0；
        - 有限且具整数语义（等于其四舍五入值），避免 astype 静默截断 1.5/NaN/Inf；
        - 合法 0-based：0 <= index < n_nodes；
        - 每个三角形三个节点索引互不相同。

    返回：
        (T, 3) int64 数组

    异常：
        ValueError: 不满足上述任一约束
    """
    tri_float = np.asarray(triangles_in, dtype=np.float64)
    if tri_float.ndim != 2:
        raise ValueError(f"triangles 必须是二维数组，实际 ndim={tri_float.ndim}")
    if tri_float.shape[1] != 3:
        raise ValueError(f"triangles 必须为 (T,3)，实际 shape={tri_float.shape}")
    if tri_float.shape[0] == 0:
        raise ValueError("triangles 为空：T=0")

    # 有限性 + 整数语义（在 astype(int) 之前）
    if not np.all(np.isfinite(tri_float)):
        raise ValueError("triangles 含非有限值（NaN/Inf），不具备整数语义")
    if not np.all(tri_float == np.round(tri_float)):
        raise ValueError("triangles 含非整数值，不具备整数语义")

    triangles = tri_float.astype(np.int64)

    # 索引范围（0-based）
    if triangles.min() < 0 or triangles.max() >= n_nodes:
        raise ValueError(
            f"三角形节点索引越界：应满足 0 <= index < {n_nodes}，"
            f"实际范围 [{int(triangles.min())}, {int(triangles.max())}]"
        )

    # 退化：三个顶点互不相同
    n_triangles = triangles.shape[0]
    for i in range(n_triangles):
        a, b, c = int(triangles[i, 0]), int(triangles[i, 1]), int(triangles[i, 2])
        if a == b or b == c or a == c:
            raise ValueError(f"三角形 {i} 存在重复顶点索引：{[a, b, c]}")

    return triangles


def _validate_triangle_areas(nodes: np.ndarray, triangles: np.ndarray) -> None:
    """验证每个三角形绝对面积严格大于 1e-12 m²。

    异常：
        ValueError: 存在退化/近零面积三角形
    """
    n_triangles = triangles.shape[0]
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


# --------------------------------------------------------------------------- #
# 内部辅助：outer_boundary 验证（用于 nbcRZ/Rbc/Zbc）—— 复用 mesh.py 统一实现   #
# --------------------------------------------------------------------------- #

def _validate_outer_boundary(outer_boundary_in) -> np.ndarray:
    """Writer 侧薄 wrapper：对 None 给出 Writer 专用错误，其余委托统一验证。

    复杂几何验证（shape/有限性/相邻重复/首尾/方向/退化面积/完整非相邻边自交：
    严格交叉、端点接触、共线重叠）全部委托给 linkmesh.mesh 的统一实现
    _mesh_validate_outer_boundary（即 Mesh2D 使用的同一个 helper），Writer 不复制
    自交算法、不维护第二套几何验证。

    该 wrapper 仅负责：
        - outer_boundary 缺失（None）时给出面向 TRIMEG 写出的 Writer 专用错误信息
          （便于定位是“未提供 outer boundary”而非几何非法）；
        - 非 None 时调用统一 helper 完成全部几何验证并返回其独立 float64 副本。

    由于 Mesh2D.outer_boundary 是公开可变 ndarray，构造后可能被外部直接修改，
    Writer 在打开目标文件前重新执行该统一验证，作为二次防线。

    返回：
        (B,2) float64 独立副本（由统一 helper 生成）

    异常：
        ValueError: outer_boundary 为 None，或统一验证判定其非法（含自交）
    """
    if outer_boundary_in is None:
        raise ValueError(
            "mesh.outer_boundary 缺失（None）：写出完整 11 数据集需要有序 outer boundary"
        )
    # 委托给 mesh.py 统一验证（含完整非相邻边自交检查），返回独立 float64 副本。
    return _mesh_validate_outer_boundary(outer_boundary_in)


# --------------------------------------------------------------------------- #
# 内部辅助：拓扑外边界节点提取                                                  #
# --------------------------------------------------------------------------- #

def _extract_boundary_indices(triangles: np.ndarray, n_nodes: int):
    """从 0-based triangles 拓扑推导外边界/内部节点的 1-based 升序索引。

    算法（v1.11）：
        - 每个三角形生成三条边 (v0,v1)、(v1,v2)、(v2,v0)；
        - 每条边规范化为无向节点对 (min(a,b), max(a,b))；
        - 统计出现次数：
              一次   → 拓扑外边界边；
              两次   → 内部共享边；
              超过两次 → 非流形边，抛 ValueError；
        - 若无任何拓扑外边界边，抛 ValueError；
        - idxevert = 所有外边界边端点并集 +1，升序（1-based）；
        - idxivert = {1..N} \\ idxevert 的升序补集。

    返回：
        (idxivert, idxevert) 两个 int64 一维数组（1-based，升序）

    异常：
        ValueError: 存在非流形边，或无拓扑外边界边，或集合一致性失败
    """
    edge_count: dict = {}
    for i in range(triangles.shape[0]):
        v0 = int(triangles[i, 0])
        v1 = int(triangles[i, 1])
        v2 = int(triangles[i, 2])
        for a, b in ((v0, v1), (v1, v2), (v2, v0)):
            key = (a, b) if a < b else (b, a)
            edge_count[key] = edge_count.get(key, 0) + 1

    boundary_nodes: set = set()
    for (a, b), cnt in edge_count.items():
        if cnt == 1:
            boundary_nodes.add(a)
            boundary_nodes.add(b)
        elif cnt == 2:
            continue
        else:
            raise ValueError(
                f"检测到非流形边 {(a, b)}：被 {cnt} 个三角形共享(>2)，"
                "网格非流形，拒绝写出"
            )

    if len(boundary_nodes) == 0:
        raise ValueError("未识别到任何拓扑外边界边，拒绝写出")

    # 1-based 升序外边界节点
    idxevert = np.array(sorted(idx + 1 for idx in boundary_nodes), dtype=np.int64)

    # 补集：{1..N} \ idxevert，升序
    all_1based = set(range(1, n_nodes + 1))
    interior_set = all_1based - set(int(x) for x in idxevert.tolist())
    idxivert = np.array(sorted(interior_set), dtype=np.int64)

    # 一致性检查：无重复、互斥、并集覆盖 1..N
    ev_set = set(int(x) for x in idxevert.tolist())
    iv_set = set(int(x) for x in idxivert.tolist())
    if len(ev_set) != len(idxevert):
        raise ValueError("idxevert 存在重复索引")
    if len(iv_set) != len(idxivert):
        raise ValueError("idxivert 存在重复索引")
    if ev_set & iv_set:
        raise ValueError(f"idxivert 与 idxevert 存在重叠：{sorted(ev_set & iv_set)}")
    if (ev_set | iv_set) != all_1based:
        raise ValueError("idxivert 与 idxevert 并集未覆盖全部节点 {1..N}")
    if len(idxivert) + len(idxevert) != n_nodes:
        raise ValueError(
            f"nvert_in + nvert_ex（{len(idxivert)}+{len(idxevert)}）"
            f"与 nvert（{n_nodes}）不一致"
        )

    return idxivert, idxevert


# --------------------------------------------------------------------------- #
# 公开接口                                                                      #
# --------------------------------------------------------------------------- #

def write_hdf5(mesh: "Mesh2D", filepath: str, verbose: bool = True) -> None:
    """将 Mesh2D 对象写出为 TRIMEG 标准 HDF5 静态网格文件（v2.0 完整 11 数据集）。

    按 TRIMEG HDF5 数据集结构写出。
    写出完整 11 个数据集，全部磁盘 dtype 为 float64。

    参数：
        mesh:     Mesh2D 对象（内部三角形索引全部为 0-based，含有序 outer_boundary）
        filepath: 输出 .h5 文件路径（若已存在则覆盖）
        verbose:  是否打印写出完成信息（默认 True）

    写出的 HDF5 结构（11 数据集，磁盘 shape / dtype）：
        /femmesh/nvert      (1,1) float64   节点总数 N（整数语义）
        /femmesh/ntria      (1,1) float64   三角形总数 T（整数语义）
        /femmesh/nvert_in   (1,1) float64   内部节点数 nvert_in
        /femmesh/nvert_ex   (1,1) float64   外边界节点数 nvert_ex
        /femmesh/vert       (N,2) float64   节点坐标，每行 [R,Z]，不转置
        /femmesh/idxivert   (1,nvert_in) float64  内部节点全局索引，1-based，升序
        /femmesh/idxevert   (1,nvert_ex) float64  外边界节点全局索引，1-based，升序
        /femmesh/tria       (T,3) float64   三角形连接表，1-based，无第四列
        /femmesh/nbcRZ      (1,1) float64   有序 outer boundary 点数 B（整数语义）
        /femmesh/Rbc        (B,1) float64   mesh.outer_boundary[:,0]
        /femmesh/Zbc        (B,1) float64   mesh.outer_boundary[:,1]

    索引与偏置约定：
        - mesh.triangles 内部保持 0-based；
        - tria 写出 (mesh.triangles + 1).astype(float64)（1-based，不转置，无第四列）；
        - region_ids 不写入 TRIMEG HDF5。

    Rbc/Zbc/nbcRZ 约定：
        - 直接读 mesh.outer_boundary（有序、顺时针、隐式闭合、不重复首点）；
        - 不由升序 idxevert 反推；不含 Steiner 点；nbcRZ=B 可小于 nvert_ex。

    outer_boundary 验证（v2.2）：
        Writer 复用 linkmesh.mesh 的统一 _validate_outer_boundary（含完整非相邻边
        自交检查），作为二次防线在打开目标文件前重新验证 mesh.outer_boundary。

    重要：全部输入验证（含 outer_boundary 统一验证）、面积检查、边统计、非流形
    检查、拓扑索引派生与集合一致性检查均在打开/覆盖目标文件之前完成；任一失败时
    不创建或截断目标文件。

    异常：
        ImportError: h5py 不可用（保留原始异常链）
        ValueError:  输入不满足 v2.0 契约（shape/有限值/整数语义/索引范围/退化/
                     非流形/无拓扑外边界/集合一致性/outer_boundary 非法或自交等）
    """
    # ------------------------------------------------------------------ #
    # Step 1：全部验证与拓扑派生（在打开文件之前）                          #
    # ------------------------------------------------------------------ #
    nodes = _validate_nodes(mesh.nodes)
    n_nodes = nodes.shape[0]

    triangles = _validate_triangles(mesh.triangles, n_nodes)
    n_triangles = triangles.shape[0]

    _validate_triangle_areas(nodes, triangles)

    idxivert, idxevert = _extract_boundary_indices(triangles, n_nodes)
    nvert_in = int(idxivert.shape[0])
    nvert_ex = int(idxevert.shape[0])

    # outer_boundary 统一验证（在打开文件之前；Writer 二次防线，不修改输入）。
    # 复杂几何验证（含非相邻边自交）委托给 mesh.py 统一 helper，返回独立 float64 副本。
    outer_boundary = _validate_outer_boundary(getattr(mesh, "outer_boundary", None))
    n_bc = int(outer_boundary.shape[0])  # B

    # ------------------------------------------------------------------ #
    # Step 2：组装 11 个磁盘数组（全部 float64）                            #
    # ------------------------------------------------------------------ #
    # 计数标量：二维单元素数组 (1,1)，不得写成 HDF5 标量
    nvert_ds = np.array([[n_nodes]], dtype=np.float64)
    ntria_ds = np.array([[n_triangles]], dtype=np.float64)
    nvert_in_ds = np.array([[nvert_in]], dtype=np.float64)
    nvert_ex_ds = np.array([[nvert_ex]], dtype=np.float64)

    # vert：直接写 (N,2)，不转置
    vert_ds = nodes.astype(np.float64)

    # tria：0-based → 1-based，(T,3)，float64，不转置、无第四列
    tria_ds = (triangles + 1).astype(np.float64)

    # idxivert / idxevert：1-based，行向量 (1, nvert_in) / (1, nvert_ex)，float64
    # 即使 nvert_in==0 也保持 shape (1,0)
    idxivert_ds = idxivert.astype(np.float64).reshape(1, nvert_in)
    idxevert_ds = idxevert.astype(np.float64).reshape(1, nvert_ex)

    # nbcRZ：(1,1) float64，值为 B（整数语义）
    nbcRZ_ds = np.array([[n_bc]], dtype=np.float64)

    # Rbc / Zbc：(B,1) float64，直接取 outer_boundary 两列（不转置成 (1,B)，不重复首点）
    Rbc_ds = outer_boundary[:, 0].astype(np.float64).reshape(n_bc, 1)
    Zbc_ds = outer_boundary[:, 1].astype(np.float64).reshape(n_bc, 1)

    # ------------------------------------------------------------------ #
    # Step 3：延迟导入 h5py 并写出（覆盖语义）                              #
    # ------------------------------------------------------------------ #
    try:
        import h5py
    except ImportError as e:
        raise ImportError(
            "无法导入 h5py。请运行：pip install h5py"
            f"\n原始错误：{e}"
        ) from e

    with h5py.File(filepath, "w") as f:
        grp = f.create_group("femmesh")

        # 8 核心数据集
        grp.create_dataset("nvert",    data=nvert_ds)
        grp.create_dataset("ntria",    data=ntria_ds)
        grp.create_dataset("nvert_in", data=nvert_in_ds)
        grp.create_dataset("nvert_ex", data=nvert_ex_ds)
        grp.create_dataset("vert",     data=vert_ds)
        grp.create_dataset("idxivert", data=idxivert_ds)
        grp.create_dataset("idxevert", data=idxevert_ds)
        grp.create_dataset("tria",     data=tria_ds)

        # 新增 3 数据集（v2.0 完整 11 数据集）
        grp.create_dataset("nbcRZ",    data=nbcRZ_ds)
        grp.create_dataset("Rbc",      data=Rbc_ds)
        grp.create_dataset("Zbc",      data=Zbc_ds)

    if verbose:
        print(
            f"HDF5 写出完成：{filepath}，"
            f"节点数: {n_nodes}，三角形数: {n_triangles}，"
            f"内部节点数: {nvert_in}，外边界节点数: {nvert_ex}，"
            f"outer boundary 点数: {n_bc}"
        )
