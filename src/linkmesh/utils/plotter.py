"""
plotter.py —— matplotlib 可视化自检工具。

职责：
    - 绘制 LCFS 边界与生成三角网格的叠加图
    - 输出 .png 自检图，供人眼快速判断网格质量

禁止事项：
    - 不得修改任何数据对象（GFileData、Mesh2D）

v1.9 M2 说明：
    - plot_mesh() 主网格只依赖 mesh.nodes / mesh.triangles；
    - 区域着色只使用实际 mesh.region_ids，不使用固定 {1:plasma,2:vacuum,3:wall} 文本映射；
    - LCFS/第一壁来自 GFileData，绘制时若首尾不同则临时追加首点用于显示闭合（不修改 GFileData）；
    - boundary_segments/boundary_markers 为空时跳过旧边界段绘制。

v1.11 M2.5 说明：
    - Mesh2D 类型注解来源改为 linkmesh.mesh（唯一正式定义），仍仅用于 TYPE_CHECKING；
    - 运行时绘图行为完全不变。

用户可见文本约定（v1.9 M2 PNG 修复）：
    - 所有会渲染进 PNG（标题、图例、统计框）或打印到终端的字符串一律使用英文，
      以避免服务器默认 DejaVu Sans 字体缺少中文字形导致的 Glyph missing 警告。
    - 代码注释与文档字符串保留中文（不交给 Matplotlib 渲染）。

"""

from __future__ import annotations

import os
import tempfile
from pathlib import Path
from typing import TYPE_CHECKING, Optional

import numpy as np

if TYPE_CHECKING:
    from linkmesh.readers.gfile_reader import GFileData
    from linkmesh.mesh import Mesh2D


def _closed_for_display(r, z):
    """返回用于显示闭合的坐标副本：若首尾不同则临时追加首点。

    仅用于绘图显示，绝不修改传入的底层数据对象。

    参数：
        r: R 坐标（array-like）
        z: Z 坐标（array-like）

    返回：
        (r_disp, z_disp) numpy 数组副本，保证首尾闭合
    """
    r_arr = np.asarray(r, dtype=np.float64).reshape(-1)
    z_arr = np.asarray(z, dtype=np.float64).reshape(-1)
    if len(r_arr) < 2:
        return r_arr, z_arr
    if not (np.isclose(r_arr[0], r_arr[-1]) and np.isclose(z_arr[0], z_arr[-1])):
        r_arr = np.append(r_arr, r_arr[0])
        z_arr = np.append(z_arr, z_arr[0])
    return r_arr, z_arr


def _reference_outside_boundary(
    reference_r,
    reference_z,
    boundary: np.ndarray,
) -> bool:
    """Return whether any finite reference point lies outside the boundary."""
    points = np.column_stack(
        (
            np.asarray(reference_r, dtype=np.float64).reshape(-1),
            np.asarray(reference_z, dtype=np.float64).reshape(-1),
        )
    )
    polygon = np.asarray(boundary, dtype=np.float64)
    if (
        points.shape[0] == 0
        or polygon.ndim != 2
        or polygon.shape[0] < 3
        or polygon.shape[1] != 2
        or not np.all(np.isfinite(points))
    ):
        return True

    scale = max(1.0, float(np.max(np.abs(polygon))))
    tolerance = 1e-12 * scale

    for point in points:
        x, y = float(point[0]), float(point[1])
        inside = False
        on_boundary = False
        previous = polygon[-1]

        for current in polygon:
            x1, y1 = float(previous[0]), float(previous[1])
            x2, y2 = float(current[0]), float(current[1])
            cross = (x - x1) * (y2 - y1) - (y - y1) * (x2 - x1)
            if (
                abs(cross) <= tolerance * scale
                and min(x1, x2) - tolerance <= x <= max(x1, x2) + tolerance
                and min(y1, y2) - tolerance <= y <= max(y1, y2) + tolerance
            ):
                on_boundary = True
                break

            if (y1 > y) != (y2 > y):
                intersection_x = x1 + (y - y1) * (x2 - x1) / (y2 - y1)
                if x < intersection_x:
                    inside = not inside
            previous = current

        if not (inside or on_boundary):
            return True

    return False


def plot_mesh(
    gfile_data: Optional["GFileData"],
    mesh: "Mesh2D",
    filepath: str,
    show_boundary_markers: bool = True,
    show_region_ids: bool = False,
    *,
    lcfs_reference_only: bool = False,
) -> None:
    """绘制 LCFS 边界与生成三角网格的叠加图，输出 PNG 自检图。

    参数：
        gfile_data:            可选 GFileData；为 None 时不绘制平衡参考信息
        mesh:                  Mesh2D 对象，提供三角网格数据
        filepath:              输出 .png 文件路径
        show_boundary_markers: 是否用不同颜色标注边界标记类型（仅当存在边界段时）
        show_region_ids:       是否按实际 region_ids 着色填充
        lcfs_reference_only:   为 True 时，GFileData 仅提供 LCFS reference，
                               不绘制第一壁或磁轴；默认 False 保持既有 GFile 绘图行为

    实现要点（v1.9 M2）：
        1. 使用 matplotlib.tri.Triangulation 绘制三角网格（仅依赖 nodes/triangles）
        2. 区域着色只使用实际 region_ids，图例标签为 "OFT region <id>"，
           三角形填色与图例色块严格共用同一离散颜色来源
        3. 叠加绘制 LCFS 边界曲线（红色实线，显示闭合）
        4. 若有第一壁数据，叠加绘制第一壁曲线（蓝色虚线，显示闭合）
        5. 标注磁轴位置（红色叉号）
        6. 保持纵横比相等（aspect='equal'），保存为 PNG，dpi=150

    离散区域颜色一致性：
        对排序后的实际 OFT 区域 ID 建立确定的离散索引映射
        （actual OFT ID → 0,1,...,N-1），使用 ListedColormap 配合 BoundaryNorm，
        使 tripcolor() 中每个三角形的颜色与图例代理色块严格一致。
        不写死区域数量或固定 {1,2,3}，合法 N=1 与 N>3 均可绘制。
    """
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.colors import BoundaryNorm, ListedColormap
    except ImportError as e:
        raise ImportError(
            "无法导入 matplotlib。请运行：pip install matplotlib"
            f"\n原始错误：{e}"
        ) from e

    fig, ax = plt.subplots(figsize=(6, 8))

    # --- 绘制三角网格（仅依赖 nodes/triangles）---
    R = mesh.nodes[:, 0]
    Z = mesh.nodes[:, 1]
    triang = mtri.Triangulation(R, Z, mesh.triangles)

    legend_handles = []

    if show_region_ids and mesh.region_ids is not None and len(mesh.region_ids) > 0:
        # 对排序后的实际 region ID 建立确定的离散索引映射：
        #   actual OFT ID → 0, 1, ..., N-1
        # 不根据整数值推断 plasma/vacuum/outer，不写死区域数量。
        unique_regions = sorted(set(int(x) for x in mesh.region_ids.tolist()))
        n_reg = len(unique_regions)
        region_to_index = {rid: i for i, rid in enumerate(unique_regions)}

        # 离散颜色表（长度等于实际区域数），来自稳定的 tab10 采样，
        # 使用 ListedColormap + BoundaryNorm 明确离散归一化，
        # 保证三角形填色与图例色块严格一致。
        base_colors = plt.get_cmap("tab10")(np.linspace(0.0, 1.0, 10))
        region_colors = [base_colors[i % 10] for i in range(n_reg)]
        listed_cmap = ListedColormap(region_colors)
        # 边界覆盖 [-0.5, 0.5, ..., N-0.5]，使离散索引 i 落在第 i 个颜色格
        bounds = np.arange(-0.5, n_reg + 0.5, 1.0)
        norm = BoundaryNorm(bounds, listed_cmap.N)

        # 每个三角形对应的离散索引
        reg_index = np.array(
            [region_to_index[int(r)] for r in mesh.region_ids],
            dtype=np.float64,
        )
        ax.tripcolor(
            triang,
            facecolors=reg_index,
            cmap=listed_cmap,
            norm=norm,
            alpha=0.35,
            edgecolors="none",
        )

        # 图例代理色块直接使用同一颜色表中的对应颜色
        from matplotlib.patches import Patch
        legend_handles = [
            Patch(facecolor=region_colors[region_to_index[rid]], alpha=0.35,
                  label=f"OFT region {rid}")
            for rid in unique_regions
        ]
        # 叠加网格线
        ax.triplot(triang, color="#555555", linewidth=0.3, alpha=0.6)
    else:
        # 仅绘制网格线
        ax.triplot(triang, color="#555555", linewidth=0.3, alpha=0.8)

    # --- 绘制实际计算域边界；该边界不暗示 LCFS/第一壁等物理语义 ---
    if mesh.outer_boundary is not None:
        boundary_r, boundary_z = _closed_for_display(
            mesh.outer_boundary[:, 0],
            mesh.outer_boundary[:, 1],
        )
        ax.plot(
            boundary_r,
            boundary_z,
            color="black",
            linewidth=1.6,
            label="Mesh boundary",
            zorder=3,
        )

    # --- GFile 平衡信息仅作为可选参考叠加 ---
    if gfile_data is not None:
        if (
            mesh.outer_boundary is not None
            and _reference_outside_boundary(
                gfile_data.lcfs_r,
                gfile_data.lcfs_z,
                mesh.outer_boundary,
            )
        ):
            print(
                "WARNING: LCFS reference is not completely inside "
                "the actual mesh boundary."
            )

        lcfs_r_disp, lcfs_z_disp = _closed_for_display(
            gfile_data.lcfs_r,
            gfile_data.lcfs_z,
        )
        ax.plot(
            lcfs_r_disp,
            lcfs_z_disp,
            color="red",
            linewidth=1.5,
            linestyle="--",
            label="LCFS reference",
            zorder=3,
        )

        if (
            not lcfs_reference_only
            and gfile_data.wall_r is not None
            and gfile_data.wall_z is not None
        ):
            wall_r_disp, wall_z_disp = _closed_for_display(
                gfile_data.wall_r,
                gfile_data.wall_z,
            )
            ax.plot(
                wall_r_disp,
                wall_z_disp,
                color="blue",
                linewidth=1.2,
                linestyle="--",
                label="First wall",
                zorder=3,
            )

        if not lcfs_reference_only:
            ax.plot(
                gfile_data.rmaxis,
                gfile_data.zmaxis,
                marker="x",
                color="red",
                markersize=8,
                markeredgewidth=2,
                label=(
                    f"Magnetic axis ({gfile_data.rmaxis:.3f}, "
                    f"{gfile_data.zmaxis:.3f})"
                ),
                zorder=4,
            )

    # --- 旧边界段绘制：仅当 boundary_segments/boundary_markers 非空时 ---
    has_boundary = (
        show_boundary_markers
        and mesh.boundary_segments is not None
        and mesh.boundary_markers is not None
        and mesh.n_boundary_segments > 0
        and len(mesh.boundary_markers) > 0
    )
    if has_boundary:
        # 通用着色：按实际 marker 值分配颜色，不使用固定语义文本
        unique_markers = sorted(set(int(m) for m in mesh.boundary_markers.tolist()))
        base_marker_colors = plt.get_cmap("Set1")(np.linspace(0.0, 1.0, 9))
        marker_to_color = {
            mk: base_marker_colors[i % 9] for i, mk in enumerate(unique_markers)
        }
        plotted: set = set()
        for i in range(mesh.n_boundary_segments):
            seg = mesh.boundary_segments[i]
            marker = int(mesh.boundary_markers[i]) if len(mesh.boundary_markers) > i else 0
            if marker == 0:
                continue
            color = marker_to_color.get(marker, "gray")
            label = f"Boundary marker {marker}" if marker not in plotted else None
            plotted.add(marker)
            p0 = mesh.nodes[seg[0]]
            p1 = mesh.nodes[seg[1]]
            ax.plot(
                [p0[0], p1[0]],
                [p0[1], p1[1]],
                color=color,
                linewidth=1.0,
                label=label,
                zorder=2,
            )

    # --- 格式设置 ---
    ax.set_xlabel("R [m]", fontsize=11)
    ax.set_ylabel("Z [m]", fontsize=11)
    ax.set_title("LinkMesh Mesh Validation", fontsize=13)
    ax.set_aspect("equal")

    # 合并图例（区域填充代理 + 曲线/磁轴自动图例）
    handles, labels = ax.get_legend_handles_labels()
    if legend_handles:
        handles = list(handles) + legend_handles
        labels = list(labels) + [h.get_label() for h in legend_handles]
    if handles:
        ax.legend(handles, labels, loc="upper right", fontsize=8, framealpha=0.8)

    # --- 统计信息文本框 ---
    # 始终显示 Nodes / Triangles；仅当存在真实边界段时显示 Boundary segments，
    # 避免空兼容边界数组导致误导性的 "Boundary segments: 0"。
    info_lines = [
        f"Nodes: {mesh.n_nodes}",
        f"Triangles: {mesh.n_triangles}",
    ]
    if mesh.n_boundary_segments > 0:
        info_lines.append(f"Boundary segments: {mesh.n_boundary_segments}")
    info_text = "\n".join(info_lines)
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"Mesh validation plot saved: {filepath}")


def plot_regions_inspection(
    plan,
    records,
    filepath: str,
) -> None:
    """Plot the inspected curves, final Region IDs, and containment tree."""
    try:
        import matplotlib.pyplot as plt
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError(
            "无法导入 matplotlib。请运行：pip install matplotlib"
        ) from exc

    record_by_curve_id = {
        record.curve_id: record
        for record in records
    }
    if set(record_by_curve_id) != {
        entry.curve_id for entry in plan.entries
    }:
        raise ValueError("inspection plan 与 regions map 的 curve_id 不一致")

    entry_by_curve_id = {
        entry.curve_id: entry
        for entry in plan.entries
    }
    ordered_records = sorted(
        records,
        key=lambda record: record.region_id,
    )
    legend_columns = (
        1
        if len(ordered_records) <= 2
        else 2
        if len(ordered_records) <= 6
        else 3
    )
    legend_rows = (
        len(ordered_records) + legend_columns - 1
    ) // legend_columns
    bottom_margin = min(0.46, 0.14 + 0.055 * legend_rows)

    figure, axis = plt.subplots(figsize=(10, 7))
    figure.subplots_adjust(
        left=0.10,
        right=0.97,
        top=0.92,
        bottom=bottom_margin,
    )
    handles = []

    for record in ordered_records:
        entry = entry_by_curve_id[record.curve_id]
        points = np.array(entry.points, dtype=np.float64, copy=True)
        display = np.vstack((points, points[0]))
        axis.fill(
            display[:, 0],
            display[:, 1],
            color=record.color,
            alpha=0.10,
            zorder=entry.depth,
        )
        axis.plot(
            display[:, 0],
            display[:, 1],
            color=record.color,
            linewidth=1.8,
            zorder=10 + entry.depth,
        )
        children = ",".join(
            str(child_id) for child_id in record.child_region_ids
        ) or "-"
        handles.append(
            Patch(
                facecolor=record.color,
                edgecolor=record.color,
                alpha=0.35,
                label=(
                    f"Region {record.region_id}: curve {record.curve_id}, "
                    f"parent {record.parent_region_id}, children {children}"
                ),
            )
        )

    axis.set_xlabel("R [m]")
    axis.set_ylabel("Z [m]")
    axis.set_title("LinkMesh Regions Inspection")
    axis.set_aspect("equal", adjustable="box")
    figure.legend(
        handles=handles,
        loc="lower center",
        bbox_to_anchor=(0.5, 0.02),
        ncols=legend_columns,
        fontsize=7,
        framealpha=0.9,
    )
    figure.savefig(filepath, dpi=160, bbox_inches="tight")
    plt.close(figure)


def plot_regions_mesh(
    mesh: "Mesh2D",
    records,
    filepath: str,
) -> None:
    """Atomically plot a generated mesh using regions-map display colors."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.colors import BoundaryNorm, ListedColormap
        from matplotlib.patches import Patch
    except ImportError as exc:
        raise ImportError(
            "无法导入 matplotlib。请运行：pip install matplotlib"
        ) from exc

    ordered_records = tuple(
        sorted(records, key=lambda record: record.region_id)
    )
    if len(ordered_records) < 2:
        raise ValueError("regions mesh plot 至少需要两个 Region map records")

    record_by_id = {
        record.region_id: record
        for record in ordered_records
    }
    if len(record_by_id) != len(ordered_records):
        raise ValueError("regions map 含重复 Region ID")

    mesh_ids = {
        int(region_id)
        for region_id in np.asarray(mesh.region_ids).reshape(-1)
    }
    if mesh_ids != set(record_by_id):
        raise ValueError(
            "Mesh2D Region ID 集合与 regions map 不一致"
        )
    if len(mesh.region_ids) != mesh.n_triangles:
        raise ValueError("Mesh2D region_ids 长度与三角形数不一致")

    output_path = Path(filepath)
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"mesh PNG 父目录不存在：{output_path.parent}"
        )

    region_ids = [record.region_id for record in ordered_records]
    region_to_index = {
        region_id: index
        for index, region_id in enumerate(region_ids)
    }
    colors = [record.color for record in ordered_records]
    color_indices = np.array(
        [region_to_index[int(value)] for value in mesh.region_ids],
        dtype=np.float64,
    )
    listed_cmap = ListedColormap(colors)
    norm = BoundaryNorm(
        np.arange(-0.5, len(colors) + 0.5, 1.0),
        listed_cmap.N,
    )

    legend_columns = (
        1
        if len(ordered_records) <= 2
        else 2
        if len(ordered_records) <= 6
        else 3
    )
    legend_rows = (
        len(ordered_records) + legend_columns - 1
    ) // legend_columns
    bottom_margin = min(0.50, 0.18 + 0.055 * legend_rows)

    temporary_path: Path | None = None
    figure = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            prefix=f".{output_path.stem}.",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        figure, axis = plt.subplots(figsize=(10, 7))
        figure.subplots_adjust(
            left=0.10,
            right=0.97,
            top=0.91,
            bottom=bottom_margin,
        )
        triangulation = mtri.Triangulation(
            mesh.nodes[:, 0],
            mesh.nodes[:, 1],
            mesh.triangles,
        )
        axis.tripcolor(
            triangulation,
            facecolors=color_indices,
            cmap=listed_cmap,
            norm=norm,
            edgecolors="none",
            alpha=0.48,
        )
        axis.triplot(
            triangulation,
            color="#555555",
            linewidth=0.28,
            alpha=0.72,
        )

        if mesh.outer_boundary is not None:
            boundary = np.vstack(
                (mesh.outer_boundary, mesh.outer_boundary[0])
            )
            axis.plot(
                boundary[:, 0],
                boundary[:, 1],
                color="black",
                linewidth=1.7,
                zorder=4,
            )

        handles = []
        for record in ordered_records:
            children = ",".join(
                str(value) for value in record.child_region_ids
            ) or "-"
            label = (
                f"Region {record.region_id}: curve {record.curve_id}, "
                f"parent {record.parent_region_id}, children {children}"
            )
            if record.physical_label:
                label += f", label {record.physical_label}"
            handles.append(
                Patch(
                    facecolor=record.color,
                    edgecolor=record.color,
                    alpha=0.55,
                    label=label,
                )
            )

        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.set_title("LinkMesh Multi-Region Mesh")
        axis.set_aspect("equal", adjustable="box")
        figure.legend(
            handles=handles,
            loc="lower center",
            bbox_to_anchor=(0.5, 0.055),
            ncols=legend_columns,
            fontsize=7,
            framealpha=0.9,
        )
        figure.text(
            0.5,
            0.015,
            (
                f"Nodes: {mesh.n_nodes}    "
                f"Triangles: {mesh.n_triangles}    "
                f"Regions: {len(ordered_records)}"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        figure.savefig(
            temporary_path,
            dpi=160,
            bbox_inches="tight",
            format="png",
        )
        plt.close(figure)
        figure = None
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if figure is not None:
            plt.close(figure)
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def plot_triangle_shape_quality(
    mesh: "Mesh2D",
    shape_quality,
    filepath: str,
) -> None:
    """Atomically plot final-mesh per-triangle shape quality Q."""
    try:
        import matplotlib.pyplot as plt
        import matplotlib.tri as mtri
        from matplotlib.colors import LinearSegmentedColormap, Normalize
    except ImportError as exc:
        raise ImportError(
            "无法导入 matplotlib。请运行：pip install matplotlib"
        ) from exc

    quality = np.array(shape_quality, dtype=np.float64, copy=True)
    if quality.ndim != 1 or quality.shape != (mesh.n_triangles,):
        raise ValueError(
            "shape quality Q 必须是一维且与 triangles 一一对应"
        )
    if not np.all(np.isfinite(quality)):
        raise ValueError("shape quality Q 含 NaN/Inf")

    tolerance = 64.0 * np.finfo(np.float64).eps
    if np.any(quality < -tolerance) or np.any(quality > 1.0 + tolerance):
        raise ValueError("shape quality Q 明显超出范围 [0,1]")
    display_quality = np.clip(quality, 0.0, 1.0)

    output_path = Path(filepath)
    if not output_path.parent.is_dir():
        raise FileNotFoundError(
            f"quality PNG 父目录不存在：{output_path.parent}"
        )

    colormap = LinearSegmentedColormap.from_list(
        "linkmesh_shape_quality",
        [(0.0, "red"), (0.5, "blue"), (1.0, "green")],
    )
    normalization = Normalize(vmin=0.0, vmax=1.0, clip=False)

    temporary_path: Path | None = None
    figure = None
    try:
        with tempfile.NamedTemporaryFile(
            suffix=".png",
            prefix=f".{output_path.stem}.",
            dir=output_path.parent,
            delete=False,
        ) as temporary:
            temporary_path = Path(temporary.name)

        figure, axis = plt.subplots(figsize=(10, 7))
        triangulation = mtri.Triangulation(
            mesh.nodes[:, 0],
            mesh.nodes[:, 1],
            mesh.triangles,
        )
        color_mesh = axis.tripcolor(
            triangulation,
            facecolors=display_quality,
            cmap=colormap,
            norm=normalization,
            edgecolors="#555555",
            linewidth=0.25,
        )

        if mesh.outer_boundary is not None:
            boundary = np.vstack(
                (mesh.outer_boundary, mesh.outer_boundary[0])
            )
            axis.plot(
                boundary[:, 0],
                boundary[:, 1],
                color="black",
                linewidth=1.7,
                zorder=4,
            )

        colorbar = figure.colorbar(color_mesh, ax=axis)
        colorbar.set_label("Shape quality Q")
        axis.set_xlabel("R [m]")
        axis.set_ylabel("Z [m]")
        axis.set_title("Triangle Shape Quality")
        axis.set_aspect("equal", adjustable="box")
        figure.text(
            0.5,
            0.015,
            (
                f"Nodes: {mesh.n_nodes}    "
                f"Triangles: {mesh.n_triangles}    "
                f"Regions: {len(np.unique(mesh.region_ids))}"
            ),
            ha="center",
            va="bottom",
            fontsize=8,
        )
        figure.subplots_adjust(bottom=0.10)
        figure.savefig(
            temporary_path,
            dpi=160,
            bbox_inches="tight",
            format="png",
        )
        plt.close(figure)
        figure = None
        os.replace(temporary_path, output_path)
        temporary_path = None
    finally:
        if figure is not None:
            plt.close(figure)
        if temporary_path is not None and temporary_path.exists():
            temporary_path.unlink()


def plot_lcfs_only(
    gfile_data: "GFileData",
    filepath: str,
) -> None:
    """仅绘制 LCFS 边界曲线。

    参数：
        gfile_data: GFileData 对象，提供 LCFS 和磁轴坐标
        filepath:   输出 .png 文件路径
    """
    try:
        import matplotlib.pyplot as plt
    except ImportError as e:
        raise ImportError(
            "无法导入 matplotlib。请运行：pip install matplotlib"
            f"\n原始错误：{e}"
        ) from e

    fig, ax = plt.subplots(figsize=(6, 8))

    # 绘制 LCFS 边界曲线（红色实线，显示闭合）
    lcfs_r_disp, lcfs_z_disp = _closed_for_display(gfile_data.lcfs_r, gfile_data.lcfs_z)
    ax.plot(
        lcfs_r_disp,
        lcfs_z_disp,
        color="red",
        linewidth=1.5,
        label="LCFS",
    )

    # 若有第一壁数据，叠加绘制（蓝色虚线，显示闭合）
    if gfile_data.wall_r is not None and gfile_data.wall_z is not None:
        wall_r_disp, wall_z_disp = _closed_for_display(gfile_data.wall_r, gfile_data.wall_z)
        ax.plot(
            wall_r_disp,
            wall_z_disp,
            color="blue",
            linewidth=1.2,
            linestyle="--",
            label="First wall",
        )

    # 标注磁轴位置（红色叉号）
    ax.plot(
        gfile_data.rmaxis,
        gfile_data.zmaxis,
        marker="x",
        color="red",
        markersize=8,
        markeredgewidth=2,
        label=f"Magnetic axis ({gfile_data.rmaxis:.3f}, {gfile_data.zmaxis:.3f})",
    )

    ax.set_xlabel("R [m]", fontsize=11)
    ax.set_ylabel("Z [m]", fontsize=11)
    ax.set_title("LinkMesh LCFS Validation", fontsize=13)
    ax.set_aspect("equal")
    ax.legend(loc="upper right", fontsize=9)

    # 统计信息
    info_text = (
        f"LCFS points: {len(gfile_data.lcfs_r)}\n"
        f"Magnetic axis: R={gfile_data.rmaxis:.4f} m, Z={gfile_data.zmaxis:.4f} m"
    )
    ax.text(
        0.02, 0.02, info_text,
        transform=ax.transAxes,
        fontsize=8,
        verticalalignment="bottom",
        bbox=dict(boxstyle="round", facecolor="wheat", alpha=0.7),
    )

    fig.tight_layout()
    fig.savefig(filepath, dpi=150, bbox_inches="tight")
    plt.close(fig)
    print(f"LCFS validation plot saved: {filepath}")
    print(f"LCFS points: {len(gfile_data.lcfs_r)}, Magnetic axis: (R={gfile_data.rmaxis:.4f}, Z={gfile_data.zmaxis:.4f})")
