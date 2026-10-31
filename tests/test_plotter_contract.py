from __future__ import annotations

import numpy as np
import pytest

pytest.importorskip("matplotlib")

from matplotlib.colors import to_rgba

from linkmesh.mesh import Mesh2D
from linkmesh.mesh_quality import (
    calculate_mesh_quality,
    calculate_triangle_shape_quality,
)
from linkmesh.regions import (
    RegionMapEntry,
    assign_final_region_ids,
    build_region_plan,
    region_map_entries,
)
from linkmesh.utils.plotter import (
    plot_mesh,
    plot_regions_inspection,
    plot_regions_mesh,
    plot_triangle_shape_quality,
)


def _mesh() -> Mesh2D:
    return Mesh2D(
        nodes=np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2], [0, 2, 3]], dtype=np.int32),
        region_ids=np.array([1, 1], dtype=np.int32),
        outer_boundary=np.array(
            [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]],
            dtype=np.float64,
        ),
    )


def test_triangle_shape_quality_math_and_report_share_values():
    mesh = Mesh2D(
        nodes=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [0.5, np.sqrt(3.0) / 2.0],
                [2.0, 0.0],
                [2.0, 1.0],
            ],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2], [1, 3, 4]], dtype=np.int32),
        region_ids=np.array([1, 1], dtype=np.int32),
    )
    quality = calculate_triangle_shape_quality(mesh)
    report = calculate_mesh_quality(mesh)

    assert quality.shape == (2,)
    assert quality.dtype == np.float64
    assert np.all(np.isfinite(quality))
    assert np.all((quality >= 0.0) & (quality <= 1.0))
    assert quality[0] == pytest.approx(1.0)
    assert quality[1] == pytest.approx(2.0 * np.sqrt(3.0) / 4.0)
    assert report.minimum_shape_quality == pytest.approx(np.min(quality))
    assert report.shape_quality_p01 == pytest.approx(
        np.percentile(quality, 1.0, method="linear")
    )
    assert report.shape_quality_p05 == pytest.approx(
        np.percentile(quality, 5.0, method="linear")
    )
    assert report.median_shape_quality == pytest.approx(np.median(quality))


def test_triangle_shape_quality_is_small_for_near_degenerate_valid_triangle():
    mesh = Mesh2D(
        nodes=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.5, 3.0e-12]],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        region_ids=np.array([1], dtype=np.int32),
    )

    quality = calculate_triangle_shape_quality(mesh)

    assert quality[0] > 0.0
    assert quality[0] < 1.0e-10
    assert np.isfinite(quality[0])


def test_plotter_generates_nonempty_png_without_gfile(tmp_path):
    output = tmp_path / "mesh.png"

    plot_mesh(None, _mesh(), str(output), show_region_ids=True)

    assert output.is_file()
    assert output.stat().st_size > 0


def test_quality_plot_fixed_colormap_normalization_and_inputs(
    tmp_path,
    monkeypatch,
):
    import matplotlib.axes
    import matplotlib.figure
    import matplotlib.pyplot as plt

    mesh = _mesh()
    quality = np.array([0.0, 1.0], dtype=np.float64)
    nodes_before = mesh.nodes.copy()
    quality_before = quality.copy()
    output = tmp_path / "quality.png"

    calls = []
    labels = []
    original_tripcolor = matplotlib.axes.Axes.tripcolor
    original_colorbar = matplotlib.figure.Figure.colorbar

    def recording_tripcolor(self, *args, **kwargs):
        calls.append(kwargs)
        return original_tripcolor(self, *args, **kwargs)

    def recording_colorbar(self, *args, **kwargs):
        colorbar = original_colorbar(self, *args, **kwargs)
        original_set_label = colorbar.set_label

        def recording_label(value, *label_args, **label_kwargs):
            labels.append(value)
            return original_set_label(
                value,
                *label_args,
                **label_kwargs,
            )

        colorbar.set_label = recording_label
        return colorbar

    monkeypatch.setattr(
        matplotlib.axes.Axes,
        "tripcolor",
        recording_tripcolor,
    )
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "colorbar",
        recording_colorbar,
    )

    plot_triangle_shape_quality(mesh, quality, str(output))

    assert output.stat().st_size > 0
    assert len(calls) == 1
    colormap = calls[0]["cmap"]
    norm = calls[0]["norm"]
    np.testing.assert_allclose(colormap(0.0), to_rgba("red"))
    np.testing.assert_allclose(
        colormap(0.5),
        to_rgba("blue"),
        atol=0.01,
    )
    np.testing.assert_allclose(colormap(1.0), to_rgba("green"))
    assert norm.vmin == 0.0
    assert norm.vmax == 1.0
    assert labels == ["Shape quality Q"]
    np.testing.assert_array_equal(mesh.nodes, nodes_before)
    np.testing.assert_array_equal(quality, quality_before)
    assert plt.get_fignums() == []


@pytest.mark.parametrize(
    "quality, message",
    [
        (np.array([[0.5, 0.5]]), "一维"),
        (np.array([0.5]), "一一对应"),
        (np.array([np.nan, 0.5]), "NaN/Inf"),
        (np.array([-0.1, 0.5]), "范围"),
        (np.array([0.5, 1.1]), "范围"),
    ],
)
def test_quality_plot_rejects_invalid_quality(quality, message, tmp_path):
    with pytest.raises(ValueError, match=message):
        plot_triangle_shape_quality(
            _mesh(),
            quality,
            str(tmp_path / "quality.png"),
        )


def test_regions_inspection_uses_figure_legend_and_no_axes_text(
    tmp_path,
    monkeypatch,
):
    outer = np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=float)
    inner = np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=float)
    plan = build_region_plan((outer, inner))
    numbered = assign_final_region_ids(plan, {1: 2, 2: 1})
    records = region_map_entries(numbered)
    snapshots = [entry.points.copy() for entry in numbered.entries]
    output = tmp_path / "regions.png"

    import matplotlib.axes
    import matplotlib.figure

    legend_calls = []
    original_legend = matplotlib.figure.Figure.legend

    def fail_axes_text(*_args, **_kwargs):
        pytest.fail("inspection PNG 不得在坐标轴内绘制文本标注")

    def recording_legend(self, *args, **kwargs):
        legend = original_legend(self, *args, **kwargs)
        legend_calls.append((legend, kwargs))
        return legend

    monkeypatch.setattr(matplotlib.axes.Axes, "text", fail_axes_text)
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "legend",
        recording_legend,
    )

    plot_regions_inspection(numbered, records, str(output))

    assert output.is_file()
    assert output.stat().st_size > 0
    assert len(legend_calls) == 1
    legend, legend_kwargs = legend_calls[0]
    assert legend.axes is None
    assert legend_kwargs["loc"] == "lower center"
    assert [text.get_text() for text in legend.get_texts()] == [
        "Region 1: curve 2, parent 2, children -",
        "Region 2: curve 1, parent 0, children 1",
    ]
    assert {record.color for record in records} == {
        entry.color for entry in numbered.entries
    }
    for entry, snapshot in zip(numbered.entries, snapshots):
        np.testing.assert_array_equal(entry.points, snapshot)


def test_regions_mesh_uses_map_colors_and_figure_legend(
    tmp_path,
    monkeypatch,
):
    mesh = Mesh2D(
        nodes=np.array(
            [
                [0.0, 0.0],
                [0.0, 2.0],
                [2.0, 2.0],
                [2.0, 0.0],
                [1.0, 1.0],
            ],
            dtype=np.float64,
        ),
        triangles=np.array(
            [[0, 1, 4], [1, 2, 4], [2, 3, 4], [3, 0, 4]],
            dtype=np.int32,
        ),
        region_ids=np.array([2, 1, 1, 2], dtype=np.int32),
        outer_boundary=np.array(
            [[0.0, 0.0], [0.0, 2.0], [2.0, 2.0], [2.0, 0.0]],
            dtype=np.float64,
        ),
    )
    records = (
        RegionMapEntry(1, 2, 2, (), "#D55E00", "core"),
        RegionMapEntry(2, 1, 0, (1,), "#0072B2", ""),
    )
    output = tmp_path / "generated.png"

    import matplotlib.axes
    import matplotlib.figure

    tripcolor_calls = []
    legend_calls = []
    original_tripcolor = matplotlib.axes.Axes.tripcolor
    original_legend = matplotlib.figure.Figure.legend

    def recording_tripcolor(self, *args, **kwargs):
        tripcolor_calls.append(kwargs)
        return original_tripcolor(self, *args, **kwargs)

    def recording_legend(self, *args, **kwargs):
        legend = original_legend(self, *args, **kwargs)
        legend_calls.append((legend, kwargs))
        return legend

    monkeypatch.setattr(
        matplotlib.axes.Axes,
        "tripcolor",
        recording_tripcolor,
    )
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "legend",
        recording_legend,
    )

    plot_regions_mesh(mesh, records, str(output))

    assert output.is_file()
    assert output.stat().st_size > 0
    assert len(tripcolor_calls) == 1
    assert list(tripcolor_calls[0]["cmap"].colors) == [
        "#D55E00",
        "#0072B2",
    ]
    assert len(legend_calls) == 1
    legend, kwargs = legend_calls[0]
    assert legend.axes is None
    assert kwargs["loc"] == "lower center"
    assert [text.get_text() for text in legend.get_texts()] == [
        "Region 1: curve 2, parent 2, children -, label core",
        "Region 2: curve 1, parent 0, children 1",
    ]


def test_regions_mesh_uses_no_axes_text_and_reports_figure_statistics(
    tmp_path,
    monkeypatch,
):
    import matplotlib.axes
    import matplotlib.figure

    figure_texts = []
    original_figure_text = matplotlib.figure.Figure.text

    def fail_axes_text(*_args, **_kwargs):
        pytest.fail("regions mesh PNG 不得在坐标轴内绘制文字")

    def recording_figure_text(self, *args, **kwargs):
        figure_texts.append(args[2])
        return original_figure_text(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "text", fail_axes_text)
    monkeypatch.setattr(
        matplotlib.figure.Figure,
        "text",
        recording_figure_text,
    )

    records = (
        RegionMapEntry(1, 2, 2, (), "#D55E00", ""),
        RegionMapEntry(2, 1, 0, (1,), "#0072B2", ""),
    )
    output = tmp_path / "statistics.png"
    plot_regions_mesh(
        Mesh2D(
            nodes=_mesh().nodes,
            triangles=_mesh().triangles,
            region_ids=np.array([1, 2], dtype=np.int32),
            outer_boundary=_mesh().outer_boundary,
        ),
        records,
        str(output),
    )

    assert output.stat().st_size > 0
    assert any(
        "Nodes: 4" in text
        and "Triangles: 2" in text
        and "Regions: 2" in text
        for text in figure_texts
    )


def test_regions_mesh_failure_does_not_replace_existing_output(
    tmp_path,
):
    output = tmp_path / "generated.png"
    output.write_bytes(b"existing")
    records = (
        RegionMapEntry(1, 2, 2, (), "#D55E00"),
        RegionMapEntry(2, 1, 0, (1,), "#0072B2"),
    )
    mesh = _mesh()

    with pytest.raises(ValueError, match="Region ID"):
        plot_regions_mesh(mesh, records, str(output))

    assert output.read_bytes() == b"existing"


def test_reference_only_mode_selects_lcfs_and_preserves_inputs(
    tmp_path,
    monkeypatch,
):
    class GFile:
        lcfs_r = np.array([0.1, 0.9, 0.9, 0.1])
        lcfs_z = np.array([0.1, 0.1, 0.9, 0.9])
        wall_r = np.array([-0.1, 1.1, 1.1, -0.1])
        wall_z = np.array([-0.1, -0.1, 1.1, 1.1])
        rmaxis = 0.5
        zmaxis = 0.5

    import matplotlib.axes

    labels = []
    original_plot = matplotlib.axes.Axes.plot

    def recording_plot(self, *args, **kwargs):
        if kwargs.get("label") is not None:
            labels.append(kwargs["label"])
        return original_plot(self, *args, **kwargs)

    monkeypatch.setattr(matplotlib.axes.Axes, "plot", recording_plot)

    mesh = _mesh()
    gfile = GFile()
    boundary_before = mesh.outer_boundary.copy()
    lcfs_before = gfile.lcfs_r.copy()
    output = tmp_path / "reference.png"

    plot_mesh(
        gfile,
        mesh,
        str(output),
        show_region_ids=True,
        lcfs_reference_only=True,
    )

    assert output.stat().st_size > 0
    assert "Mesh boundary" in labels
    assert "LCFS reference" in labels
    assert "First wall" not in labels
    assert not any(label.startswith("Magnetic axis") for label in labels)
    np.testing.assert_array_equal(mesh.outer_boundary, boundary_before)
    np.testing.assert_array_equal(gfile.lcfs_r, lcfs_before)
