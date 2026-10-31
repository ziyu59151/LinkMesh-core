from __future__ import annotations

import sys
from types import SimpleNamespace
from unittest.mock import MagicMock

import numpy as np
import pytest

from linkmesh import cli
from linkmesh.config import ConfigError, LocalRefinementConfig
from linkmesh.mesh import Mesh2D
from linkmesh.regions import RegionsError


def _parse(*arguments):
    return cli.build_parser().parse_args(list(arguments))


def _install_boundaries(monkeypatch):
    gfile = SimpleNamespace(
        lcfs_r=np.array([0.0, 1.0, 0.0]),
        lcfs_z=np.array([0.0, 0.0, 1.0]),
        wall_r=None,
        wall_z=None,
        rmaxis=0.5,
        zmaxis=0.5,
    )
    boundary = np.array(
        [[0.0, 0.0], [0.0, 1.0], [1.0, 1.0], [1.0, 0.0]]
    )
    mesh = SimpleNamespace(n_nodes=4, n_triangles=2)

    mocks = {
        "read_gfile": MagicMock(return_value=gfile),
        "read_boundary": MagicMock(return_value=boundary),
        "generate": MagicMock(return_value=mesh),
        "hdf5": MagicMock(),
        "build_spec": MagicMock(return_value=object()),
        "model": MagicMock(),
        "smb": MagicMock(),
        "plot": MagicMock(),
        "quality_plot": MagicMock(),
    }
    monkeypatch.setattr(
        "linkmesh.readers.gfile_reader.read_gfile",
        mocks["read_gfile"],
    )
    monkeypatch.setattr(
        "linkmesh.readers.boundary_reader.read_boundary_file",
        mocks["read_boundary"],
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.generate_mesh",
        mocks["generate"],
    )
    monkeypatch.setattr(
        "linkmesh.writers.hdf5_writer.write_hdf5",
        mocks["hdf5"],
    )
    monkeypatch.setattr(
        "linkmesh.m3dc1_model.build_single_face_model_spec",
        mocks["build_spec"],
    )
    monkeypatch.setattr(
        "linkmesh.writers.m3dc1_model_writer.write_m3dc1_model",
        mocks["model"],
    )
    monkeypatch.setattr(
        "linkmesh.writers.smb_writer.write_smb",
        mocks["smb"],
    )
    monkeypatch.setattr(
        "linkmesh.utils.plotter.plot_mesh",
        mocks["plot"],
    )
    monkeypatch.setattr(
        "linkmesh.utils.plotter.plot_triangle_shape_quality",
        mocks["quality_plot"],
    )
    mocks.update(gfile=gfile, boundary=boundary, mesh=mesh)
    return mocks


def _assert_group_order(help_text, *titles):
    positions = [help_text.index(f"{title}:") for title in titles]
    assert positions == sorted(positions)


def _usage_block(help_text):
    lines = help_text.splitlines()
    start = next(
        index for index, line in enumerate(lines)
        if line.startswith("usage:")
    )
    block = [lines[start]]
    for line in lines[start + 1:]:
        if not line.startswith(" "):
            break
        block.append(line)
    return "\n".join(block)


def _assert_help_contains(help_text, expected):
    normalized_help = " ".join(help_text.split())
    missing = [
        item
        for item in expected
        if item not in normalized_help
    ]
    assert not missing, f"help 缺少以下契约：{missing}"


def test_top_level_help_describes_both_workflows_and_group_order(capsys):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as captured:
        parser.parse_args(["--help"])

    assert captured.value.code == 0
    help_text = capsys.readouterr().out
    assert "单区域网格" in help_text
    assert "nested_three_region" not in help_text
    assert "--profile" not in help_text
    assert "--plasma-dx" not in help_text
    assert "--vacuum-dx" not in help_text
    assert "--outer-dx" not in help_text
    assert "多区域网格" in help_text
    assert "linkmesh regions inspect" in help_text
    assert "linkmesh regions generate" in help_text
    assert "regions {inspect,generate}" in help_text
    assert "linkmesh mesh" not in help_text
    assert "M4" not in help_text
    assert "schema_version = 1" in help_text
    for toml_path in (
        "[task]",
        "[input]",
        "[domain]",
        "[mesh]",
        "[[mesh.local_refinement]]",
        "[output]",
        "[runtime]",
        "TOML: output.plot_lcfs",
    ):
        assert toml_path in help_text
    assert "input.plot_lcfs" not in help_text
    assert "帮助分组仅用于阅读" in help_text
    for option in (
        "--config",
        "--no-config",
        "--target",
        "--boundary-source",
        "--gfile",
        "--array-file",
        "--plot-lcfs",
        "--mesh-dx",
        "--local-refinement",
        "--hdf5",
        "--h5",
        "--model",
        "--smb",
        "--plot",
        "--no-hdf5",
        "--no-smb",
        "--verbose",
    ):
        assert option in help_text
    _assert_group_order(
        help_text,
        "Configuration Loading",
        "Task ([task])",
        "Domain ([domain])",
        "Input ([input])",
        "Mesh ([mesh])",
        "Output ([output])",
        "Runtime ([runtime])",
    )


def test_regions_help_hierarchy_and_old_parser_scope(capsys):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as captured:
        parser.parse_args(["regions", "--help"])

    assert captured.value.code == 0
    help_text = capsys.readouterr().out
    assert "inspect" in help_text
    assert "generate" in help_text
    assert "setup-only" in help_text
    assert "不生成完整 Triangle 网格" in help_text
    assert "经用户确认的 regions map" in help_text
    usage = _usage_block(help_text)
    assert "linkmesh regions {inspect,generate} ..." in usage
    assert "regions {inspect,generate} ... regions" not in usage
    assert "--gfile" not in help_text
    assert "--hdf5" not in help_text
    assert "--smb" not in help_text
    assert "不同的严格 TOML schema" in help_text
    assert "schema_version = 1" in help_text
    assert "inspect -h" in help_text
    assert "generate -h" in help_text
    assert "[regions.mesh.region_dx]" not in help_text

    old = parser.parse_args(
        [
            "--no-config",
            "--target",
            "mesh_only",
            "--boundary-source",
            "lcfs",
            "--gfile",
            "eq.gfile",
        ]
    )
    assert old.command is None


@pytest.mark.parametrize(
    "removed_option",
    ("--profile", "--plasma-dx", "--vacuum-dx", "--outer-dx"),
)
def test_root_rejects_removed_legacy_options(removed_option):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as captured:
        parser.parse_args([removed_option, "legacy"])

    assert captured.value.code != 0


def test_root_rejects_artificial_outer_boundary_source():
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as captured:
        parser.parse_args(["--boundary-source", "artificial_outer"])

    assert captured.value.code != 0


def test_regions_inspect_help_is_scoped_and_grouped(capsys):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as captured:
        parser.parse_args(["regions", "inspect", "--help"])

    assert captured.value.code == 0
    help_text = capsys.readouterr().out
    _assert_help_contains(
        help_text,
        (
            "--config",
            "--multi-array-file",
            "--boundary-list",
            "--regions-map",
            "--plot",
            "--verbose",
            "不生成完整 Triangle 网格",
            "inspection PNG",
            "供 generate 使用",
            "schema_version = 1",
            "[regions]",
            'mode = "inspect"',
            "[runtime]",
            "TOML: regions.multi_array_file",
            "TOML: regions.regions_map",
            "TOML: regions.plot",
            "TOML: runtime.verbose",
        ),
    )
    usage = _usage_block(help_text)
    assert "linkmesh regions inspect [options]" in usage
    assert "regions inspect" not in usage.replace(
        "linkmesh regions inspect",
        "",
        1,
    )
    assert "--region-dx" not in help_text
    assert "--transition-width" not in help_text
    assert "--transition-gradient" not in help_text
    assert "--local-refinement" not in help_text
    assert "--hdf5" not in help_text
    assert "--smb" not in help_text
    _assert_group_order(
        help_text,
        "Configuration",
        "Input",
        "Output",
        "Runtime",
    )


def test_regions_cli_rejects_root_and_parent_boundary_list_scope():
    parser = cli.build_parser()

    with pytest.raises(SystemExit):
        parser.parse_args(["--boundary-list", "boundaries.toml"])

    with pytest.raises(SystemExit):
        parser.parse_args(["regions", "--boundary-list", "boundaries.toml"])

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "regions",
                "inspect",
                "--multi-array-file",
                "multi.rz",
                "--boundary-list",
                "boundaries.toml",
            ]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "regions",
                "generate",
                "--multi-array-file",
                "multi.rz",
                "--boundary-list",
                "boundaries.toml",
            ]
        )


def test_regions_inspect_runs_setup_only_map_and_plot(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    blocks = (
        np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=float),
        np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=float),
    )
    read = MagicMock(return_value=blocks)
    setup = MagicMock(return_value={1: 2, 2: 1})
    plot = MagicMock()
    write_map = MagicMock()
    monkeypatch.setattr(
        "linkmesh.readers.boundary_reader.read_multi_boundary_file",
        read,
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.inspect_multi_region_ids",
        setup,
    )
    monkeypatch.setattr(
        "linkmesh.utils.plotter.plot_regions_inspection",
        plot,
    )
    monkeypatch.setattr(
        "linkmesh.regions.write_regions_map",
        write_map,
    )

    cli._run(
        _parse(
            "regions",
            "inspect",
            "--multi-array-file",
            "multi.rz",
            "--regions-map",
            "regions.toml",
            "--plot",
            "regions.png",
        )
    )

    assert read.call_count == 1
    assert setup.call_count == 1
    assert plot.call_count == 1
    assert write_map.call_count == 1


def test_regions_generate_help_is_scoped(capsys):
    parser = cli.build_parser()

    with pytest.raises(SystemExit) as captured:
        parser.parse_args(["regions", "generate", "--help"])

    assert captured.value.code == 0
    help_text = capsys.readouterr().out
    _assert_help_contains(
        help_text,
        (
            "--config",
            "--target",
            "--multi-array-file",
            "--boundary-list",
            "--regions-map",
            "--mesh-dx",
            "--region-dx",
            "--transition-width",
            "--transition-gradient",
            "--local-refinement",
            "--hdf5",
            "--model",
            "--smb",
            "--plot",
            "--verbose",
            "{mesh_only,trimeg,m3dc1}",
            "schema_version = 1",
            'mode = "generate"',
            "[regions]",
            "[regions.mesh]",
            "[regions.mesh.region_dx]",
            "[[regions.mesh.local_refinement]]",
            "[runtime]",
            "TOML: regions.mesh.mesh_dx",
            "TOML: regions.mesh.region_dx",
            "TOML: regions.mesh.transition_width",
            "TOML: regions.mesh.transition_gradient",
            "TOML: [[regions.mesh.local_refinement]]",
            "覆盖所有最终 Region",
            "互斥",
            "1/m",
            "对数网格尺度梯度",
            "自动推导",
            "变化层厚度",
            "粗侧",
            "完整 CLI 列表就替换 TOML 列表而不是追加",
        ),
    )
    usage = _usage_block(help_text)
    assert "linkmesh regions generate [options]" in usage
    assert "regions generate" not in usage.replace(
        "linkmesh regions generate",
        "",
        1,
    )
    assert "M4" not in help_text
    assert "--gfile" not in help_text
    assert "--no-hdf5" not in help_text
    assert "--no-smb" not in help_text
    _assert_group_order(
        help_text,
        "Configuration",
        "Task",
        "Input",
        "Mesh Resolution",
        "Output",
        "Runtime",
    )


def test_quality_plot_cli_scope_and_help(capsys):
    parser = cli.build_parser()

    root = parser.parse_args(["--quality-plot", "quality.png"])
    assert root.quality_plot == "quality.png"

    generated = parser.parse_args(
        ["regions", "generate", "--quality-plot", "quality.png"]
    )
    assert generated.quality_plot == "quality.png"

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["regions", "inspect", "--quality-plot", "quality.png"]
        )

    with pytest.raises(SystemExit):
        parser.parse_args(["--help"])
    root_help = " ".join(capsys.readouterr().out.split())
    assert "TOML: output.quality_plot" in root_help
    assert "Q=0 红、Q=0.5 蓝、Q=1 绿" in root_help

    with pytest.raises(SystemExit):
        parser.parse_args(["regions", "generate", "--help"])
    generate_help = " ".join(capsys.readouterr().out.split())
    assert "TOML: regions.quality_plot" in generate_help
    assert "Q=0 红、Q=0.5 蓝、Q=1 绿" in generate_help


def test_local_refinement_cli_scope_repetition_and_format():
    parser = cli.build_parser()

    root = parser.parse_args(
        [
            "--local-refinement",
            "1.5,-0.4,0.015,0.25",
            "--local-refinement",
            "2.1,0.2,0.02,0.3",
        ]
    )
    assert root.local_refinement == [
        "1.5,-0.4,0.015,0.25",
        "2.1,0.2,0.02,0.3",
    ]

    regions = parser.parse_args(
        [
            "regions",
            "generate",
            "--local-refinement",
            "1.5,-0.4,0.015,0.25",
        ]
    )
    assert regions.local_refinement == ["1.5,-0.4,0.015,0.25"]

    with pytest.raises(SystemExit):
        parser.parse_args(
            ["regions", "inspect", "--local-refinement", "1,0,0.01,0.2"]
        )


def test_transition_gradient_scope_and_cli_mutual_exclusion():
    parser = cli.build_parser()

    parsed = parser.parse_args(
        [
            "regions",
            "generate",
            "--transition-gradient",
            "2.5",
        ]
    )
    assert parsed.transition_gradient == 2.5
    assert parsed.transition_width is cli.UNSET

    with pytest.raises(SystemExit):
        parser.parse_args(["--transition-gradient", "2.5"])
    with pytest.raises(SystemExit):
        parser.parse_args(
            ["regions", "inspect", "--transition-gradient", "2.5"]
        )
    with pytest.raises(SystemExit):
        parser.parse_args(
            [
                "regions",
                "generate",
                "--transition-width",
                "0.1",
                "--transition-gradient",
                "2.5",
            ]
        )


def test_regions_error_is_reported_without_traceback(
    monkeypatch,
    capsys,
):
    monkeypatch.setattr(
        sys,
        "argv",
        ["linkmesh", "regions", "generate", "--help"],
    )
    monkeypatch.setattr(
        cli,
        "_run",
        MagicMock(side_effect=RegionsError("请重新执行 linkmesh regions inspect")),
    )
    monkeypatch.setattr(
        cli,
        "build_parser",
        MagicMock(
            return_value=SimpleNamespace(
                parse_args=MagicMock(return_value=SimpleNamespace())
            )
        ),
    )

    with pytest.raises(SystemExit) as captured:
        cli.main()

    assert captured.value.code != 0
    stderr = capsys.readouterr().err
    assert "[LinkMesh] Region 工作流错误：" in stderr
    assert "请重新执行 linkmesh regions inspect" in stderr
    assert "Traceback" not in stderr


def test_unknown_internal_exception_is_not_swallowed(monkeypatch):
    monkeypatch.setattr(
        cli,
        "_run",
        MagicMock(side_effect=RuntimeError("internal failure")),
    )
    monkeypatch.setattr(
        cli,
        "build_parser",
        MagicMock(
            return_value=SimpleNamespace(
                parse_args=MagicMock(return_value=SimpleNamespace())
            )
        ),
    )

    with pytest.raises(RuntimeError, match="internal failure"):
        cli.main()


def test_regions_generate_rebuilds_validates_and_plots(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    blocks = (
        np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=float),
        np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=float),
    )
    records = (
        SimpleNamespace(
            region_id=1,
            curve_id=2,
            parent_region_id=2,
            child_region_ids=(),
            color="#D55E00",
            physical_label="core",
        ),
        SimpleNamespace(
            region_id=2,
            curve_id=1,
            parent_region_id=0,
            child_region_ids=(1,),
            color="#0072B2",
            physical_label="",
        ),
    )
    plan = object()
    numbered_plan = object()
    mesh = Mesh2D(
        nodes=np.array(
            [
                [0.0, 0.0],
                [1.0, 0.0],
                [1.0, 1.0],
                [0.0, 1.0],
                [0.5, 0.5],
            ],
            dtype=np.float64,
        ),
        triangles=np.array(
            [
                [0, 1, 4],
                [1, 2, 4],
                [2, 3, 4],
            ],
            dtype=np.int32,
        ),
        region_ids=np.array([1, 1, 2], dtype=np.int32),
    )

    read_map = MagicMock(return_value=records)
    read_blocks = MagicMock(return_value=blocks)
    build_plan = MagicMock(return_value=plan)
    setup = MagicMock(return_value={1: 2, 2: 1})
    assign = MagicMock(return_value=numbered_plan)
    validate = MagicMock()
    resolve = MagicMock(return_value={1: 0.03, 2: 0.08})
    generate = MagicMock(return_value=mesh)
    plot = MagicMock()
    hdf5 = MagicMock()

    monkeypatch.setattr("linkmesh.regions.read_regions_map", read_map)
    monkeypatch.setattr(
        "linkmesh.readers.boundary_reader.read_multi_boundary_file",
        read_blocks,
    )
    monkeypatch.setattr("linkmesh.regions.build_region_plan", build_plan)
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.inspect_multi_region_ids",
        setup,
    )
    monkeypatch.setattr(
        "linkmesh.regions.assign_final_region_ids",
        assign,
    )
    monkeypatch.setattr(
        "linkmesh.regions.validate_plan_against_map",
        validate,
    )
    monkeypatch.setattr("linkmesh.regions.resolve_region_dx", resolve)
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.generate_multi_region_mesh",
        generate,
    )
    monkeypatch.setattr(
        "linkmesh.utils.plotter.plot_regions_mesh",
        plot,
    )
    monkeypatch.setattr(
        "linkmesh.writers.hdf5_writer.write_hdf5",
        hdf5,
    )

    cli._run(
        _parse(
            "regions",
            "generate",
            "--multi-array-file",
            "multi.rz",
            "--regions-map",
            "regions.toml",
            "--plot",
            "mesh.png",
            "--mesh-dx",
            "0.08",
            "--region-dx",
            "1=0.03",
            "--verbose",
        )
    )

    validate.assert_called_once_with(numbered_plan, records)
    resolve.assert_called_once_with(records, 0.08, {1: 0.03})
    generate.assert_called_once_with(
        numbered_plan,
        {1: 0.03, 2: 0.08},
    )
    plot.assert_called_once()
    plotted_mesh, plotted_records, plotted_path = plot.call_args.args
    assert isinstance(plotted_mesh, Mesh2D)
    assert plotted_mesh is not mesh
    assert plotted_mesh.n_nodes == 5
    assert plotted_mesh.n_triangles == 3
    assert np.array_equal(plotted_mesh.nodes, mesh.nodes)
    assert np.array_equal(plotted_mesh.triangles, mesh.triangles)
    assert np.array_equal(plotted_mesh.region_ids, np.array([1, 1, 2]))
    assert plotted_records is records
    assert plotted_path == str((tmp_path / "mesh.png").resolve())
    assert hdf5.call_count == 0
    output = capsys.readouterr().out
    assert "Region 1" in output
    assert "triangles=2" in output
    assert "Region 2" in output


def test_regions_generate_transition_refines_once_and_reuses_final_mesh(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    blocks = (
        np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=float),
        np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=float),
    )
    records = (
        SimpleNamespace(
            region_id=1,
            curve_id=2,
            parent_region_id=2,
            child_region_ids=(),
            color="#D55E00",
            physical_label="",
        ),
        SimpleNamespace(
            region_id=2,
            curve_id=1,
            parent_region_id=0,
            child_region_ids=(1,),
            color="#0072B2",
            physical_label="",
        ),
    )
    initial_mesh = SimpleNamespace(
        n_nodes=5,
        n_triangles=3,
        region_ids=np.array([1, 1, 2], dtype=np.int32),
    )
    final_mesh = SimpleNamespace(
        n_nodes=7,
        n_triangles=6,
        region_ids=np.array([1, 1, 1, 2, 2, 2], dtype=np.int32),
    )
    generate = MagicMock(return_value=initial_mesh)
    diagnostics = SimpleNamespace(enabled=True)
    refine = MagicMock(
        return_value=SimpleNamespace(
            mesh=final_mesh,
            diagnostics=diagnostics,
        )
    )
    plot = MagicMock()
    quality_plot = MagicMock()
    quality_values = np.array(
        [0.7, 0.8, 0.9, 0.75, 0.85, 0.95],
        dtype=np.float64,
    )

    monkeypatch.setattr(
        "linkmesh.readers.boundary_reader.read_multi_boundary_file",
        MagicMock(return_value=blocks),
    )
    monkeypatch.setattr(
        "linkmesh.regions.read_regions_map",
        MagicMock(return_value=records),
    )
    monkeypatch.setattr(
        "linkmesh.regions.build_region_plan",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.inspect_multi_region_ids",
        MagicMock(return_value={1: 2, 2: 1}),
    )
    monkeypatch.setattr(
        "linkmesh.regions.assign_final_region_ids",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        "linkmesh.regions.validate_plan_against_map",
        MagicMock(),
    )
    monkeypatch.setattr(
        "linkmesh.regions.resolve_region_dx",
        MagicMock(return_value={1: 0.03, 2: 0.08}),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.generate_multi_region_mesh",
        generate,
    )
    monkeypatch.setattr(
        "linkmesh.mesher.triangle_refinement.refine_transition_mesh",
        refine,
    )
    monkeypatch.setattr(
        "linkmesh.utils.plotter.plot_regions_mesh",
        plot,
    )
    monkeypatch.setattr(
        "linkmesh.utils.plotter.plot_triangle_shape_quality",
        quality_plot,
    )
    monkeypatch.setattr(
        "linkmesh.mesh_quality.calculate_triangle_shape_quality",
        MagicMock(return_value=quality_values),
    )

    cli._run(
        _parse(
            "regions",
            "generate",
            "--multi-array-file",
            "multi.rz",
            "--regions-map",
            "regions.toml",
            "--plot",
            "mesh.png",
            "--mesh-dx",
            "0.08",
            "--region-dx",
            "1=0.03",
            "--transition-width",
            "0.12",
            "--local-refinement",
            "1.5,0.0,0.01,0.2",
            "--quality-plot",
            "quality.png",
        )
    )

    generate.assert_called_once()
    refine.assert_called_once_with(
        initial_mesh,
        {1: 0.03, 2: 0.08},
        0.12,
        transition_gradient=None,
        local_sources=(
            LocalRefinementConfig(
                r=1.5,
                z=0.0,
                dx=0.01,
                width=0.2,
            ),
        ),
    )
    plot.assert_called_once_with(
        final_mesh,
        records,
        str((tmp_path / "mesh.png").resolve()),
    )
    quality_plot.assert_called_once_with(
        final_mesh,
        quality_values,
        str((tmp_path / "quality.png").resolve()),
    )


def test_regions_generate_zero_transition_returns_disabled_result(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    blocks = (
        np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=float),
        np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=float),
    )
    records = (
        SimpleNamespace(
            region_id=1,
            curve_id=2,
            parent_region_id=2,
            child_region_ids=(),
            color="#D55E00",
            physical_label="",
        ),
        SimpleNamespace(
            region_id=2,
            curve_id=1,
            parent_region_id=0,
            child_region_ids=(1,),
            color="#0072B2",
            physical_label="",
        ),
    )
    mesh = SimpleNamespace(
        n_nodes=5,
        n_triangles=3,
        region_ids=np.array([1, 1, 2], dtype=np.int32),
    )
    refine = MagicMock(
        return_value=SimpleNamespace(
            mesh=mesh,
            diagnostics=SimpleNamespace(enabled=False),
        )
    )

    monkeypatch.setattr(
        "linkmesh.readers.boundary_reader.read_multi_boundary_file",
        MagicMock(return_value=blocks),
    )
    monkeypatch.setattr(
        "linkmesh.regions.read_regions_map",
        MagicMock(return_value=records),
    )
    monkeypatch.setattr(
        "linkmesh.regions.build_region_plan",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.inspect_multi_region_ids",
        MagicMock(return_value={1: 2, 2: 1}),
    )
    monkeypatch.setattr(
        "linkmesh.regions.assign_final_region_ids",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        "linkmesh.regions.validate_plan_against_map",
        MagicMock(),
    )
    monkeypatch.setattr(
        "linkmesh.regions.resolve_region_dx",
        MagicMock(return_value={1: 0.03, 2: 0.08}),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.generate_multi_region_mesh",
        MagicMock(return_value=mesh),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.triangle_refinement.refine_transition_mesh",
        refine,
    )
    plot = MagicMock()
    monkeypatch.setattr(
        "linkmesh.utils.plotter.plot_regions_mesh",
        plot,
    )

    cli._run(
        _parse(
            "regions",
            "generate",
            "--multi-array-file",
            "multi.rz",
            "--regions-map",
            "regions.toml",
            "--plot",
            "mesh.png",
            "--mesh-dx",
            "0.08",
        )
    )

    refine.assert_called_once_with(
        mesh,
        {1: 0.03, 2: 0.08},
        0.0,
        transition_gradient=None,
        local_sources=(),
    )
    plot.assert_called_once_with(
        mesh,
        records,
        str((tmp_path / "mesh.png").resolve()),
    )


def test_regions_generate_verbose_quality_uses_final_mesh(
    monkeypatch,
    tmp_path,
    capsys,
):
    monkeypatch.chdir(tmp_path)
    blocks = (
        np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=float),
        np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=float),
    )
    records = (
        SimpleNamespace(
            region_id=1,
            curve_id=2,
            parent_region_id=2,
            child_region_ids=(),
            color="#D55E00",
            physical_label="",
        ),
        SimpleNamespace(
            region_id=2,
            curve_id=1,
            parent_region_id=0,
            child_region_ids=(1,),
            color="#0072B2",
            physical_label="",
        ),
    )
    initial_mesh = SimpleNamespace(
        n_nodes=5,
        n_triangles=3,
        region_ids=np.array([1, 1, 2], dtype=np.int32),
    )
    final_mesh = SimpleNamespace(
        n_nodes=7,
        n_triangles=6,
        region_ids=np.array([1, 1, 1, 2, 2, 2], dtype=np.int32),
    )
    diagnostics = SimpleNamespace(
        enabled=True,
        transition_interface_count=1,
    )
    refinement_result = SimpleNamespace(
        mesh=final_mesh,
        diagnostics=diagnostics,
    )
    quality_report = SimpleNamespace(
        node_count=7,
        triangle_count=6,
        region_count=2,
        minimum_angle=20.0,
        minimum_angle_p01=21.0,
        minimum_angle_p05=22.0,
        maximum_angle=100.0,
        minimum_shape_quality=0.4,
        shape_quality_p01=0.45,
        shape_quality_p05=0.5,
        median_shape_quality=0.8,
        minimum_equivalent_size=0.02,
        median_equivalent_size=0.05,
        maximum_equivalent_size=0.1,
        maximum_adjacent_size_ratio=2.0,
        adjacent_size_ratio_p95=1.8,
        adjacent_size_ratio_p99=1.9,
        degenerate_triangle_count=0,
        duplicate_triangle_count=0,
        non_manifold_edge_count=0,
        refinement=SimpleNamespace(
            enabled=True,
            transition_mode="width",
            transition_width=0.12,
            transition_gradient=None,
            transition_interface_count=1,
            iteration_count=2,
            initial_node_count=5,
            initial_triangle_count=3,
            final_node_count=7,
            final_triangle_count=6,
            affected_coarse_triangle_count=3,
            final_maximum_area_excess=0.0,
            final_violating_triangle_count=0,
            interfaces=(
                SimpleNamespace(
                    region_ids=(1, 2),
                    fine_region_id=1,
                    coarse_region_id=2,
                    fine_dx=0.03,
                    coarse_dx=0.08,
                    transition_width=0.12,
                    affected_coarse_triangle_count=3,
                    maximum_area_excess=0.0,
                ),
            ),
            local_refinement_enabled=False,
            local_sources=(),
        ),
    )
    calculate = MagicMock(return_value=quality_report)
    plot = MagicMock()

    monkeypatch.setattr(
        "linkmesh.readers.boundary_reader.read_multi_boundary_file",
        MagicMock(return_value=blocks),
    )
    monkeypatch.setattr(
        "linkmesh.regions.read_regions_map",
        MagicMock(return_value=records),
    )
    monkeypatch.setattr(
        "linkmesh.regions.build_region_plan",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.inspect_multi_region_ids",
        MagicMock(return_value={1: 2, 2: 1}),
    )
    monkeypatch.setattr(
        "linkmesh.regions.assign_final_region_ids",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        "linkmesh.regions.validate_plan_against_map",
        MagicMock(),
    )
    monkeypatch.setattr(
        "linkmesh.regions.resolve_region_dx",
        MagicMock(return_value={1: 0.03, 2: 0.08}),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.generate_multi_region_mesh",
        MagicMock(return_value=initial_mesh),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.triangle_refinement.refine_transition_mesh",
        MagicMock(return_value=refinement_result),
    )
    monkeypatch.setattr(
        "linkmesh.mesh_quality.calculate_mesh_quality",
        calculate,
    )
    monkeypatch.setattr(
        "linkmesh.utils.plotter.plot_regions_mesh",
        plot,
    )

    cli._run(
        _parse(
            "regions",
            "generate",
            "--multi-array-file",
            "multi.rz",
            "--regions-map",
            "regions.toml",
            "--plot",
            "mesh.png",
            "--mesh-dx",
            "0.08",
            "--region-dx",
            "1=0.03",
            "--transition-width",
            "0.12",
            "--verbose",
        )
    )

    calculate.assert_called_once_with(final_mesh, diagnostics)
    plot.assert_called_once_with(
        final_mesh,
        records,
        str((tmp_path / "mesh.png").resolve()),
    )
    output = capsys.readouterr().out
    assert "Mesh quality:" in output
    assert "Transition quality:" in output
    assert "angle [degree]" in output
    assert "equivalent size [m]" in output


def test_gradient_verbose_prints_global_gradient_and_interface_width(capsys):
    report = SimpleNamespace(
        node_count=7,
        triangle_count=6,
        region_count=2,
        minimum_angle=20.0,
        minimum_angle_p01=21.0,
        minimum_angle_p05=22.0,
        maximum_angle=100.0,
        minimum_shape_quality=0.4,
        shape_quality_p01=0.45,
        shape_quality_p05=0.5,
        median_shape_quality=0.8,
        minimum_equivalent_size=0.02,
        median_equivalent_size=0.05,
        maximum_equivalent_size=0.1,
        maximum_adjacent_size_ratio=2.0,
        adjacent_size_ratio_p95=1.8,
        adjacent_size_ratio_p99=1.9,
        degenerate_triangle_count=0,
        duplicate_triangle_count=0,
        non_manifold_edge_count=0,
        refinement=SimpleNamespace(
            enabled=True,
            transition_mode="gradient",
            transition_width=None,
            transition_gradient=2.5,
            transition_interface_count=1,
            iteration_count=1,
            initial_node_count=5,
            initial_triangle_count=3,
            final_node_count=7,
            final_triangle_count=6,
            affected_coarse_triangle_count=3,
            final_maximum_area_excess=0.0,
            final_violating_triangle_count=0,
            interfaces=(
                SimpleNamespace(
                    region_ids=(1, 2),
                    fine_region_id=1,
                    coarse_region_id=2,
                    fine_dx=0.03,
                    coarse_dx=0.08,
                    transition_width=0.392331,
                    affected_coarse_triangle_count=3,
                    maximum_area_excess=0.0,
                ),
            ),
            local_refinement_enabled=False,
            local_sources=(),
        ),
    )

    cli._print_mesh_quality_report(report)

    output = capsys.readouterr().out
    assert "mode=gradient" in output
    assert "G [1/m]=2.5" in output
    assert "Wi=0.392331 m" in output
    assert "W [m]=" not in output


def test_regions_generate_without_verbose_skips_quality(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    blocks = (
        np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=float),
        np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=float),
    )
    records = (
        SimpleNamespace(
            region_id=1,
            curve_id=2,
            parent_region_id=2,
            child_region_ids=(),
            color="#D55E00",
            physical_label="",
        ),
        SimpleNamespace(
            region_id=2,
            curve_id=1,
            parent_region_id=0,
            child_region_ids=(1,),
            color="#0072B2",
            physical_label="",
        ),
    )
    mesh = SimpleNamespace(
        n_nodes=5,
        n_triangles=3,
        region_ids=np.array([1, 1, 2], dtype=np.int32),
    )
    calculate = MagicMock()

    monkeypatch.setattr(
        "linkmesh.readers.boundary_reader.read_multi_boundary_file",
        MagicMock(return_value=blocks),
    )
    monkeypatch.setattr(
        "linkmesh.regions.read_regions_map",
        MagicMock(return_value=records),
    )
    monkeypatch.setattr(
        "linkmesh.regions.build_region_plan",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.inspect_multi_region_ids",
        MagicMock(return_value={1: 2, 2: 1}),
    )
    monkeypatch.setattr(
        "linkmesh.regions.assign_final_region_ids",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        "linkmesh.regions.validate_plan_against_map",
        MagicMock(),
    )
    monkeypatch.setattr(
        "linkmesh.regions.resolve_region_dx",
        MagicMock(return_value={1: 0.03, 2: 0.08}),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.generate_multi_region_mesh",
        MagicMock(return_value=mesh),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.triangle_refinement.refine_transition_mesh",
        MagicMock(
            return_value=SimpleNamespace(
                mesh=mesh,
                diagnostics=SimpleNamespace(enabled=False),
            )
        ),
    )
    monkeypatch.setattr(
        "linkmesh.mesh_quality.calculate_mesh_quality",
        calculate,
    )
    monkeypatch.setattr(
        "linkmesh.utils.plotter.plot_regions_mesh",
        MagicMock(),
    )

    cli._run(
        _parse(
            "regions",
            "generate",
            "--multi-array-file",
            "multi.rz",
            "--regions-map",
            "regions.toml",
            "--plot",
            "mesh.png",
            "--mesh-dx",
            "0.08",
        )
    )

    assert calculate.call_count == 0


def test_regions_generate_trimeg_reuses_mesh_and_calls_hdf5_once(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    blocks = (
        np.array([[0, 0], [0, 3], [3, 3], [3, 0]], dtype=float),
        np.array([[1, 1], [1, 2], [2, 2], [2, 1]], dtype=float),
    )
    records = (
        SimpleNamespace(
            region_id=1,
            curve_id=2,
            parent_region_id=2,
            child_region_ids=(),
            color="#D55E00",
            physical_label="",
        ),
        SimpleNamespace(
            region_id=2,
            curve_id=1,
            parent_region_id=0,
            child_region_ids=(1,),
            color="#0072B2",
            physical_label="",
        ),
    )
    mesh = SimpleNamespace(
        n_nodes=5,
        n_triangles=3,
        region_ids=np.array([1, 1, 2], dtype=np.int32),
    )
    generate = MagicMock(return_value=mesh)
    hdf5 = MagicMock()

    monkeypatch.setattr(
        "linkmesh.readers.boundary_reader.read_multi_boundary_file",
        MagicMock(return_value=blocks),
    )
    monkeypatch.setattr(
        "linkmesh.regions.read_regions_map",
        MagicMock(return_value=records),
    )
    monkeypatch.setattr(
        "linkmesh.regions.build_region_plan",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.inspect_multi_region_ids",
        MagicMock(return_value={1: 2, 2: 1}),
    )
    monkeypatch.setattr(
        "linkmesh.regions.assign_final_region_ids",
        MagicMock(return_value=object()),
    )
    monkeypatch.setattr(
        "linkmesh.regions.validate_plan_against_map",
        MagicMock(),
    )
    monkeypatch.setattr(
        "linkmesh.regions.resolve_region_dx",
        MagicMock(return_value={1: 0.03, 2: 0.08}),
    )
    monkeypatch.setattr(
        "linkmesh.mesher.oft_adapter.generate_multi_region_mesh",
        generate,
    )
    monkeypatch.setattr(
        "linkmesh.mesher.triangle_refinement.refine_transition_mesh",
        MagicMock(
            return_value=SimpleNamespace(
                mesh=mesh,
                diagnostics=SimpleNamespace(enabled=False),
            )
        ),
    )
    monkeypatch.setattr(
        "linkmesh.writers.hdf5_writer.write_hdf5",
        hdf5,
    )
    monkeypatch.setattr(
        "linkmesh.utils.plotter.plot_regions_mesh",
        MagicMock(),
    )
    build_spec = MagicMock()
    model_writer = MagicMock()
    smb_writer = MagicMock()
    monkeypatch.setattr(
        "linkmesh.m3dc1_model.build_single_face_model_spec",
        build_spec,
    )
    monkeypatch.setattr(
        "linkmesh.writers.m3dc1_model_writer.write_m3dc1_model",
        model_writer,
    )
    monkeypatch.setattr(
        "linkmesh.writers.smb_writer.write_smb",
        smb_writer,
    )

    cli._run(
        _parse(
            "regions",
            "generate",
            "--target",
            "trimeg",
            "--multi-array-file",
            "multi.rz",
            "--regions-map",
            "regions.toml",
            "--plot",
            "mesh.png",
            "--mesh-dx",
            "0.08",
            "--hdf5",
            "mesh.h5",
        )
    )

    generate.assert_called_once()
    hdf5.assert_called_once_with(
        mesh,
        str((tmp_path / "mesh.h5").resolve()),
        verbose=False,
    )
    assert build_spec.call_count == 0
    assert model_writer.call_count == 0
    assert smb_writer.call_count == 0


def test_mesh_only_selects_mesher_without_writers(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mocks = _install_boundaries(monkeypatch)

    cli._run(
        _parse(
            "--no-config",
            "--target",
            "mesh_only",
            "--boundary-source",
            "lcfs",
            "--gfile",
            "eq.gfile",
        )
    )

    actual_boundary = mocks["generate"].call_args.args[0]
    expected_boundary = np.column_stack(
        (
            mocks["gfile"].lcfs_r,
            mocks["gfile"].lcfs_z,
        )
    )
    assert actual_boundary.shape == (3, 2)
    np.testing.assert_array_equal(actual_boundary, expected_boundary)
    assert mocks["hdf5"].call_count == 0
    assert mocks["model"].call_count == 0
    assert mocks["smb"].call_count == 0


def test_root_local_refinement_reuses_one_initial_and_final_mesh(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    mocks = _install_boundaries(monkeypatch)
    initial_mesh = Mesh2D(
        nodes=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        region_ids=np.array([7], dtype=np.int32),
    )
    final_mesh = SimpleNamespace(
        n_nodes=4,
        n_triangles=3,
        region_ids=np.array([7, 7, 7], dtype=np.int32),
    )
    mocks["generate"].return_value = initial_mesh
    refine = MagicMock(
        return_value=SimpleNamespace(
            mesh=final_mesh,
            diagnostics=SimpleNamespace(enabled=True),
        )
    )
    monkeypatch.setattr(
        "linkmesh.mesher.triangle_refinement.refine_transition_mesh",
        refine,
    )

    cli._run(
        _parse(
            "--no-config",
            "--target",
            "trimeg",
            "--boundary-source",
            "lcfs",
            "--gfile",
            "eq.gfile",
            "--hdf5",
            "mesh.h5",
            "--local-refinement",
            "0.2,0.2,0.01,0.2",
        )
    )

    mocks["generate"].assert_called_once()
    assert refine.call_args.args[:3] == (initial_mesh, {7: 0.08}, 0.0)
    assert len(refine.call_args.kwargs["local_sources"]) == 1
    assert mocks["hdf5"].call_args.args[0] is final_mesh


def test_trimeg_passes_same_mesh_to_hdf5_only(monkeypatch, tmp_path):
    monkeypatch.chdir(tmp_path)
    mocks = _install_boundaries(monkeypatch)

    cli._run(
        _parse(
            "--no-config",
            "--target",
            "trimeg",
            "--boundary-source",
            "lcfs",
            "--gfile",
            "eq.gfile",
            "--hdf5",
            "mesh.h5",
        )
    )

    assert mocks["hdf5"].call_args.args[0] is mocks["mesh"]
    assert mocks["build_spec"].call_count == 0
    assert mocks["smb"].call_count == 0


def test_m3dc1_reuses_one_model_spec_and_skips_hdf5(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    mocks = _install_boundaries(monkeypatch)

    cli._run(
        _parse(
            "--no-config",
            "--target",
            "m3dc1",
            "--boundary-source",
            "lcfs",
            "--gfile",
            "eq.gfile",
            "--model",
            "model.txt",
            "--smb",
            "mesh.smb",
        )
    )

    spec = mocks["build_spec"].return_value
    assert mocks["build_spec"].call_args.args[0] is mocks["mesh"]
    assert mocks["model"].call_args.args[0] is spec
    assert mocks["smb"].call_args.args[:2] == (mocks["mesh"], spec)
    assert mocks["hdf5"].call_count == 0


def test_array_lcfs_reference_never_enters_mesher_or_writer(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    mocks = _install_boundaries(monkeypatch)

    cli._run(
        _parse(
            "--no-config",
            "--target",
            "trimeg",
            "--boundary-source",
            "array_file",
            "--array-file",
            "boundary",
            "--plot-lcfs",
            "--gfile",
            "eq.gfile",
            "--plot",
            "mesh.png",
            "--hdf5",
            "mesh.h5",
        )
    )

    assert mocks["generate"].call_args.args[0] is mocks["boundary"]
    assert mocks["generate"].call_args.args[0] is not mocks["gfile"]
    assert mocks["hdf5"].call_args.args[0] is mocks["mesh"]
    assert mocks["plot"].call_args.args[:2] == (
        mocks["gfile"],
        mocks["mesh"],
    )
    assert mocks["plot"].call_args.kwargs["lcfs_reference_only"] is True


def test_root_quality_plot_uses_final_mesh_and_is_independent(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    mocks = _install_boundaries(monkeypatch)
    initial_mesh = Mesh2D(
        nodes=np.array(
            [[0.0, 0.0], [1.0, 0.0], [0.0, 1.0]],
            dtype=np.float64,
        ),
        triangles=np.array([[0, 1, 2]], dtype=np.int32),
        region_ids=np.array([7], dtype=np.int32),
    )
    final_mesh = Mesh2D(
        nodes=initial_mesh.nodes.copy(),
        triangles=initial_mesh.triangles.copy(),
        region_ids=initial_mesh.region_ids.copy(),
    )
    mocks["generate"].return_value = initial_mesh
    monkeypatch.setattr(
        "linkmesh.mesher.triangle_refinement.refine_transition_mesh",
        MagicMock(
            return_value=SimpleNamespace(
                mesh=final_mesh,
                diagnostics=SimpleNamespace(enabled=True),
            )
        ),
    )
    quality = np.array([0.75], dtype=np.float64)
    calculate = MagicMock(return_value=quality)
    monkeypatch.setattr(
        "linkmesh.mesh_quality.calculate_triangle_shape_quality",
        calculate,
    )

    cli._run(
        _parse(
            "--no-config",
            "--target",
            "mesh_only",
            "--boundary-source",
            "lcfs",
            "--gfile",
            "eq.gfile",
            "--local-refinement",
            "0.2,0.2,0.01,0.2",
            "--quality-plot",
            "quality.png",
        )
    )

    calculate.assert_called_once_with(final_mesh)
    mocks["quality_plot"].assert_called_once_with(
        final_mesh,
        quality,
        str((tmp_path / "quality.png").resolve()),
    )
    assert mocks["plot"].call_count == 0
    mocks["generate"].assert_called_once()


def test_configuration_failure_precedes_reader_and_mesher(
    monkeypatch,
    tmp_path,
):
    monkeypatch.chdir(tmp_path)
    mocks = _install_boundaries(monkeypatch)
    arguments = _parse(
        "--no-config",
        "--target",
        "mesh_only",
        "--boundary-source",
        "array_file",
        "--array-file",
        "boundary",
        "--plot-lcfs",
        "--plot",
        "mesh.png",
    )

    with pytest.raises(ConfigError, match="gfile_path"):
        cli._run(arguments)

    assert mocks["read_boundary"].call_count == 0
    assert mocks["read_gfile"].call_count == 0
    assert mocks["generate"].call_count == 0
