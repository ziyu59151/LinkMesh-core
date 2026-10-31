from __future__ import annotations

from pathlib import Path

import pytest

from linkmesh.config import (
    CliOverrides,
    ConfigError,
    RegionsGenerateOverrides,
    RegionsInspectOverrides,
    load_effective_config,
    load_regions_generate_config,
    load_regions_inspect_config,
)


def _load(tmp_path: Path, **overrides):
    return load_effective_config(
        CliOverrides(**overrides),
        no_config=True,
        cwd=tmp_path,
    )


def test_regions_source_cli_overrides_toml_source(tmp_path):
    config_path = tmp_path / "regions.toml"
    config_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'inspect'\n"
        "multi_array_file = 'old.rz'\n"
        "regions_map = 'map.toml'\n"
        "plot = 'plot.png'\n",
        encoding="utf-8",
    )

    config = load_regions_inspect_config(
        RegionsInspectOverrides(boundary_list="new.toml"),
        config_arg=str(config_path),
        cwd=tmp_path,
    )

    assert config.multi_array_file_path is None
    assert config.boundary_list_path == (tmp_path / "new.toml").resolve()


def test_regions_inspect_config_is_isolated_and_cli_overrides_toml(tmp_path):
    config_dir = tmp_path / "regions-config"
    config_dir.mkdir()
    config_path = config_dir / "regions.toml"
    config_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'inspect'\n"
        "multi_array_file = 'input.rz'\n"
        "regions_map = 'map.toml'\n"
        "plot = 'toml.png'\n"
        "[runtime]\nverbose = true\n",
        encoding="utf-8",
    )

    config = load_regions_inspect_config(
        RegionsInspectOverrides(plot="cli.png"),
        config_arg=str(config_path),
        cwd=tmp_path,
    )

    assert config.mode == "inspect"
    assert config.multi_array_file_path == (config_dir / "input.rz").resolve()
    assert config.regions_map_path == (config_dir / "map.toml").resolve()
    assert config.plot_path == (tmp_path / "cli.png").resolve()
    assert config.verbose is True


def test_regions_inspect_config_rejects_unknown_key_and_missing_required(
    tmp_path,
):
    path = tmp_path / "regions.toml"
    path.write_text(
        "schema_version = 1\n"
        "[regions]\nmode = 'inspect'\nunknown = true\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="未知键"):
        load_regions_inspect_config(
            RegionsInspectOverrides(),
            config_arg=str(path),
            cwd=tmp_path,
        )

    with pytest.raises(ConfigError, match="multi-array-file"):
        load_regions_inspect_config(
            RegionsInspectOverrides(),
            cwd=tmp_path,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "multi_array_file": "shared",
            "regions_map": "shared",
            "plot": "plot.png",
        },
        {
            "multi_array_file": "shared",
            "regions_map": "regions.toml",
            "plot": "shared",
        },
        {
            "multi_array_file": "input.rz",
            "regions_map": "shared",
            "plot": "shared",
        },
    ],
)
def test_regions_inspect_config_rejects_each_path_collision(
    tmp_path,
    overrides,
):
    with pytest.raises(ConfigError, match="路径冲突"):
        load_regions_inspect_config(
            RegionsInspectOverrides(**overrides),
            cwd=tmp_path,
        )



def test_regions_inspect_config_rejects_normalized_toml_cli_collision(
    tmp_path,
):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    shared_path = (config_dir / "shared").resolve()
    config_path = config_dir / "regions.toml"
    config_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'inspect'\n"
        "multi_array_file = 'shared'\n"
        "regions_map = 'map.toml'\n"
        "plot = 'plot.png'\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="multi_array_file.*regions_map"):
        load_regions_inspect_config(
            RegionsInspectOverrides(
                regions_map=str(shared_path),
            ),
            config_arg=str(config_path),
            cwd=tmp_path,
        )


def test_regions_generate_source_cli_overrides_toml_source(tmp_path):
    config_path = tmp_path / "regions.toml"
    config_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'generate'\n"
        "boundary_list = 'old.toml'\n"
        "regions_map = 'map.toml'\n"
        "plot = 'plot.png'\n"
        "[regions.mesh]\n"
        "mesh_dx = 0.08\n",
        encoding="utf-8",
    )

    config = load_regions_generate_config(
        RegionsGenerateOverrides(multi_array_file="new.rz"),
        config_arg=str(config_path),
        cwd=tmp_path,
    )

    assert config.boundary_list_path is None
    assert config.multi_array_file_path == (tmp_path / "new.rz").resolve()


def test_regions_generate_config_merges_toml_and_cli_region_dx(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "regions.toml"
    config_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'generate'\n"
        "multi_array_file = 'multi.rz'\n"
        "regions_map = 'regions-map.toml'\n"
        "plot = 'toml.png'\n"
        "[regions.mesh]\n"
        "mesh_dx = 0.08\n"
        "[regions.mesh.region_dx]\n"
        "1 = 0.03\n"
        "2 = 0.05\n",
        encoding="utf-8",
    )

    config = load_regions_generate_config(
        RegionsGenerateOverrides(
            target="trimeg",
            plot="cli.png",
            hdf5="cli.h5",
            mesh_dx=0.10,
            region_dx=["2=0.04", "3=0.02"],
        ),
        config_arg=str(config_path),
        cwd=tmp_path,
    )

    assert config.multi_array_file_path == (config_dir / "multi.rz").resolve()
    assert config.regions_map_path == (
        config_dir / "regions-map.toml"
    ).resolve()
    assert config.target == "trimeg"
    assert config.plot_path == (tmp_path / "cli.png").resolve()
    assert config.hdf5_path == (tmp_path / "cli.h5").resolve()
    assert config.mesh_dx == 0.10
    assert dict(config.region_dx) == {1: 0.03, 2: 0.04, 3: 0.02}
    assert config.transition_mode == "disabled"
    assert config.transition_width == 0.0
    assert config.transition_gradient is None
    assert config.local_refinement == ()


def test_regions_generate_transition_width_merges_and_validates(tmp_path):
    config_path = tmp_path / "regions.toml"
    config_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'generate'\n"
        "multi_array_file = 'multi.rz'\n"
        "regions_map = 'regions-map.toml'\n"
        "plot = 'mesh.png'\n"
        "[regions.mesh]\n"
        "mesh_dx = 0.08\n"
        "transition_width = 0.12\n",
        encoding="utf-8",
    )

    toml_config = load_regions_generate_config(
        RegionsGenerateOverrides(),
        config_arg=str(config_path),
        cwd=tmp_path,
    )
    assert toml_config.transition_width == 0.12

    cli_config = load_regions_generate_config(
        RegionsGenerateOverrides(transition_width=0.2),
        config_arg=str(config_path),
        cwd=tmp_path,
    )
    assert cli_config.transition_width == 0.2

    zero_config = load_regions_generate_config(
        RegionsGenerateOverrides(
            multi_array_file="multi.rz",
            regions_map="regions.toml",
            plot="mesh.png",
            mesh_dx=0.08,
            transition_width=0.0,
        ),
        cwd=tmp_path,
    )
    assert zero_config.transition_width == 0.0

    for invalid in (-0.1, float("nan"), float("inf")):
        with pytest.raises(ConfigError, match="transition_width"):
            load_regions_generate_config(
                RegionsGenerateOverrides(
                    multi_array_file="multi.rz",
                    regions_map="regions.toml",
                    plot="mesh.png",
                    mesh_dx=0.08,
                    transition_width=invalid,
                ),
                cwd=tmp_path,
            )


def test_regions_generate_transition_gradient_modes_and_overrides(tmp_path):
    gradient_path = tmp_path / "gradient.toml"
    gradient_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'generate'\n"
        "multi_array_file = 'multi.rz'\n"
        "regions_map = 'regions-map.toml'\n"
        "plot = 'mesh.png'\n"
        "[regions.mesh]\n"
        "mesh_dx = 0.08\n"
        "transition_gradient = 2.5\n",
        encoding="utf-8",
    )

    gradient = load_regions_generate_config(
        RegionsGenerateOverrides(),
        config_arg=str(gradient_path),
        cwd=tmp_path,
    )
    assert gradient.transition_mode == "gradient"
    assert gradient.transition_width is None
    assert gradient.transition_gradient == 2.5

    width_override = load_regions_generate_config(
        RegionsGenerateOverrides(transition_width=0.2),
        config_arg=str(gradient_path),
        cwd=tmp_path,
    )
    assert width_override.transition_mode == "width"
    assert width_override.transition_width == 0.2
    assert width_override.transition_gradient is None

    disabled = load_regions_generate_config(
        RegionsGenerateOverrides(transition_width=0.0),
        config_arg=str(gradient_path),
        cwd=tmp_path,
    )
    assert disabled.transition_mode == "disabled"
    assert disabled.transition_width == 0.0
    assert disabled.transition_gradient is None

    width_path = tmp_path / "width.toml"
    width_path.write_text(
        gradient_path.read_text(encoding="utf-8").replace(
            "transition_gradient = 2.5",
            "transition_width = 0.12",
        ),
        encoding="utf-8",
    )
    gradient_override = load_regions_generate_config(
        RegionsGenerateOverrides(transition_gradient=3.0),
        config_arg=str(width_path),
        cwd=tmp_path,
    )
    assert gradient_override.transition_mode == "gradient"
    assert gradient_override.transition_width is None
    assert gradient_override.transition_gradient == 3.0


@pytest.mark.parametrize(
    "value",
    (0.0, -0.1, float("nan"), float("inf")),
)
def test_regions_generate_rejects_invalid_transition_gradient(
    tmp_path,
    value,
):
    with pytest.raises(ConfigError, match="transition_gradient"):
        load_regions_generate_config(
            RegionsGenerateOverrides(
                multi_array_file="multi.rz",
                regions_map="regions.toml",
                plot="mesh.png",
                mesh_dx=0.08,
                transition_gradient=value,
            ),
            cwd=tmp_path,
        )


def test_regions_generate_rejects_toml_transition_mode_conflict(tmp_path):
    path = tmp_path / "regions.toml"
    path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'generate'\n"
        "multi_array_file = 'multi.rz'\n"
        "regions_map = 'regions-map.toml'\n"
        "plot = 'mesh.png'\n"
        "[regions.mesh]\n"
        "mesh_dx = 0.08\n"
        "transition_width = 0.0\n"
        "transition_gradient = 2.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="互斥"):
        load_regions_generate_config(
            RegionsGenerateOverrides(),
            config_arg=str(path),
            cwd=tmp_path,
        )


def test_regions_inspect_rejects_transition_width_toml_key(tmp_path):
    path = tmp_path / "regions.toml"
    path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'inspect'\n"
        "multi_array_file = 'multi.rz'\n"
        "regions_map = 'regions-map.toml'\n"
        "plot = 'mesh.png'\n"
        "[regions.mesh]\n"
        "transition_width = 0.1\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="未知"):
        load_regions_inspect_config(
            RegionsInspectOverrides(),
            config_arg=str(path),
            cwd=tmp_path,
        )


def test_regions_inspect_rejects_transition_gradient_toml_key(tmp_path):
    path = tmp_path / "regions.toml"
    path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'inspect'\n"
        "multi_array_file = 'multi.rz'\n"
        "regions_map = 'regions-map.toml'\n"
        "plot = 'mesh.png'\n"
        "[regions.mesh]\n"
        "transition_gradient = 2.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="未知"):
        load_regions_inspect_config(
            RegionsInspectOverrides(),
            config_arg=str(path),
            cwd=tmp_path,
        )


def test_root_config_rejects_transition_gradient(tmp_path):
    path = tmp_path / "linkmesh.in"
    path.write_text(
        "schema_version = 1\n"
        "[mesh]\n"
        "transition_gradient = 2.5\n",
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="未知键"):
        load_effective_config(CliOverrides(), cwd=tmp_path)


def test_local_refinement_toml_and_cli_list_override(tmp_path):
    root_path = tmp_path / "linkmesh.in"
    root_path.write_text(
        "schema_version = 1\n"
        "[input]\n"
        "gfile = 'eq.gfile'\n"
        "[[mesh.local_refinement]]\n"
        "r = 1.5\nz = -0.2\ndx = 0.015\nwidth = 0.25\n"
        "[[mesh.local_refinement]]\n"
        "r = 1.8\nz = 0.3\ndx = 0.02\nwidth = 0.30\n",
        encoding="utf-8",
    )
    root = load_effective_config(CliOverrides(), cwd=tmp_path)
    assert len(root.local_refinement) == 2
    assert root.local_refinement[0].z == -0.2

    overridden = load_effective_config(
        CliOverrides(local_refinement=["2.0,-0.4,0.01,0.2"]),
        cwd=tmp_path,
    )
    assert len(overridden.local_refinement) == 1
    assert overridden.local_refinement[0].r == 2.0

    regions_path = tmp_path / "regions-local.toml"
    regions_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'generate'\n"
        "multi_array_file = 'multi.rz'\n"
        "regions_map = 'regions-map.toml'\n"
        "plot = 'mesh.png'\n"
        "[regions.mesh]\n"
        "mesh_dx = 0.08\n"
        "[[regions.mesh.local_refinement]]\n"
        "r = 1.5\nz = 0.0\ndx = 0.015\nwidth = 0.25\n",
        encoding="utf-8",
    )
    regions = load_regions_generate_config(
        RegionsGenerateOverrides(
            local_refinement=[
                "2.1,-0.4,0.02,0.3",
                "1.7,0.2,0.01,0.2",
            ]
        ),
        config_arg=str(regions_path),
        cwd=tmp_path,
    )
    assert len(regions.local_refinement) == 2
    assert regions.local_refinement[0].z == -0.4


@pytest.mark.parametrize(
    "record",
    (
        "r = 1.0\nz = 0.0\ndx = 0.01\n",
        "r = 1.0\nz = 0.0\ndx = 0.01\nwidth = 0.2\nextra = 1\n",
        "r = nan\nz = 0.0\ndx = 0.01\nwidth = 0.2\n",
        "r = 1.0\nz = 0.0\ndx = 0.0\nwidth = 0.2\n",
        "r = 1.0\nz = 0.0\ndx = 0.01\nwidth = -0.2\n",
    ),
)
def test_root_rejects_invalid_local_refinement_record(tmp_path, record):
    path = tmp_path / "linkmesh.in"
    path.write_text(
        "schema_version = 1\n"
        "[input]\ngfile = 'eq.gfile'\n"
        "[[mesh.local_refinement]]\n"
        + record,
        encoding="utf-8",
    )
    with pytest.raises(ConfigError):
        load_effective_config(CliOverrides(), cwd=tmp_path)


def test_regions_inspect_rejects_local_refinement(tmp_path):
    path = tmp_path / "regions.toml"
    path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'inspect'\n"
        "multi_array_file = 'multi.rz'\n"
        "regions_map = 'regions-map.toml'\n"
        "plot = 'mesh.png'\n"
        "[[regions.mesh.local_refinement]]\n"
        "r = 1.0\nz = 0.0\ndx = 0.01\nwidth = 0.2\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="未知"):
        load_regions_inspect_config(
            RegionsInspectOverrides(),
            config_arg=str(path),
            cwd=tmp_path,
        )


def test_regions_generate_target_defaults_and_hdf5_contract(tmp_path):
    base = {
        "multi_array_file": "multi.rz",
        "regions_map": "regions.toml",
        "plot": "mesh.png",
        "mesh_dx": 0.08,
    }

    mesh_only = load_regions_generate_config(
        RegionsGenerateOverrides(**base),
        cwd=tmp_path,
    )
    assert mesh_only.target == "mesh_only"
    assert mesh_only.hdf5_path is None

    with pytest.raises(ConfigError, match="mesh_only.*禁止"):
        load_regions_generate_config(
            RegionsGenerateOverrides(**base, hdf5="mesh.h5"),
            cwd=tmp_path,
        )

    with pytest.raises(ConfigError, match="trimeg.*需要"):
        load_regions_generate_config(
            RegionsGenerateOverrides(**base, target="trimeg"),
            cwd=tmp_path,
        )

    trimeg = load_regions_generate_config(
        RegionsGenerateOverrides(
            **base,
            target="trimeg",
            hdf5="mesh.h5",
        ),
        cwd=tmp_path,
    )
    assert trimeg.target == "trimeg"
    assert trimeg.hdf5_path == (tmp_path / "mesh.h5").resolve()


@pytest.mark.parametrize("target", ["invalid", 1])
def test_regions_generate_rejects_unsupported_target(tmp_path, target):
    with pytest.raises(ConfigError, match="target"):
        load_regions_generate_config(
            RegionsGenerateOverrides(
                target=target,
                multi_array_file="multi.rz",
                regions_map="regions.toml",
                plot="mesh.png",
                mesh_dx=0.08,
            ),
            cwd=tmp_path,
        )


def test_regions_generate_m3dc1_requires_model_and_smb(tmp_path):
    base = {
        "target": "m3dc1",
        "multi_array_file": "multi.rz",
        "regions_map": "regions.toml",
        "plot": "mesh.png",
        "mesh_dx": 0.08,
    }

    with pytest.raises(ConfigError, match="model.*smb"):
        load_regions_generate_config(
            RegionsGenerateOverrides(**base),
            cwd=tmp_path,
        )

    config = load_regions_generate_config(
        RegionsGenerateOverrides(
            **base,
            model="model.txt",
            smb="mesh.smb",
        ),
        cwd=tmp_path,
    )
    assert config.target == "m3dc1"
    assert config.model_path == (tmp_path / "model.txt").resolve()
    assert config.smb_path == (tmp_path / "mesh.smb").resolve()
    assert config.hdf5_path is None

    with pytest.raises(ConfigError, match="禁止.*hdf5"):
        load_regions_generate_config(
            RegionsGenerateOverrides(
                **base,
                model="model.txt",
                smb="mesh.smb",
                hdf5="mesh.h5",
            ),
            cwd=tmp_path,
        )


def test_regions_generate_cli_overrides_toml_target_and_hdf5_bases(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "regions.toml"
    config_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'generate'\n"
        "target = 'trimeg'\n"
        "multi_array_file = 'multi.rz'\n"
        "regions_map = 'regions-map.toml'\n"
        "plot = 'mesh.png'\n"
        "hdf5 = 'toml.h5'\n"
        "[regions.mesh]\n"
        "mesh_dx = 0.08\n",
        encoding="utf-8",
    )

    toml_config = load_regions_generate_config(
        RegionsGenerateOverrides(),
        config_arg=str(config_path),
        cwd=tmp_path,
    )
    assert toml_config.hdf5_path == (config_dir / "toml.h5").resolve()

    cli_config = load_regions_generate_config(
        RegionsGenerateOverrides(
            target="trimeg",
            hdf5="cli.h5",
        ),
        config_arg=str(config_path),
        cwd=tmp_path,
    )
    assert cli_config.target == "trimeg"
    assert cli_config.hdf5_path == (tmp_path / "cli.h5").resolve()


@pytest.mark.parametrize(
    "hdf5, message",
    [
        ("multi.rz", "multi_array_file.*hdf5"),
        ("regions.toml", "regions_map.*hdf5"),
        ("mesh.png", "plot.*hdf5"),
    ],
)
def test_regions_generate_rejects_hdf5_path_collisions(
    tmp_path,
    hdf5,
    message,
):
    with pytest.raises(ConfigError, match=message):
        load_regions_generate_config(
            RegionsGenerateOverrides(
                target="trimeg",
                multi_array_file="multi.rz",
                regions_map="regions.toml",
                plot="mesh.png",
                hdf5=hdf5,
                mesh_dx=0.08,
            ),
            cwd=tmp_path,
        )


@pytest.mark.parametrize(
    "tokens, message",
    [
        (["1=0.03", "1=0.04"], "重复"),
        (["0=0.03"], "正整数"),
        (["1=nan"], "有限正数"),
        (["bad"], "ID=DX"),
    ],
)
def test_regions_generate_rejects_invalid_cli_region_dx(
    tmp_path,
    tokens,
    message,
):
    with pytest.raises(ConfigError, match=message):
        load_regions_generate_config(
            RegionsGenerateOverrides(
                multi_array_file="multi.rz",
                regions_map="regions.toml",
                plot="mesh.png",
                region_dx=tokens,
            ),
            cwd=tmp_path,
        )


def test_regions_generate_rejects_path_collision_and_missing_resolution(
    tmp_path,
):
    with pytest.raises(ConfigError, match="路径冲突"):
        load_regions_generate_config(
            RegionsGenerateOverrides(
                multi_array_file="shared",
                regions_map="regions.toml",
                plot="shared",
                mesh_dx=0.08,
            ),
            cwd=tmp_path,
        )

    with pytest.raises(ConfigError, match="mesh-dx|mesh_dx"):
        load_regions_generate_config(
            RegionsGenerateOverrides(
                multi_array_file="multi.rz",
                regions_map="regions.toml",
                plot="mesh.png",
            ),
            cwd=tmp_path,
        )


def test_builtin_defaults_and_target_conditional_output(tmp_path):
    config = _load(tmp_path, gfile="eq.gfile")

    assert config.target == "trimeg"
    assert config.boundary_source == "lcfs"
    assert config.mesh_dx == 0.08
    assert config.hdf5_path == (tmp_path / "mesh.h5").resolve()
    assert config.plot_path is None
    assert config.verbose is False
    assert config.plot_lcfs is False


def test_toml_cli_priority_and_distinct_path_bases(tmp_path):
    config_dir = tmp_path / "config"
    config_dir.mkdir()
    config_path = config_dir / "linkmesh.in"
    config_path.write_text(
        "schema_version = 1\n"
        "[task]\ntarget = 'trimeg'\n"
        "[input]\ngfile = 'toml.gfile'\n"
        "[domain]\nboundary_source = 'lcfs'\n"
        "[mesh]\nmesh_dx = 0.04\n"
        "[output]\nhdf5 = 'toml.h5'\n",
        encoding="utf-8",
    )

    config = load_effective_config(
        CliOverrides(gfile="cli.gfile", mesh_dx=0.06),
        config_arg=str(config_path),
        cwd=tmp_path,
    )

    assert config.gfile_path == (tmp_path / "cli.gfile").resolve()
    assert config.hdf5_path == (config_dir / "toml.h5").resolve()
    assert config.mesh_dx == 0.06


@pytest.mark.parametrize(
    "text, message",
    [
        ("[mesh]\nmesh_dx = 0.1\n", "schema_version"),
        ("schema_version = 2\n", "schema_version"),
        ("schema_version = 1\nunknown = 1\n", "未知顶层键"),
        ("schema_version = 1\n[mesh]\nmesh_dx = true\n", "mesh_dx"),
        ("schema_version = 1\n[output]\nplot_lcfs = 'yes'\n", "plot_lcfs"),
    ],
)
def test_strict_toml_rejects_distinct_schema_failures(tmp_path, text, message):
    path = tmp_path / "linkmesh.in"
    path.write_text(text, encoding="utf-8")

    with pytest.raises(ConfigError, match=message):
        load_effective_config(
            CliOverrides(gfile="eq.gfile"),
            cwd=tmp_path,
        )


@pytest.mark.parametrize(
    "overrides",
    [
        {
            "target": "mesh_only",
            "boundary_source": "lcfs",
            "gfile": "eq.gfile",
        },
        {
            "target": "trimeg",
            "boundary_source": "array_file",
            "array_file": "boundary",
        },
        {
            "target": "m3dc1",
            "boundary_source": "array_file",
            "array_file": "boundary",
            "model": "model.txt",
            "smb": "mesh.smb",
        },
    ],
)
def test_current_legal_workflows(tmp_path, overrides):
    config = _load(tmp_path, **overrides)
    assert not hasattr(config, "profile")


@pytest.mark.parametrize(
    "overrides, exception",
    [
        (
            {
                "boundary_source": "first_wall",
                "gfile": "eq.gfile",
            },
            NotImplementedError,
        ),
    ],
)
def test_forbidden_and_unimplemented_combinations(
    tmp_path,
    overrides,
    exception,
):
    with pytest.raises(exception):
        _load(tmp_path, **overrides)


@pytest.mark.parametrize(
    "overrides, message",
    [
        (
            {
                "boundary_source": "lcfs",
            },
            "gfile",
        ),
        (
            {
                "boundary_source": "array_file",
            },
            "array_file",
        ),
        (
            {
                "target": "m3dc1",
                "boundary_source": "lcfs",
                "gfile": "eq.gfile",
                "smb": "mesh.smb",
            },
            "model_path",
        ),
        (
            {
                "target": "mesh_only",
                "boundary_source": "array_file",
                "array_file": "boundary",
                "plot_lcfs": True,
                "plot": "mesh.png",
            },
            "gfile_path",
        ),
    ],
)
def test_required_fields(tmp_path, overrides, message):
    with pytest.raises(ConfigError, match=message):
        _load(tmp_path, **overrides)


def test_effective_config_has_exact_single_region_fields(tmp_path):
    config = _load(tmp_path, gfile="eq.gfile")

    assert tuple(config.__dataclass_fields__) == (
        "schema_version",
        "config_path",
        "target",
        "boundary_source",
        "gfile_path",
        "array_file_path",
        "mesh_dx",
        "hdf5_path",
        "plot_path",
        "quality_plot_path",
        "model_path",
        "smb_path",
        "verbose",
        "plot_lcfs",
        "local_refinement",
    )


@pytest.mark.parametrize(
    "removed_key",
    (
        "[domain]\nprofile = 'single_region'\n",
        "[domain]\nprofile = 'nested_three_region'\n",
        "[mesh]\nplasma_dx = 0.03\n",
        "[mesh]\nvacuum_dx = 0.04\n",
        "[mesh]\nouter_dx = 0.05\n",
    ),
)
def test_removed_root_toml_fields_are_rejected(tmp_path, removed_key):
    path = tmp_path / "linkmesh.in"
    path.write_text(
        "schema_version = 1\n" + removed_key,
        encoding="utf-8",
    )

    with pytest.raises(ConfigError, match="未知键"):
        load_effective_config(CliOverrides(), cwd=tmp_path)


def test_quality_plot_config_paths_defaults_overrides_and_collisions(tmp_path):
    root_path = tmp_path / "linkmesh.in"
    root_path.write_text(
        "schema_version = 1\n"
        "[input]\ngfile = 'eq.gfile'\n"
        "[output]\nplot = 'geometry.png'\n"
        "quality_plot = 'toml-quality.png'\n",
        encoding="utf-8",
    )
    root = load_effective_config(
        CliOverrides(quality_plot="cli-quality.png"),
        cwd=tmp_path,
    )
    assert root.quality_plot_path == (
        tmp_path / "cli-quality.png"
    ).resolve()

    default_root = _load(tmp_path, gfile="eq.gfile")
    assert default_root.quality_plot_path is None

    regions_path = tmp_path / "regions-quality.toml"
    regions_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'generate'\n"
        "multi_array_file = 'multi.rz'\n"
        "regions_map = 'regions.toml'\n"
        "quality_plot = 'quality.png'\n"
        "[regions.mesh]\nmesh_dx = 0.08\n",
        encoding="utf-8",
    )
    regions = load_regions_generate_config(
        RegionsGenerateOverrides(),
        config_arg=str(regions_path),
        cwd=tmp_path,
    )
    assert regions.plot_path is None
    assert regions.quality_plot_path == (tmp_path / "quality.png").resolve()

    with pytest.raises(ConfigError, match="路径冲突"):
        _load(
            tmp_path,
            gfile="eq.gfile",
            plot="same.png",
            quality_plot="./same.png",
        )

    with pytest.raises(ConfigError, match="路径冲突"):
        load_regions_generate_config(
            RegionsGenerateOverrides(
                multi_array_file="multi.rz",
                regions_map="regions.toml",
                plot="same.png",
                quality_plot="./same.png",
                mesh_dx=0.08,
            ),
            cwd=tmp_path,
        )


def test_regions_inspect_rejects_quality_plot(tmp_path):
    path = tmp_path / "inspect.toml"
    path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'inspect'\n"
        "multi_array_file = 'multi.rz'\n"
        "regions_map = 'regions.toml'\n"
        "plot = 'geometry.png'\n"
        "quality_plot = 'quality.png'\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="未知键"):
        load_regions_inspect_config(
            RegionsInspectOverrides(),
            config_arg=str(path),
            cwd=tmp_path,
        )


def test_root_mesh_dx_default_toml_and_cli_priority(tmp_path):
    path = tmp_path / "linkmesh.in"
    path.write_text(
        "schema_version = 1\n"
        "[input]\ngfile = 'eq.gfile'\n"
        "[domain]\nboundary_source = 'lcfs'\n"
        "[mesh]\nmesh_dx = 0.06\n",
        encoding="utf-8",
    )

    toml_config = load_effective_config(CliOverrides(), cwd=tmp_path)
    cli_config = load_effective_config(
        CliOverrides(mesh_dx=0.04),
        cwd=tmp_path,
    )

    assert toml_config.mesh_dx == 0.06
    assert cli_config.mesh_dx == 0.04


def test_regions_config_boundary_list_is_optional_and_preserves_old_source(
    tmp_path,
):
    inspect_config = load_regions_inspect_config(
        RegionsInspectOverrides(
            multi_array_file="multi.rz",
            regions_map="regions.toml",
            plot="inspect.png",
        ),
        cwd=tmp_path,
    )
    assert inspect_config.multi_array_file_path == (
        tmp_path / "multi.rz"
    ).resolve()
    assert inspect_config.boundary_list_path is None

    generate_config = load_regions_generate_config(
        RegionsGenerateOverrides(
            multi_array_file="multi.rz",
            regions_map="regions.toml",
            plot="mesh.png",
            mesh_dx=0.08,
        ),
        cwd=tmp_path,
    )
    assert generate_config.multi_array_file_path == (
        tmp_path / "multi.rz"
    ).resolve()
    assert generate_config.boundary_list_path is None


def test_regions_config_preserves_boundary_list_path_and_source_exclusivity(
    tmp_path,
):
    inspect_path = tmp_path / "inspect.toml"
    inspect_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'inspect'\n"
        "boundary_list = 'boundaries.toml'\n"
        "regions_map = 'regions.toml'\n"
        "plot = 'inspect.png'\n",
        encoding="utf-8",
    )
    inspect_config = load_regions_inspect_config(
        RegionsInspectOverrides(),
        config_arg=str(inspect_path),
        cwd=tmp_path,
    )
    assert inspect_config.multi_array_file_path is None
    assert inspect_config.boundary_list_path == (
        tmp_path / "boundaries.toml"
    ).resolve()

    generate_path = tmp_path / "generate.toml"
    generate_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'generate'\n"
        "boundary_list = 'boundaries.toml'\n"
        "regions_map = 'regions.toml'\n"
        "plot = 'mesh.png'\n"
        "[regions.mesh]\n"
        "mesh_dx = 0.08\n",
        encoding="utf-8",
    )
    generate_config = load_regions_generate_config(
        RegionsGenerateOverrides(),
        config_arg=str(generate_path),
        cwd=tmp_path,
    )
    assert generate_config.multi_array_file_path is None
    assert generate_config.boundary_list_path == (
        tmp_path / "boundaries.toml"
    ).resolve()

    conflict_path = tmp_path / "conflict.toml"
    conflict_path.write_text(
        "schema_version = 1\n"
        "[regions]\n"
        "mode = 'inspect'\n"
        "multi_array_file = 'multi.rz'\n"
        "boundary_list = 'boundaries.toml'\n"
        "regions_map = 'regions.toml'\n"
        "plot = 'inspect.png'\n",
        encoding="utf-8",
    )
    with pytest.raises(ConfigError, match="互斥"):
        load_regions_inspect_config(
            RegionsInspectOverrides(),
            config_arg=str(conflict_path),
            cwd=tmp_path,
        )
