"""
writers —— 静态网格翻译导出子包（两阶段战役核心）。

公开接口：
    write_hdf5(mesh: Mesh2D, filepath: str) -> None   # 第一战役：TRIMEG HDF5
    write_smb(mesh: Mesh2D, filepath: str) -> None    # 第二战役：M3DC1 PUMI .smb

开发顺序铁律：
    必须先完成 write_hdf5()（M3 里程碑）并验收通过，
    才能开始实现 write_smb()（M4 里程碑）。
"""
