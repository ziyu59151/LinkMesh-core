"""
LinkMesh —— 轻量级托卡马克二维非结构网格转换命令行工具。

包结构：
    readers/   : GFile/EQDSK 平衡文件读取
    mesher/    : TokaMaker/Triangle 唯一底座封装
    writers/   : TRIMEG HDF5 与 M3DC1 PUMI .smb 静态网格写出
    utils/     : matplotlib 可视化自检工具
"""

__version__ = "0.2.0"
