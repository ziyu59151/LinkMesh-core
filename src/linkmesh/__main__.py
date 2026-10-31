"""
__main__.py —— python -m linkmesh 入口。

职责：
    - 使 `python -m linkmesh` 调用进入 linkmesh.cli.main
    - 不承载任何参数解析或业务逻辑

"""

from linkmesh.cli import main

if __name__ == "__main__":
    main()
