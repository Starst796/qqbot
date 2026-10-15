"""NoneBot 插件包。

`nonebot.load_plugins("plugins")` 会加载本目录下的全部子插件；
显式声明为包，保证 `from plugins.rag import retrieve` 这类跨插件引用稳定可用。
"""
