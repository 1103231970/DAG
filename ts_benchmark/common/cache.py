# common_cache.py
from typing import Any, Optional
from enum import Enum, unique


@unique  # 确保枚举值唯一
class CacheKey(Enum):
    """缓存键名枚举类，统一管理所有缓存键"""
    DATASET_NAME = "dataset_name"
    DATASET_NAME_LIST = "dataset_name_list"


class GlobalCache:
    """通用内存缓存类，支持设置、获取、删除任意键值对缓存"""
    def __init__(self):
        self._cache = {}  # 存储缓存的字典

    def set(self, key: str, value: Any) -> None:
        """设置缓存"""
        self._cache[key] = value

    def get(self, key: str, default: Optional[Any] = None) -> Any:
        """获取        获取缓存
        :param key: 缓存键名
        :param default: 键不存在时的默认返回值
        """
        return self._cache.get(key, default)

    def delete(self, key: str) -> None:
        """删除指定缓存"""
        if key in self._cache:
            del self._cache[key]

    def clear(self) -> None:
        """清空所有缓存"""
        self._cache.clear()

    def has(self, key: str) -> bool:
        """判断缓存是否存在"""
        return key in self._cache


# 实例化全局缓存对象，供其他模块直接使用
cache = GlobalCache()