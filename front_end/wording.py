from typing import Any, Dict


WORDING: Dict[str, Any] = {
    "debugger": "人脸识别",
    "swapper": "AI换脸",
    "enhancer": "高清放大",
    "colorizer": "颜色重建",
    "debugger_describe": "debugger_describe",
    "swapper_describe": "swapper_describe",
    "enhancer_describe": "enhancer_describe",
    "colorizer_describe": "colorizer_describe",
    "preview_label": "效果预览",
}


def get(key: str) -> str:
    return WORDING[key]
