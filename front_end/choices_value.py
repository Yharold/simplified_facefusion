from typing import Any, Dict, List, Literal
from mid_end.vision import resolve_relative_path

tool_choices: List[str] = ["人脸识别", "AI换脸", "高清放大", "颜色重建"]
preview_image_path: Dict[
    Literal["face_debugger", "face_swapper", "frame_enhancer", "frame_colorizer"], str
] = {
    "face_debugger": resolve_relative_path("../front_end/assets/face_debugger.png"),
    "face_swapper": resolve_relative_path("../front_end/assets/face_swapper.png"),
    "frame_enhancer": resolve_relative_path("../front_end/assets/frame_enhancer.png"),
    "frame_colorizer": resolve_relative_path("../front_end/assets/frame_colorizer.png"),
}
