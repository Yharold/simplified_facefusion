from typing import List, Literal


source_path = None
target_path = None
output_path = None

frame_processor: Literal[
    "face_debugger", "face_swapper", "frame_enhancer", "frame_colorizer"
] = "face_debugger"

reference_frame_number = 0
