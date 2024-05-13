from mid_end.typing import ModelSet
from mid_end.vision import resolve_relative_path

MODELS: ModelSet = {
    "face_detector": {
        "path": resolve_relative_path("../models/yoloface_8n.onnx"),
        "face_detector_size": (640, 640),
    },
    "face_recognizer": {
        "path": resolve_relative_path("../models/arcface_w600k_r50.onnx"),
        "template": "arcface_112_v2",
        "size": (112, 112),
    },
    "face_landmarker": {
        "path": resolve_relative_path("../models/2dfan4.onnx"),
    },
    "gender_age": {
        "path": resolve_relative_path("../models/gender_age.onnx"),
    },
    "face_parser": {
        "path": resolve_relative_path("../models/face_parser.onnx"),
        "size": (512, 512),
    },
    "inswapper": {
        "path": resolve_relative_path("../models/inswapper_128_fp16.onnx"),
        "template": "arcface_128_v2",
        "size": (128, 128),
        "mean": [0.0, 0.0, 0.0],
        "standard_deviation": [1.0, 1.0, 1.0],
    },
    "codeformer": {
        "path": resolve_relative_path("../models/codeformer.onnx"),
        "template": "ffhq_512",
        "size": (512, 512),
    },
    "real_esrgan": {
        "path": resolve_relative_path("../models/real_esrgan_x4_fp16.onnx"),
        "size": (128, 8, 2),
        "scale": 4,
    },
    "ddcolor": {
        "path": resolve_relative_path("../models/ddcolor.onnx"),
        "size": (512, 512),
    },
}
face_distance = 0.6
face_detector_score = 0.5
