from collections import namedtuple
from typing import IO, Any, Callable, Dict, Literal, Tuple, TypedDict, List
import numpy

# face_analyser
Resolution = Tuple[int, int]
Matrix = numpy.ndarray[Any, Any]
Translation = numpy.ndarray[Any, Any]
ModelValue = Dict[str, Any]
ModelSet = Dict[str, ModelValue]
BoundingBox = numpy.ndarray[Any, Any]
FaceLandmark5 = numpy.ndarray[Any, Any]
FaceLandmark68 = numpy.ndarray[Any, Any]
FaceLandmarkSet = TypedDict(
    "FaceLandmarkSet", {"5": FaceLandmark5, "68": FaceLandmark68}
)
Score = float
FaceScoreSet = TypedDict("FaceScoreSet", {"detector": Score, "landmarker": Score})
Embedding = numpy.ndarray[Any, Any]

Face = namedtuple(
    "Face",
    [
        "bounding_box",
        "landmarks",
        "scores",
        "embedding",
        "normed_embedding",
        "gender",
        "age",
    ],
)
FaceSet = Dict[str, List[Face]]
FaceStore = TypedDict(
    "FaceStore", {"static_faces": FaceSet, "reference_faces": FaceSet}
)
FaceMaskRegion = Literal[
    "skin",
    "left-eyebrow",
    "right-eyebrow",
    "left-eye",
    "right-eye",
    "eye-glasses",
    "nose",
    "mouth",
    "upper-lip",
    "lower-lip",
]
VisionFrame = numpy.ndarray[Any, Any]
Mask = numpy.ndarray[Any, Any]
ModelFrame = numpy.ndarray[Any, Any]
# execution
ValueAndUnit = TypedDict(
    "ValueAndUnit",
    {
        "value": str,
        "unit": str,
    },
)
ExecutionDeviceFramework = TypedDict(
    "ExecutionDeviceFramework",
    {
        "name": str,
        "version": str,
    },
)
ExecutionDeviceProduct = TypedDict(
    "ExecutionDeviceProduct",
    {
        "vendor": str,
        "name": str,
        "architecture": str,
    },
)
ExecutionDeviceVideoMemory = TypedDict(
    "ExecutionDeviceVideoMemory",
    {
        "total": ValueAndUnit,
        "free": ValueAndUnit,
    },
)
ExecutionDeviceUtilization = TypedDict(
    "ExecutionDeviceUtilization",
    {
        "gpu": ValueAndUnit,
        "memory": ValueAndUnit,
    },
)

ExecutionDevice = TypedDict(
    "ExecutionDevice",
    {
        "driver_version": str,
        "framework": ExecutionDeviceFramework,
        "product": ExecutionDeviceProduct,
        "video_memory": ExecutionDeviceVideoMemory,
        "utilization": ExecutionDeviceUtilization,
    },
)
# face_helper
WarpTemplate = Literal["arcface_112_v1", "arcface_112_v2", "arcface_128_v2", "ffhq_512"]
WarpTemplateSet = Dict[WarpTemplate, numpy.ndarray[Any, Any]]
# back_end.core
QueuePayload = TypedDict("QueuePayload", {"frame_number": int, "frame_path": str})
UpdateProgress = Callable[[int], None]
ProcessFrames = Callable[[List[str], List[QueuePayload], UpdateProgress], None]
# logger
LogLevel = Literal["error", "warn", "info", "debug"]
# process_manager
ProcessState = Literal["checking", "processing", "stopping", "pending"]


# face_debug
FaceDebuggerInputs = TypedDict(
    "FaceDebuggerInputs",
    {"target_vision_frame": VisionFrame},
)
# FaceDebuggerInputs = TypedDict(
#     "FaceDebuggerInputs",
#     {"reference_faces": FaceSet, "target_vision_frame": VisionFrame},
# )
# face_swapper
FaceSwapperInputs = TypedDict(
    "FaceSwapperInputs",
    {
        "reference_faces": FaceSet,
        "source_face": Face,
        "target_vision_frame": VisionFrame,
    },
)
ProcessMode = Literal["output", "preview", "stream"]
FaceEnhancerInputs = TypedDict(
    "FaceEnhancerInputs",
    {"reference_faces": FaceSet, "target_vision_frame": VisionFrame},
)
FrameEnhancerInputs = TypedDict(
    "FrameEnhancerInputs", {"target_vision_frame": VisionFrame}
)
FrameColorizerInputs = TypedDict(
    "FrameColorizerInputs", {"target_vision_frame": VisionFrame}
)
# ffmpeg
Fps = float

File = IO[Any]
