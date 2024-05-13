from mid_end.execution import get_providers
import threading, cv2, numpy, time, onnxruntime
from typing import Dict
from mid_end.models import MODELS
from mid_end.typing import FaceMaskRegion, Mask, VisionFrame
from mid_end import process_manager

FACE_PARSER = None
THREAD_LOCK: threading.Lock = threading.Lock()
FACE_MASK_REGIONS: Dict[FaceMaskRegion, int] = {
    "skin": 1,
    "left-eyebrow": 2,
    "right-eyebrow": 3,
    "left-eye": 4,
    "right-eye": 5,
    "eye-glasses": 6,
    "nose": 10,
    "mouth": 11,
    "upper-lip": 12,
    "lower-lip": 13,
}


def get_face_parser():
    global FACE_PARSER
    with THREAD_LOCK:
        while process_manager.is_checking():
            time.sleep(0.5)
        if FACE_PARSER is None:
            FACE_PARSER = onnxruntime.InferenceSession(
                MODELS.get("face_parser").get("path"), providers=get_providers()
            )
    return FACE_PARSER


def clear_face_parser():
    global FACE_PARSER
    FACE_PARSER = None


def create_mask(crop_vision_frame: VisionFrame) -> Mask:
    face_parser = get_face_parser()
    prepare_vision_frame = cv2.resize(
        crop_vision_frame, MODELS.get("face_parser").get("size")
    )
    prepare_vision_frame = prepare_vision_frame[:, :, ::-1] / 127.5 - 1
    prepare_vision_frame = numpy.expand_dims(
        prepare_vision_frame.transpose(2, 0, 1), axis=0
    ).astype(numpy.float32)
    mask: Mask = face_parser.run(
        None, {face_parser.get_inputs()[0].name: prepare_vision_frame}
    )[0][0]
    mask = numpy.isin(mask.argmax(0), [item for item in FACE_MASK_REGIONS.values()])
    mask = cv2.resize(mask.astype(numpy.float32), crop_vision_frame.shape[:2][::-1])
    mask = (cv2.GaussianBlur(mask.clip(0, 1), (0, 0), 5).clip(0.5, 1) - 0.5) * 2
    return mask
