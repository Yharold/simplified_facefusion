from typing import List
from back_end.face_analyser import (
    clear_face_analyser,
    find_similar_faces,
)
from back_end.face_helper import paste_back, warp_face_by_face_landmark_5
from back_end.face_parser import clear_face_parser, create_mask
from mid_end.execution import get_providers
from mid_end.face_store import get_reference_faces
from mid_end.typing import (
    Face,
    FaceEnhancerInputs,
    ProcessMode,
    QueuePayload,
    UpdateProgress,
    VisionFrame,
)
from mid_end.vision import (
    read_image,
    read_static_image,
    write_image,
)
import threading, time, onnxruntime, numpy, cv2
from mid_end import process_manager
from mid_end.models import MODELS
from back_end import core

"""
这个文件主要功能是面部增强
"""
FRAME_PROCESSOR = None
THREAD_LOCK: threading.Lock = threading.Lock()
THREAD_SEMAPHORE: threading.Semaphore = threading.Semaphore()


def get_frame_processor() -> None:
    global FRAME_PROCESSOR

    with THREAD_LOCK:
        while process_manager.is_checking():
            time.sleep(0.5)
        if FRAME_PROCESSOR is None:
            FRAME_PROCESSOR = onnxruntime.InferenceSession(
                MODELS.get("codeformer").get("path"),
                providers=get_providers(),
            )
    return FRAME_PROCESSOR


def clear_frame_processor() -> None:
    global FRAME_PROCESSOR
    FRAME_PROCESSOR = None


def pre_process(mode: ProcessMode) -> bool:
    return True


def post_process() -> None:
    read_static_image.cache_clear()
    clear_frame_processor()
    clear_face_analyser()
    clear_face_parser()


def get_reference_frame():
    pass


def process_frame(inputs: FaceEnhancerInputs) -> VisionFrame:
    reference_faces = inputs.get("reference_faces")
    target_vision_frame = inputs.get("target_vision_frame")
    similar_faces = find_similar_faces(reference_faces, target_vision_frame)
    if similar_faces:
        for similar_face in similar_faces:
            target_vision_frame = enhance_face(similar_face, target_vision_frame)
    return target_vision_frame


def process_frames(
    source_paths: List[str],
    queue_payloads: List[QueuePayload],
    update_progress: UpdateProgress,
) -> None:
    reference_faces = get_reference_faces()
    for queue_payload in process_manager.manage(queue_payloads):
        target_vision_path = queue_payload["frame_path"]
        target_vision_frame = read_image(target_vision_path)
        output_vision_frame = process_frame(
            {
                "reference_faces": reference_faces,
                "target_vision_frame": target_vision_frame,
            }
        )
        write_image(target_vision_path, output_vision_frame)
        update_progress(1)


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    reference_faces = get_reference_faces()
    target_vision_frame = read_static_image(target_path)
    output_vision_frame = process_frame(
        {
            "reference_faces": reference_faces,
            "target_vision_frame": target_vision_frame,
        }
    )
    write_image(output_path, output_vision_frame)


def process_video(source_paths: List[str], temp_frame_paths: List[str]) -> None:
    core.multi_process_frames(source_paths, temp_frame_paths, process_frames)


def enhance_face(target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
    model_template = MODELS.get("codeformer").get("template")
    model_size = MODELS.get("codeformer").get("size")
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
        temp_vision_frame, target_face.landmarks.get("5"), model_template, model_size
    )
    # 预处理
    crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
    crop_vision_frame = (crop_vision_frame - 0.5) / 0.5
    crop_vision_frame = numpy.expand_dims(
        crop_vision_frame.transpose(2, 0, 1), axis=0
    ).astype(numpy.float32)
    # 应用模型
    frame_processor = get_frame_processor()
    frame_processor_inputs = {}
    for frame_processor_input in frame_processor.get_inputs():
        if frame_processor_input.name == "input":
            frame_processor_inputs[frame_processor_input.name] = crop_vision_frame
        if frame_processor_input.name == "weight":
            weight = numpy.array([1]).astype(numpy.double)
            frame_processor_inputs[frame_processor_input.name] = weight
    with THREAD_SEMAPHORE:
        crop_vision_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
    # 后处理
    crop_vision_frame = numpy.clip(crop_vision_frame, -1, 1)
    crop_vision_frame = (crop_vision_frame + 1) / 2
    crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
    crop_vision_frame = (crop_vision_frame * 255.0).round()
    crop_vision_frame = crop_vision_frame.astype(numpy.uint8)[:, :, ::-1]
    # 面部遮罩
    crop_mask = create_mask(crop_vision_frame)
    crop_mask = crop_mask.clip(0, 1)
    paste_vision_frame = paste_back(
        temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix
    )
    # 混合
    face_enhancer_blend = 1 - (80 / 100)
    temp_vision_frame = cv2.addWeighted(
        temp_vision_frame,
        face_enhancer_blend,
        paste_vision_frame,
        1 - face_enhancer_blend,
        0,
    )
    return temp_vision_frame
