from typing import Any, List
from back_end.face_helper import paste_back, warp_face_by_face_landmark_5
from back_end.face_parser import clear_face_parser, create_mask
from mid_end.execution import get_providers
from mid_end.face_store import get_reference_faces
from mid_end.models import MODELS
from mid_end.typing import (
    Face,
    FaceSwapperInputs,
    ModelFrame,
    ProcessMode,
    QueuePayload,
    UpdateProgress,
    VisionFrame,
)
import numpy
import threading, time, onnx, onnxruntime
from back_end.face_analyser import (
    clear_face_analyser,
    find_similar_faces,
    get_average_face,
)
from mid_end.vision import (
    read_image,
    read_static_image,
    read_static_images,
    write_image,
)
from mid_end import process_manager
from onnx import numpy_helper
from back_end import core

"""
这个文件的主要目的是实现换脸功能
"""
FRAME_PROCESSOR = None
MODEL_INITIALIZER = None
THREAD_LOCK: threading.Lock = threading.Lock()


def get_frame_processor() -> None:
    global FRAME_PROCESSOR

    with THREAD_LOCK:
        while process_manager.is_checking():
            time.sleep(0.5)
        if FRAME_PROCESSOR is None:
            FRAME_PROCESSOR = onnxruntime.InferenceSession(
                MODELS.get("inswapper").get("path"),
                providers=get_providers(),
            )
    return FRAME_PROCESSOR


def clear_frame_processor() -> None:
    global FRAME_PROCESSOR
    FRAME_PROCESSOR = None


def get_model_initializer() -> Any:
    global MODEL_INITIALIZER

    with THREAD_LOCK:
        while process_manager.is_checking():
            time.sleep(0.5)
        if MODEL_INITIALIZER is None:
            model = onnx.load(MODELS.get("inswapper").get("path"))
            MODEL_INITIALIZER = numpy_helper.to_array(model.graph.initializer[-1])
    return MODEL_INITIALIZER


def clear_model_initializer() -> None:
    global MODEL_INITIALIZER

    MODEL_INITIALIZER = None


def pre_process(mode: ProcessMode) -> bool:
    return True


def post_process() -> None:
    read_static_image.cache_clear()
    clear_model_initializer()
    clear_frame_processor()
    clear_face_analyser()
    clear_face_parser()


def get_reference_frame():
    pass


def process_frame(inputs: FaceSwapperInputs) -> VisionFrame:
    reference_faces = inputs.get("reference_faces")
    source_face = inputs.get("source_face")
    target_vision_frame = inputs.get("target_vision_frame")
    similar_faces = find_similar_faces(reference_faces, target_vision_frame)
    if similar_faces:
        for similar_face in similar_faces:
            target_vision_frame = swap_face(
                source_face, similar_face, target_vision_frame
            )
    return target_vision_frame


def process_frames(
    source_paths: List[str],
    queue_payloads: List[QueuePayload],
    update_progress: UpdateProgress,
) -> None:
    reference_faces = get_reference_faces()
    source_frames = read_static_images(source_paths)
    source_face = get_average_face(source_frames)
    for queue_payload in process_manager.manage(queue_payloads):
        target_vision_path = queue_payload["frame_path"]
        target_vision_frame = read_image(target_vision_path)
        output_vision_frame = process_frame(
            {
                "reference_faces": reference_faces,
                "source_face": source_face,
                "target_vision_frame": target_vision_frame,
            }
        )
        write_image(target_vision_path, output_vision_frame)
        update_progress(1)


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    reference_faces = get_reference_faces()
    source_frames = read_static_images(source_paths)
    source_face = get_average_face(source_frames)
    target_vision_frame = read_static_image(target_path)
    output_vision_frame = process_frame(
        {
            "reference_faces": reference_faces,
            "source_face": source_face,
            "target_vision_frame": target_vision_frame,
        }
    )
    write_image(output_path, output_vision_frame)


def process_video(source_paths: List[str], temp_frame_paths: List[str]) -> None:
    core.multi_process_frames(source_paths, temp_frame_paths, process_frames)


def swap_face(
    source_face: Face, target_face: Face, temp_vision_frame: VisionFrame
) -> VisionFrame:
    model_template = MODELS.get("inswapper").get("template")
    model_size = MODELS.get("inswapper").get("size")
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
        temp_vision_frame, target_face.landmarks.get("5"), model_template, model_size
    )
    crop_vision_frame = prepare_crop_frame(crop_vision_frame)
    # 换脸
    frame_processor = get_frame_processor()
    frame_processor_inputs = {}
    for frame_processor_input in frame_processor.get_inputs():
        if frame_processor_input.name == "source":
            model_initializer = get_model_initializer()
            source_embedding = source_face.embedding.reshape((1, -1))
            frame_processor_inputs[frame_processor_input.name] = numpy.dot(
                source_embedding, model_initializer
            ) / numpy.linalg.norm(source_embedding)
        if frame_processor_input.name == "target":
            frame_processor_inputs[frame_processor_input.name] = crop_vision_frame
    crop_vision_frame = frame_processor.run(None, frame_processor_inputs)[0][0]
    # 数据处理
    crop_vision_frame = crop_vision_frame.transpose(1, 2, 0)
    crop_vision_frame = (crop_vision_frame * 255.0).round()
    crop_vision_frame = crop_vision_frame[:, :, ::-1]
    crop_mask = create_mask(crop_vision_frame)
    crop_mask = crop_mask.clip(0, 1)
    temp_vision_frame = paste_back(
        temp_vision_frame, crop_vision_frame, crop_mask, affine_matrix
    )
    return temp_vision_frame


def prepare_crop_frame(crop_vision_frame: VisionFrame) -> ModelFrame:
    model_mean = MODELS.get("inswapper").get("mean")
    model_standard_deviation = MODELS.get("inswapper").get("standard_deviation")
    crop_vision_frame = crop_vision_frame[:, :, ::-1] / 255.0
    crop_vision_frame = (crop_vision_frame - model_mean) / model_standard_deviation
    # 转置
    crop_vision_frame = crop_vision_frame.transpose(2, 0, 1)
    # 扩展维度
    crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0).astype(
        numpy.float32
    )
    return crop_vision_frame


