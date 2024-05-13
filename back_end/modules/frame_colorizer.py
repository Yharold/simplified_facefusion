from mid_end.execution import get_providers
from mid_end.models import MODELS
from mid_end.typing import (
    FrameColorizerInputs,
    QueuePayload,
    UpdateProgress,
    VisionFrame,
)
import threading, time, onnxruntime, numpy, cv2
from typing import Any, List
from back_end.face_analyser import clear_face_analyser
from mid_end.vision import (
    read_image,
    read_static_image,
    write_image,
)
from mid_end import process_manager
from back_end import core

"""
这个文件的功能是重上色
"""
FRAME_PROCESSOR = None
THREAD_LOCK: threading.Lock = threading.Lock()
THREAD_SEMAPHORE: threading.Semaphore = threading.Semaphore()


def get_frame_processor() -> Any:
    global FRAME_PROCESSOR

    with THREAD_LOCK:
        while process_manager.is_checking():
            time.sleep(0.5)
        if FRAME_PROCESSOR is None:
            FRAME_PROCESSOR = onnxruntime.InferenceSession(
                MODELS.get("ddcolor").get("path"),
                providers=get_providers(),
            )
    return FRAME_PROCESSOR


def clear_frame_processor() -> None:
    global FRAME_PROCESSOR
    FRAME_PROCESSOR = None


def pre_process() -> bool:
    return True


def post_process() -> None:
    read_static_image.cache_clear()
    clear_frame_processor()
    clear_face_analyser()


def get_reference_frame():
    pass


def process_frame(inputs: FrameColorizerInputs) -> VisionFrame:
    target_vision_frame = inputs.get("target_vision_frame")
    return colorize_frame(target_vision_frame)


def process_frames(
    source_paths: List[str],
    queue_payloads: List[QueuePayload],
    update_progress: UpdateProgress,
) -> None:
    for queue_payload in process_manager.manage(queue_payloads):
        target_vision_path = queue_payload["frame_path"]
        target_vision_frame = read_image(target_vision_path)
        output_vision_frame = process_frame(
            {
                "target_vision_frame": target_vision_frame,
            }
        )
        write_image(target_vision_path, output_vision_frame)
        update_progress(1)


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    target_vision_frame = read_static_image(target_path)
    output_vision_frame = process_frame({"target_vision_frame": target_vision_frame})
    write_image(output_path, output_vision_frame)


def process_video(source_paths: List[str], temp_frame_paths: List[str]) -> None:
    core.multi_process_frames(None, temp_frame_paths, process_frames)


def colorize_frame(temp_vision_frame: VisionFrame) -> VisionFrame:
    frame_processor = get_frame_processor()
    # 预处理
    prepare_vision_frame = prepare_temp_frame(temp_vision_frame)
    with THREAD_SEMAPHORE:
        color_vision_frame = frame_processor.run(
            None, {frame_processor.get_inputs()[0].name: prepare_vision_frame}
        )[0][0]
    # 后处理color_vision_frame(2,height,width)
    color_vision_frame = color_vision_frame.transpose(1, 2, 0)
    color_vision_frame = cv2.resize(
        color_vision_frame, (temp_vision_frame.shape[1], temp_vision_frame.shape[0])
    )
    temp_vision_frame = (temp_vision_frame / 255.0).astype(numpy.float32)
    # 只保留原始照片明度信息
    temp_vision_frame = cv2.cvtColor(temp_vision_frame, cv2.COLOR_BGR2LAB)[:, :, :1]
    # 明度+颜色构成完整的LAB格式
    color_vision_frame = numpy.concatenate(
        (temp_vision_frame, color_vision_frame), axis=-1
    )
    # LAB转BGR
    color_vision_frame = cv2.cvtColor(color_vision_frame, cv2.COLOR_LAB2BGR)
    color_vision_frame = (color_vision_frame * 255.0).round().astype(numpy.uint8)
    return color_vision_frame


def prepare_temp_frame(temp_vision_frame: VisionFrame) -> VisionFrame:
    model_size = MODELS.get("ddcolor").get("size")
    # GBR转GRAY，再GRAY转RGB，去掉照片颜色
    temp_vision_frame = cv2.cvtColor(temp_vision_frame, cv2.COLOR_BGR2GRAY)
    temp_vision_frame = cv2.cvtColor(temp_vision_frame, cv2.COLOR_GRAY2RGB)
    # 预处理
    temp_vision_frame = (temp_vision_frame / 255.0).astype(numpy.float32)
    # 转为LAB，只需要L通道，即明度通道
    temp_vision_frame = cv2.cvtColor(temp_vision_frame, cv2.COLOR_RGB2LAB)[:, :, :1]
    temp_vision_frame = numpy.concatenate(
        (
            temp_vision_frame,
            numpy.zeros_like(temp_vision_frame),
            numpy.zeros_like(temp_vision_frame),
        ),
        axis=-1,
    )
    # 再转为RGB，此时只剩下明度信息
    temp_vision_frame = cv2.cvtColor(temp_vision_frame, cv2.COLOR_LAB2RGB)
    # 调整大小
    temp_vision_frame = cv2.resize(temp_vision_frame, model_size)
    # 转为onnx运行格式
    temp_vision_frame = temp_vision_frame.transpose((2, 0, 1))
    temp_vision_frame = numpy.expand_dims(temp_vision_frame, axis=0).astype(
        numpy.float32
    )
    return temp_vision_frame
