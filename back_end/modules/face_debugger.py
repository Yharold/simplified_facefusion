from typing import List
from back_end.face_analyser import (
    clear_face_analyser,
    find_similar_faces,
    get_many_faces,
)
from back_end.face_helper import warp_face_by_face_landmark_5
from back_end.face_parser import clear_face_parser, create_mask
import cv2, numpy
from mid_end.face_store import get_reference_faces
from mid_end.typing import (
    Face,
    FaceDebuggerInputs,
    QueuePayload,
    UpdateProgress,
    VisionFrame,
)
from mid_end.vision import read_image, read_static_image, write_image
from mid_end import process_manager
from back_end import core

"""
主要是展示检测到的面部和使用模型探测面部区域
"""


def get_frame_processor() -> None:
    pass


def clear_frame_processor() -> None:
    pass


def pre_process() -> bool:
    return True


def post_process() -> None:
    read_static_image.cache_clear()
    clear_frame_processor()
    clear_face_analyser()
    clear_face_parser()


def get_reference_frame():
    pass


def process_frame(inputs: FaceDebuggerInputs) -> VisionFrame:
    target_vision_frame = inputs.get("target_vision_frame")
    faces = get_many_faces(target_vision_frame)
    if faces:
        for face in faces:
            target_vision_frame = debug_face(face, target_vision_frame)
    return target_vision_frame


def process_frames(
    source_path: List[str],
    queue_payloads: List[QueuePayload],
    update_progress: UpdateProgress,
) -> None:
    # reference_faces = get_reference_faces()
    for queue_payload in process_manager.manage(queue_payloads):
        target_path = queue_payload["frame_path"]
        target_vision_frame = read_image(target_path)
        output_vision_frame = process_frame(
            {
                "target_vision_frame": target_vision_frame,
            }
        )
        # output_vision_frame = process_frame(
        #     {
        #         "reference_faces": reference_faces,
        #         "target_vision_frame": target_vision_frame,
        #     }
        # )
        write_image(target_path, output_vision_frame)
        update_progress(1)


def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
    target_vision_frame = read_static_image(target_path)
    output_vision_frame = process_frame(
        {
            "target_vision_frame": target_vision_frame,
        }
    )
    write_image(output_path, output_vision_frame)


# def process_image(source_paths: List[str], target_path: str, output_path: str) -> None:
#     reference_faces = get_reference_faces()
#     target_vision_frame = read_static_image(target_path)
#     output_vision_frame = process_frame(
#         {
#             "reference_faces": reference_faces,
#             "target_vision_frame": target_vision_frame,
#         }
#     )
#     write_image(output_path, output_vision_frame)


def process_video(source_paths: List[str], temp_frame_paths: List[str]) -> None:
    core.multi_process_frames(source_paths, temp_frame_paths, process_frames)


# 展示探测结果
def debug_face(target_face: Face, temp_vision_frame: VisionFrame) -> VisionFrame:
    temp_vision_frame = temp_vision_frame.copy()
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
        temp_vision_frame,
        target_face.landmarks.get("5"),
        "arcface_128_v2",
        (512, 512),
    )
    # 计算面部区域遮罩,绘制遮罩区域
    crop_mask = create_mask(crop_vision_frame)
    crop_mask = (crop_mask.clip(0, 1) * 255).astype(numpy.uint8)
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    inverse_vision_frame = cv2.warpAffine(
        crop_mask, inverse_matrix, temp_vision_frame.shape[:2][::-1]
    )
    inverse_vision_frame = cv2.threshold(
        inverse_vision_frame, 100, 255, cv2.THRESH_BINARY
    )[1]
    inverse_vision_frame[inverse_vision_frame > 0] = 255
    inverse_contours = cv2.findContours(
        inverse_vision_frame, cv2.RETR_LIST, cv2.CHAIN_APPROX_NONE
    )[0]

    primary_color = (0, 0, 255)  # 红
    secondary_color = (0, 255, 0)  # 绿
    tertiary_color = (255, 255, 0)  # 灰白
    bounding_box = target_face.bounding_box.astype(numpy.int32)
    left = bounding_box[0]
    top = bounding_box[1]
    right = bounding_box[2]
    bottom = bounding_box[3]
    # 框
    cv2.rectangle(temp_vision_frame, (left, top), (right, bottom), primary_color, 2)
    # 遮罩
    cv2.drawContours(
        temp_vision_frame,
        inverse_contours,
        -1,
        tertiary_color,
        2,
    )
    # 绘制landmark_68
    landmark_68 = target_face.landmarks.get("68")
    for index in range(landmark_68.shape[0]):
        p = landmark_68[index].astype(numpy.int32)
        cv2.circle(
            temp_vision_frame,
            (p[0], p[1]),
            3,
            primary_color,
            -1,
        )
    # detector_score_text = "detector:" + str(
    #     round(target_face.scores.get("detector"), 2)
    # )
    # left = left - 20
    # top = top - 10
    # cv2.putText(
    #     temp_vision_frame,
    #     detector_score_text,
    #     (left, top),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     secondary_color,
    #     2,
    # )
    # landmarker_score_text = "landmarker:" + str(
    #     round(target_face.scores.get("landmarker"), 2)
    # )
    # right = right - 30
    # cv2.putText(
    #     temp_vision_frame,
    #     landmarker_score_text,
    #     (right, top),
    #     cv2.FONT_HERSHEY_SIMPLEX,
    #     0.5,
    #     secondary_color,
    #     2,
    # )
    face_age_text = str(target_face.age)
    top = top + 10
    cv2.putText(
        temp_vision_frame,
        face_age_text,
        (left, top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        secondary_color,
        2,
    )
    face_gender_text = "male" if target_face.gender == 1 else "female"
    top = top + 20
    cv2.putText(
        temp_vision_frame,
        face_gender_text,
        (left, top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        secondary_color,
        2,
    )
    return temp_vision_frame
