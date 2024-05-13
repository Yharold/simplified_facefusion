from typing import List, Tuple
import cv2
from mid_end.typing import (
    BoundingBox,
    FaceLandmark5,
    Mask,
    Matrix,
    Translation,
    VisionFrame,
    WarpTemplate,
    WarpTemplateSet,
)
import numpy
from cv2.typing import Size

"""
这个文件是一些辅助函数，主要针对于有关Face数据
"""
WARP_TEMPLATES: WarpTemplateSet = {
    ## calc_embedding face_recognizer
    "arcface_112_v2": numpy.array(
        [
            [0.34191607, 0.46157411],
            [0.65653393, 0.45983393],
            [0.50022500, 0.64050536],
            [0.37097589, 0.82469196],
            [0.63151696, 0.82325089],
        ]
    ),
    ## inswapper
    "arcface_128_v2": numpy.array(
        [
            [0.36167656, 0.40387734],
            [0.63696719, 0.40235469],
            [0.50019687, 0.56044219],
            [0.38710391, 0.72160547],
            [0.61507734, 0.72034453],
        ]
    ),
    # codeformer
    "ffhq_512": numpy.array(
        [
            [0.37691676, 0.46864664],
            [0.62285697, 0.46912813],
            [0.50123859, 0.61331904],
            [0.39308822, 0.72541100],
            [0.61150205, 0.72490465],
        ]
    ),
}


def apply_nms(bounding_box_list: List[BoundingBox]) -> List[int]:
    keep_indices = []
    iou_threshold = 0.1
    dimension_list = numpy.reshape(bounding_box_list, (-1, 4))
    x1 = dimension_list[:, 0]
    y1 = dimension_list[:, 1]
    x2 = dimension_list[:, 2]
    y2 = dimension_list[:, 3]
    areas = (x2 - x1 + 1) * (y2 - y1 + 1)
    indices = numpy.arange(len(bounding_box_list))
    while indices.size > 0:
        index = indices[0]
        remain_indices = indices[1:]
        keep_indices.append(index)
        xx1 = numpy.maximum(x1[index], x1[remain_indices])
        yy1 = numpy.maximum(y1[index], y1[remain_indices])
        xx2 = numpy.minimum(x2[index], x2[remain_indices])
        yy2 = numpy.minimum(y2[index], y2[remain_indices])
        width = numpy.maximum(0, xx2 - xx1 + 1)
        height = numpy.maximum(0, yy2 - yy1 + 1)
        iou = width * height / (areas[index] + areas[remain_indices] - width * height)
        indices = indices[numpy.where(iou <= iou_threshold)[0] + 1]
    return keep_indices


def warp_face_by_translation(
    temp_vision_frame: VisionFrame,
    translation: Translation,
    scale: float,
    crop_size: Size,
) -> Tuple[VisionFrame, Matrix]:
    affine_matrix = numpy.array(
        [[scale, 0, translation[0]], [0, scale, translation[1]]]
    )
    crop_vision_frame = cv2.warpAffine(temp_vision_frame, affine_matrix, crop_size)
    return crop_vision_frame, affine_matrix


def estimate_matrix_by_face_landmark_5(
    face_landmark_5: FaceLandmark5, warp_template: WarpTemplate, crop_size: Size
) -> Matrix:
    normed_warp_template = WARP_TEMPLATES.get(warp_template) * crop_size
    affine_matrix = cv2.estimateAffinePartial2D(
        face_landmark_5,
        normed_warp_template,
        method=cv2.RANSAC,
        ransacReprojThreshold=100,
    )[0]
    return affine_matrix


def warp_face_by_face_landmark_5(
    vision_frame: VisionFrame,
    face_landmark_5: FaceLandmark5,
    warp_template: WarpTemplate,
    crop_size: Size,
) -> Tuple[VisionFrame, Matrix]:
    affine_matrix = estimate_matrix_by_face_landmark_5(
        face_landmark_5, warp_template, crop_size
    )
    crop_vision_frame = cv2.warpAffine(
        vision_frame, affine_matrix, crop_size, borderMode=cv2.BORDER_REPLICATE
    )
    return crop_vision_frame, affine_matrix


def paste_back(
    temp_vision_frame: VisionFrame,
    crop_vision_frame: VisionFrame,
    crop_mask: Mask,
    affine_matrix: Matrix,
) -> VisionFrame:
    # 求逆矩阵
    inverse_matrix = cv2.invertAffineTransform(affine_matrix)
    # 输出图像大小,temp_vision_frame(560,560,3)
    temp_size = temp_vision_frame.shape[:2][::-1]
    # 和输出大小匹配的mask,inverse_mask(560,560)
    inverse_mask = cv2.warpAffine(crop_mask, inverse_matrix, temp_size).clip(0, 1)
    # 将crop_vision_fram:(560,560,3)
    # cv2.BORDER_REPLICATE将边缘像素复制到空白地方
    inverse_vision_frame = cv2.warpAffine(
        crop_vision_frame, inverse_matrix, temp_size, borderMode=cv2.BORDER_REPLICATE
    )
    # 复制原始图像
    paste_vision_frame = temp_vision_frame.copy()
    # 最终图像是换脸后的图像和原始图像的加权求和，系数分别是mask和1-mask
    # 这里不用循环，是利用广播机制高效操作
    paste_vision_frame[:, :, 0] = (
        inverse_mask * inverse_vision_frame[:, :, 0]
        + (1 - inverse_mask) * temp_vision_frame[:, :, 0]
    )
    paste_vision_frame[:, :, 1] = (
        inverse_mask * inverse_vision_frame[:, :, 1]
        + (1 - inverse_mask) * temp_vision_frame[:, :, 1]
    )
    paste_vision_frame[:, :, 2] = (
        inverse_mask * inverse_vision_frame[:, :, 2]
        + (1 - inverse_mask) * temp_vision_frame[:, :, 2]
    )
    return paste_vision_frame
