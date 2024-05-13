import threading
from typing import Any, Dict, List, Optional, Tuple
from back_end.face_helper import (
    apply_nms,
    warp_face_by_face_landmark_5,
    warp_face_by_translation,
)
from mid_end.execution import get_providers
from mid_end.face_store import get_static_faces, set_static_faces
from mid_end.typing import (
    Embedding,
    Face,
    FaceLandmark68,
    FaceLandmarkSet,
    FaceMaskRegion,
    FaceScoreSet,
    FaceSet,
    ModelFrame,
    Score,
    VisionFrame,
    BoundingBox,
    FaceLandmark5,
    Mask,
)
from mid_end.vision import resize_frame_resolution
import numpy, time
import onnxruntime
import cv2
from mid_end.models import MODELS, face_detector_score, face_distance
from mid_end import process_manager

FACE_ANALYSER = None
THREAD_SEMAPHORE: threading.Semaphore = threading.Semaphore()
THREAD_LOCK: threading.Lock = threading.Lock()


def get_face_analyser() -> Any:
    global FACE_ANALYSER

    with THREAD_LOCK:
        while process_manager.is_checking():
            time.sleep(0.5)
        if FACE_ANALYSER is None:
            face_detector = onnxruntime.InferenceSession(
                MODELS.get("face_detector").get("path"), providers=get_providers()
            )
            face_recognizer = onnxruntime.InferenceSession(
                MODELS.get("face_recognizer").get("path"), providers=get_providers()
            )
            face_landmarker = onnxruntime.InferenceSession(
                MODELS.get("face_landmarker").get("path"), providers=get_providers()
            )
            gender_age = onnxruntime.InferenceSession(
                MODELS.get("gender_age").get("path"), providers=get_providers()
            )
            FACE_ANALYSER = {
                "face_detector": face_detector,
                "face_recognizer": face_recognizer,
                "face_landmarker": face_landmarker,
                "gender_age": gender_age,
            }
    return FACE_ANALYSER


def clear_face_analyser() -> None:
    global FACE_ANALYSER
    FACE_ANALYSER = None


# 探测一张图上的所有人的脸
def get_many_faces(vision_frame: VisionFrame) -> List[Face]:
    faces = []
    try:
        face_cache = get_static_faces(vision_frame)
        if face_cache:
            faces = face_cache
        else:
            # 探测人脸
            bounding_box_list, face_landmark_5_list, score_list = detect_face(
                vision_frame
            )
            # 按分值排序,numpy.argsort不是排序后的列表，而是排序后的索引
            sort_indices = numpy.argsort(-numpy.array(score_list))
            bounding_box_list = [bounding_box_list[i] for i in sort_indices]
            face_landmark_5_list = [face_landmark_5_list[i] for i in sort_indices]
            score_list = [score_list[i] for i in sort_indices]
            # 使用nms得到不重合的人脸范围
            keep_indices = apply_nms(bounding_box_list)
            # 计算每张人脸的数据
            for idx in keep_indices:
                # 计算face_landmark_68
                face_landmark_68, face_alndmark_68_score = detect_face_landmark_68(
                    vision_frame, bounding_box_list[idx]
                )
                # 计算embedding
                embedding, normed_embedding = calc_embedding(
                    vision_frame, face_landmark_5_list[idx]
                )
                # 计算gender,age
                gender, age = detect_gender_age(vision_frame, bounding_box_list[idx])
                landmarks: FaceLandmarkSet = {
                    "5": face_landmark_5_list[idx],
                    "68": face_landmark_68,
                }
                scores: FaceScoreSet = {
                    "detector": score_list[idx],
                    "landmarker": face_alndmark_68_score,
                }
                faces.append(
                    Face(
                        bounding_box=bounding_box_list[idx],
                        landmarks=landmarks,
                        scores=scores,
                        embedding=embedding,
                        normed_embedding=normed_embedding,
                        gender=gender,
                        age=age,
                    )
                )
            if faces:
                set_static_faces(vision_frame, faces)
    except (AttributeError, ValueError):
        pass

    return faces


# 得到指定的人脸
def get_one_face(vision_frame: VisionFrame, position: int = 0) -> Optional[Face]:
    many_faces = get_many_faces(vision_frame)
    if many_faces:
        try:
            return many_faces[position]
        except IndexError:
            return many_faces[-1]
    return None


def get_average_face(
    vision_frames: List[VisionFrame], position: int = 0
) -> Optional[Face]:
    average_face = None
    faces = []
    embedding_list = []
    normed_embedding_list = []
    for vision_frame in vision_frames:
        face = get_one_face(vision_frame)
        if face:
            faces.append(face)
            embedding_list.append(face.embedding)
            normed_embedding_list.append(face.normed_embedding)
    if faces:
        first_face = faces[0]
        average_face = Face(
            bounding_box=first_face.bounding_box,
            landmarks=first_face.landmarks,
            scores=first_face.scores,
            embedding=numpy.mean(embedding_list, axis=0),
            normed_embedding=numpy.mean(normed_embedding_list, axis=0),
            gender=first_face.gender,
            age=first_face.age,
        )
    return average_face


# 得到最接近的参照人脸
def find_similar_faces(
    reference_faces: FaceSet, vision_frame: VisionFrame
) -> List[Face]:

    similar_faces: List[Face] = []
    many_faces = get_many_faces(vision_frame)
    if reference_faces:
        for reference_set in reference_faces:
            if not similar_faces:
                for reference_face in reference_faces[reference_set]:
                    for face in many_faces:
                        if hasattr(face, "normed_embedding") and hasattr(
                            reference_face, "normed_embedding"
                        ):
                            result = 1 - numpy.dot(
                                face.normed_embedding, reference_face.normed_embedding
                            )
                        else:
                            result = 0
                        if result < face_distance:
                            similar_faces.append(face)
    return similar_faces


def detect_face_landmark_68(
    vision_frame: VisionFrame, bounding_box: BoundingBox
) -> Tuple[FaceLandmark68, Score]:
    face_landmarker = get_face_analyser().get("face_landmarker")
    # 接下来就是对图像进行处理
    # 目的是从原图vision_frame裁剪下大小为（256,256）的人脸图像，所以得得到缩放和中心点坐标，
    # 无论bounding_box多大，都缩放为195*195大小，所以得到了scale
    scale = 195 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max()
    # 位移量
    translation = (256 - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
    # 裁剪下人脸crop_vision_frame，并得到仿射变换矩阵affine_matrix
    crop_vision_frame, affine_matrix = warp_face_by_translation(
        vision_frame, translation, scale, (256, 256)
    )
    # 处理图像数据
    crop_vision_frame = prepare_landmark_68(crop_vision_frame)
    # 运行模型
    face_landmark_68, face_heatmap = face_landmarker.run(
        None, {face_landmarker.get_inputs()[0].name: crop_vision_frame}
    )
    # 处理模型数据,本来还是想写个函数，但face_heatmap这种数据不好定义类型
    # 所以就放弃了，直接写处理过程吧
    face_landmark_68 = face_landmark_68[:, :, :2][0] / 64
    face_landmark_68 = face_landmark_68.reshape(1, -1, 2) * 256
    face_landmark_68 = cv2.transform(
        face_landmark_68, cv2.invertAffineTransform(affine_matrix)
    )
    face_landmark_68 = face_landmark_68.reshape(-1, 2)
    face_landmark_68_score = numpy.amax(face_heatmap, axis=(2, 3))
    face_landmark_68_score = numpy.mean(face_landmark_68_score)
    return face_landmark_68, face_landmark_68_score


def prepare_landmark_68(crop_vision_frame: VisionFrame) -> ModelFrame:
    # 将 crop_vision_frame 图像从 RGB 颜色空间转换到 Lab 颜色空间
    crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_RGB2Lab)
    # 如果平均亮度值小于 30，需要进行亮度增强
    if numpy.mean(crop_vision_frame[:, :, 0]) < 30:
        # 对比度受限的自适应直方图均衡化（CLAHE）来增强亮度
        crop_vision_frame[:, :, 0] = cv2.createCLAHE(clipLimit=2).apply(
            crop_vision_frame[:, :, 0]
        )
    # 从 Lab 颜色空间转换回 RGB 颜色空间
    crop_vision_frame = cv2.cvtColor(crop_vision_frame, cv2.COLOR_Lab2RGB)
    crop_vision_frame = numpy.expand_dims(
        crop_vision_frame.transpose(2, 0, 1) / 255.0, axis=0
    ).astype(numpy.float32)
    return crop_vision_frame


def calc_embedding(
    vision_frame: VisionFrame, face_landmark_5: FaceLandmark5
) -> Tuple[Embedding, Embedding]:
    face_recognizer = get_face_analyser().get("face_recognizer")
    # 得到裁剪图像
    crop_vision_frame, affine_matrix = warp_face_by_face_landmark_5(
        vision_frame,
        face_landmark_5,
        MODELS.get("face_recognizer").get("template"),
        MODELS.get("face_recognizer").get("size"),
    )
    # 预处理数据，变为模型数据
    crop_vision_frame = crop_vision_frame / 127.5 - 1
    crop_vision_frame = (
        crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
    )
    crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0)
    embedding = face_recognizer.run(
        None, {face_recognizer.get_inputs()[0].name: crop_vision_frame}
    )[0]
    embedding = embedding.ravel()
    normed_embedding = embedding / numpy.linalg.norm(embedding)
    return embedding, normed_embedding


def detect_gender_age(
    vision_frame: VisionFrame, bounding_box: BoundingBox
) -> Tuple[int, int]:
    gender_age = get_face_analyser().get("gender_age")
    scale = 64 / numpy.subtract(bounding_box[2:], bounding_box[:2]).max()
    # 位移量
    translation = (96 - numpy.add(bounding_box[2:], bounding_box[:2]) * scale) * 0.5
    # 裁剪下人脸crop_vision_frame，并得到仿射变换矩阵affine_matrix
    crop_vision_frame, affine_matrix = warp_face_by_translation(
        vision_frame, translation, scale, (96, 96)
    )
    crop_vision_frame = (
        crop_vision_frame[:, :, ::-1].transpose(2, 0, 1).astype(numpy.float32)
    )
    crop_vision_frame = numpy.expand_dims(crop_vision_frame, axis=0)
    prediction = gender_age.run(
        None, {gender_age.get_inputs()[0].name: crop_vision_frame}
    )[0][0]
    gender = int(numpy.argmax(prediction[:2]))
    age = int(numpy.round(prediction[2] * 100))
    return gender, age


def detect_face(
    vision_frame: VisionFrame,
) -> Tuple[List[BoundingBox], List[FaceLandmark5], List[Score]]:
    # 载入模型
    face_detector = get_face_analyser().get("face_detector")
    face_detector_size = MODELS.get("face_detector").get("face_detector_size")

    # 将输入图片包括长宽比的情况下重建分辨率，可以比预设的小，不能比预设的大
    temp_vision_frame = resize_frame_resolution(vision_frame, face_detector_size)
    ratio_height = vision_frame.shape[0] / temp_vision_frame.shape[0]
    ratio_width = vision_frame.shape[1] / temp_vision_frame.shape[1]
    bounding_box_list = []
    face_landmark_5_list = []
    score_list = []
    # 处理数据，此时还是frame格式，即(height,width,3)，要改为模型的格式(1,3,height,width)
    temp_vision_frame = prepare_detect_frame(temp_vision_frame, face_detector_size)
    with THREAD_SEMAPHORE:
        # detections(1,20,length),bounding_box,landmark_5,score都在这里面
        detections = face_detector.run(
            None, {face_detector.get_inputs()[0].name: temp_vision_frame}
        )
    detections = numpy.squeeze(detections).T
    bounding_box_raw, score_raw, face_landmark_5_raw = numpy.split(
        detections, [4, 5], axis=1
    )
    keep_indices = numpy.where(score_raw > face_detector_score)[0]
    if keep_indices.any():
        bounding_box_raw, face_landmark_5_raw, score_raw = (
            bounding_box_raw[keep_indices],
            face_landmark_5_raw[keep_indices],
            score_raw[keep_indices],
        )
        for bounding_box in bounding_box_raw:
            bounding_box_list.append(
                numpy.array(
                    [
                        (bounding_box[0] - bounding_box[2] / 2) * ratio_width,
                        (bounding_box[1] - bounding_box[3] / 2) * ratio_height,
                        (bounding_box[0] + bounding_box[2] / 2) * ratio_width,
                        (bounding_box[1] + bounding_box[3] / 2) * ratio_height,
                    ]
                )
            )
        face_landmark_5_raw[:, 0::3] = (face_landmark_5_raw[:, 0::3]) * ratio_width
        face_landmark_5_raw[:, 1::3] = (face_landmark_5_raw[:, 1::3]) * ratio_height
        for face_landmark_5 in face_landmark_5_raw:
            face_landmark_5_list.append(
                numpy.array(face_landmark_5.reshape(-1, 3)[:, :2])
            )
        score_list = score_raw.ravel().tolist()
    return bounding_box_list, face_landmark_5_list, score_list


# 原始这里返回还是VisionFrame,但我觉得两个数据格式变化了还是区分开更好一些
# temp_vision_frame(height,width,3)
# return (1,3,detector_height,detector_width)
def prepare_detect_frame(
    temp_vision_frame: VisionFrame, face_detector_size: Tuple[int, int]
) -> ModelFrame:
    # 创建符合大小的空数据
    model_frame = numpy.zeros((face_detector_size[0], face_detector_size[1], 3))
    # 复制数据
    model_frame[: temp_vision_frame.shape[0], : temp_vision_frame.shape[1], :] = (
        temp_vision_frame
    )
    # 缩放到（0,1）之间
    model_frame = (model_frame - 127.5) / 128.0
    # 扩维，调整维度顺序，类型改为float
    model_frame = numpy.expand_dims(model_frame.transpose(2, 0, 1), axis=0).astype(
        numpy.float32
    )
    return model_frame
