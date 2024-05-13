from typing import List
from back_end.face_analyser import (
    calc_embedding,
    detect_gender_age,
    get_many_faces,
    detect_face,
    detect_face_landmark_68,
)
from back_end.face_helper import apply_nms
from mid_end.typing import Face
import cv2, numpy


def draw_face(image, face: Face):
    left = int(face.bounding_box[0])
    top = int(face.bounding_box[1])
    right = int(face.bounding_box[2])
    bottom = int(face.bounding_box[3])
    # 画框
    cv2.rectangle(
        image,
        (left, top),
        (right, bottom),
        (0, 0, 255),
        2,
    )
    # 画点
    face_landmark_68 = face.landmarks.get("68").astype(numpy.int32)
    for index in range(face_landmark_68.shape[0]):
        p1 = int(face_landmark_68[index][0])
        p2 = int(face_landmark_68[index][1])
        cv2.circle(
            image,
            (p1, p2),
            3,
            (0, 255, 0),
            -1,
        )
    # 画分数
    face_score_text = "landmark:" + str(round(face.scores.get("landmarker"), 2))
    left = left - 30
    top = top + 20
    cv2.putText(
        image,
        face_score_text,
        (left, top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    top = top + 20
    face_score_text = "position:" + str(round(face.scores.get("detector"), 2))
    cv2.putText(
        image,
        face_score_text,
        (left, top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    # 画年龄
    left = left + 30
    top = top + 20
    age = str(round(face.age, 2))
    cv2.putText(
        image,
        age,
        (left, top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 255, 0),
        2,
    )
    # 画性别
    face_gender_text = "female" if face.gender == 0 else "male"
    top = top + 20
    cv2.putText(
        image,
        face_gender_text,
        (left, top),
        cv2.FONT_HERSHEY_SIMPLEX,
        0.5,
        (0, 0, 255),
        2,
    )


def test_get_many_faces():
    image_path = r"input\7.jpg"
    image = cv2.imread(image_path)
    faces = get_many_faces(image)
    for face in faces:
        print(f"box width:{face.bounding_box[2]-face.bounding_box[0]}")
        print(f"box height:{face.bounding_box[3]-face.bounding_box[1]}")
        draw_face(image, face)
    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()


def test_detect_face():
    image_path = r"input\9.jpg"
    image = cv2.imread(image_path)
    bounding_box_list, face_landmark_5_list, score_list = detect_face(image)
    sort_indices = numpy.argsort(-numpy.array(score_list))
    bounding_box_list = [bounding_box_list[i] for i in sort_indices]
    face_landmark_5_list = [face_landmark_5_list[i] for i in sort_indices]
    score_list = [score_list[i] for i in sort_indices]
    # 使用nms得到不重合的人脸范围
    keep_indices = apply_nms(bounding_box_list)
    for index in keep_indices:
        left = int(bounding_box_list[index][0])
        top = int(bounding_box_list[index][1])
        right = int(bounding_box_list[index][2])
        bottom = int(bounding_box_list[index][3])
        # 画框
        cv2.rectangle(
            image,
            (left, top),
            (right, bottom),
            (0, 0, 255),
            2,
        )

    cv2.imshow("Image", image)
    cv2.waitKey(0)
    cv2.destroyAllWindows()

    pass


def test_detect_landmar():
    image_path = r"input\3.jpg"
    image = cv2.imread(image_path)
    bounding_box_list, face_landmark_5_list, score_list = detect_face(image)
    sort_indices = numpy.argsort(-numpy.array(score_list))
    bounding_box_list = [bounding_box_list[i] for i in sort_indices]
    face_landmark_5_list = [face_landmark_5_list[i] for i in sort_indices]
    score_list = [score_list[i] for i in sort_indices]
    # 使用nms得到不重合的人脸范围
    keep_indices = apply_nms(bounding_box_list)
    face_landmark_68, face_alndmark_68_score = detect_face_landmark_68(
        image, bounding_box_list[keep_indices[0]]
    )
    print("")
    pass


def test_calc_embedding():
    image_path = r"input\3.jpg"
    image = cv2.imread(image_path)
    bounding_box_list, face_landmark_5_list, score_list = detect_face(image)
    sort_indices = numpy.argsort(-numpy.array(score_list))
    bounding_box_list = [bounding_box_list[i] for i in sort_indices]
    face_landmark_5_list = [face_landmark_5_list[i] for i in sort_indices]
    score_list = [score_list[i] for i in sort_indices]
    # 使用nms得到不重合的人脸范围
    keep_indices = apply_nms(bounding_box_list)
    embedding, normed_embedding = calc_embedding(
        image, face_landmark_5_list[keep_indices[0]]
    )
    print("")
    pass


def test_detect_gender_age():
    image_path = r"input\3.jpg"
    image = cv2.imread(image_path)
    bounding_box_list, face_landmark_5_list, score_list = detect_face(image)
    sort_indices = numpy.argsort(-numpy.array(score_list))
    bounding_box_list = [bounding_box_list[i] for i in sort_indices]
    face_landmark_5_list = [face_landmark_5_list[i] for i in sort_indices]
    score_list = [score_list[i] for i in sort_indices]
    # 使用nms得到不重合的人脸范围
    keep_indices = apply_nms(bounding_box_list)
    gender, age = detect_gender_age(image, bounding_box_list[keep_indices[0]])
    print("")


test_get_many_faces()
