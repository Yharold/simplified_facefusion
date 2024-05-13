from typing import Any, List, Optional, Tuple

from altair import Literal
from back_end.face_analyser import get_many_faces
from front_end.core import get_ui_component
import gradio as gr
from mid_end.core import conditional_append_reference_faces
from mid_end.face_store import (
    append_reference_faces,
    clear_reference_faces,
    clear_static_faces,
    get_reference_faces,
)
from mid_end.typing import VisionFrame
from mid_end.vision import (
    count_video_frame_number,
    get_temp_frame,
    is_image,
    is_video,
    normalize_frame_color,
    read_static_image,
)
from mid_end import globals


REFERENCE_GALLERY: gr.Gallery = None
REFERENCE_SLIDER: gr.Slider = None
REFERENCE_DROPDOWN: gr.Dropdown = None
REFERENCE_BUTTON: gr.Button = None


def render() -> None:
    global REFERENCE_GALLERY, REFERENCE_SLIDER, REFERENCE_DROPDOWN, REFERENCE_BUTTON
    REFERENCE_GALLERY = gr.Gallery(
        show_label=False,
        visible=False,
        columns=10,
        object_fit="cover",
        allow_preview=False,
        height=150,
    )
    with gr.Row():
        REFERENCE_SLIDER = gr.Slider(show_label=False, visible=False, scale=4)
        REFERENCE_DROPDOWN = gr.Dropdown(
            label="选择要替换的人物", visible=False, scale=1
        )
        REFERENCE_BUTTON = gr.Button(
            value="确认", visible=False, scale=1, variant="primary"
        )


def listen() -> None:
    target_image = get_ui_component("target_image")
    target_video = get_ui_component("target_video")
    # 决定组件是否显示
    target_image.change(
        update,
        outputs=[
            REFERENCE_GALLERY,
            REFERENCE_SLIDER,
            REFERENCE_DROPDOWN,
            REFERENCE_BUTTON,
        ],
    )
    target_video.change(
        update,
        outputs=[
            REFERENCE_GALLERY,
            REFERENCE_SLIDER,
            REFERENCE_DROPDOWN,
            REFERENCE_BUTTON,
        ],
    )
    # gallery更新dropdown一定更新
    # slider更新后gallery更新
    REFERENCE_SLIDER.release(
        update_reference_frame,
        inputs=REFERENCE_SLIDER,
        outputs=[REFERENCE_GALLERY, REFERENCE_DROPDOWN],
    )
    # button用来确认reference，更新gallery以便于让人确认，
    REFERENCE_BUTTON.click(
        update_reference_face,
        inputs=[REFERENCE_GALLERY, REFERENCE_DROPDOWN],
        outputs=[REFERENCE_GALLERY, REFERENCE_DROPDOWN],
    )


def update_reference_face(reference_frame: List[Any], index: str) -> gr.Gallery:
    index = int(index)
    if reference_frame and index is not None:
        conditional_append_reference_faces(reference_frame[index]["name"])
        temp_frame = read_static_image(reference_frame[index]["name"])
        return gr.Gallery(visible=True, value=[temp_frame]), gr.Dropdown(
            choices=[0], value=0, visible=True
        )
    return gr.Gallery(visible=True), gr.Dropdown(choices=None, value=None, visible=True)


def update_reference_frame(
    reference_number: int,
) -> Tuple[gr.Gallery, gr.Dropdown]:
    if is_video(globals.target_path):
        globals.reference_frame_number = reference_number
        temp_frame = get_temp_frame(globals.target_path, globals.reference_frame_number)
        reference_image_faces = get_reference_frame_faces(temp_frame)
        if reference_image_faces:
            dropdown_choice = [i for i in range(len(reference_image_faces))]
            dropdown_value = 0
            gallery_label = [str(i) for i in range(len(reference_image_faces))]
        else:
            dropdown_choice = None
            dropdown_value = None
            gallery_label = None
        return (
            gr.Gallery(visible=True, value=reference_image_faces, label=gallery_label),
            gr.Dropdown(choices=dropdown_choice, value=dropdown_value, visible=True),
        )
    return gr.Gallery(visible=True, value=None), gr.Dropdown(
        visible=True, choices=None, value=None
    )


# 显示需要的组件
def update() -> Tuple[gr.Gallery, gr.Slider, gr.Dropdown, gr.Button]:
    if globals.target_path and globals.frame_processor == "face_swapper":
        if is_image(globals.target_path):
            frame = read_static_image(globals.target_path)
            reference_image_faces = get_reference_frame_faces(frame)
            # 如果是图片，那滑块就不显示
            slider_visible = False
            slider_value = 0
            slider_maximum = 1
        if is_video(globals.target_path):
            temp_frame = get_temp_frame(
                globals.target_path, globals.reference_frame_number
            )
            reference_image_faces = get_reference_frame_faces(temp_frame)
            # 如果是视频，那滑块就显示，用来切换帧
            slider_visible = True
            slider_value = globals.reference_frame_number
            slider_maximum = count_video_frame_number(globals.target_path)

        # 如果有人物，则显示对应值，没有人物，那就不显示
        if reference_image_faces:
            gallery_value = reference_image_faces
            gallery_label = [str(i) for i in range(len(reference_image_faces))]
            dropdown_value = 0
            dropdown_choices = [i for i in range(len(reference_image_faces))]
            dropdown_button_visible = True
        else:
            gallery_value = None
            gallery_label = None
            dropdown_value = None
            dropdown_choices = None
            dropdown_button_visible = False
        return (
            gr.Gallery(visible=True, value=gallery_value, label=gallery_label),
            gr.Slider(
                visible=slider_visible, maximum=slider_maximum, value=slider_value
            ),
            gr.Dropdown(
                choices=dropdown_choices,
                visible=dropdown_button_visible,
                value=dropdown_value,
            ),
            gr.Button("确认", visible=dropdown_button_visible, variant="primary"),
        )
    return (
        gr.Gallery(visible=False),
        gr.Slider(visible=False),
        gr.Dropdown(visible=False),
        gr.Button("确认", visible=False),
    )


# 根据图片返回所有人物头像
def get_reference_frame_faces(temp_vision_frame: VisionFrame) -> Optional[VisionFrame]:
    gallery_vision_frames = []
    faces = get_many_faces(temp_vision_frame)
    if faces:
        for face in faces:
            start_x, start_y, end_x, end_y = map(int, face.bounding_box)
            padding_x = int((end_x - start_x) * 0.25)
            padding_y = int((end_y - start_y) * 0.25)
            start_x = max(0, start_x - padding_x)
            start_y = max(0, start_y - padding_y)
            end_x = max(0, end_x + padding_x)
            end_y = max(0, end_y + padding_y)
            crop_vision_frame = temp_vision_frame[start_y:end_y, start_x:end_x]
            crop_vision_frame = normalize_frame_color(crop_vision_frame)
            gallery_vision_frames.append(crop_vision_frame)
    return gallery_vision_frames
